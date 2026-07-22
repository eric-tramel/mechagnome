"""Immutable host kernel for persistent, dynamically authored tools."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mechagnome.bootstrap import BOOTSTRAP_TOOLS, CORE_NAMES, CORE_SCHEMAS

if TYPE_CHECKING:
    from mechagnome.model_provider import _BoundedModelProvider

JsonValue = Any

_SCHEMA_VERSION = 6
_SESSION_KINDS = frozenset({"generic", "conversation", "completion"})
_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_TOOLBOX_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


class ToolboxError(RuntimeError):
    """A structured error raised by the toolbox kernel."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable error observation."""
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "details": self.details,
            }
        }


@dataclass(frozen=True)
class InvocationScope:
    """Immutable toolbox selection and working directory for one call tree."""

    session_id: str
    toolbox_ids: tuple[str, ...]
    cwd: str


@dataclass
class _InvocationState:
    scope: InvocationScope
    max_depth: int
    max_calls: int
    model_provider: _BoundedModelProvider
    calls: int = 0

    @property
    def session_id(self) -> str:
        return self.scope.session_id


@dataclass(frozen=True)
class _ResolvedTool:
    id: int
    toolbox_id: str
    toolbox_name: str
    name: str
    version: int
    description: str
    input_schema: dict[str, Any]
    source: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ToolboxError(
            "not_json", "value is not JSON-serializable", reason=str(error)
        ) from error


class SessionAccess:
    """Bounded read-only session access exposed to every tool."""

    def __init__(self, kernel: Kernel, session_id: str) -> None:
        self._kernel = kernel
        self.id = session_id

    def list(self, limit: int = 20, cursor: int = 0) -> dict[str, Any]:
        """List saved sessions in reverse creation order."""
        return self._kernel.list_sessions(limit=limit, cursor=cursor)

    def read(self, session_id: str, after: int = 0, limit: int = 50) -> dict[str, Any]:
        """Read a page of events from a saved session."""
        return self._kernel.read_session(session_id, after=after, limit=limit)

    def current(self, after: int = 0, limit: int = 50) -> dict[str, Any]:
        """Read a page of events from the currently running session."""
        return self.read(self.id, after=after, limit=limit)

    def metadata(self, session_id: str | None = None) -> dict[str, Any]:
        """Return durable identity and lineage for one saved session."""
        return self._kernel.session_metadata(
            self.id if session_id is None else session_id
        )


class ToolContext:
    """The functional interface supplied to authored tools."""

    def __init__(
        self,
        kernel: Kernel,
        state: _InvocationState,
        call_id: str,
        depth: int,
        logical_slot: str | None,
    ) -> None:
        self._kernel = kernel
        self._state = state
        self._call_id = call_id
        self._depth = depth
        self._logical_slot = logical_slot
        self.caller_session_id = state.session_id
        self.sessions = SessionAccess(kernel, state.session_id)
        from mechagnome.model_provider import ToolModelProvider

        self.model_provider = ToolModelProvider(
            state.model_provider.for_origin(call_id)
        )

    @property
    def kernel(self) -> _KernelCapability:
        """Return the narrow capability belonging to a distinguished core slot."""
        if self._logical_slot is None:
            raise ToolboxError(
                "capability_denied",
                "ordinary tools do not receive a core kernel capability",
            )
        return _KernelCapability(self)

    def call_tool(
        self, name: str, args: dict[str, Any], version: int | None = None
    ) -> JsonValue:
        """Invoke a tool through the snapshotted editable dispatcher."""
        envelope: dict[str, Any] = {"name": name, "args": args}
        if version is not None:
            envelope["version"] = version
        return self._kernel._invoke(
            "call_tool",
            envelope,
            state=self._state,
            parent_call_id=self._call_id,
            depth=self._depth + 1,
        )


class _KernelCapability:
    """Low-level operation granted according to the logical core binding slot."""

    def __init__(self, context: ToolContext) -> None:
        self._context = context

    def _require(self, slot: str) -> None:
        if self._context._logical_slot != slot:
            raise ToolboxError(
                "capability_denied",
                f"{self._context._logical_slot} cannot use the {slot} capability",
            )

    def catalog(self, include_core: bool = True) -> list[dict[str, Any]]:
        """Return effective metadata to the search implementation."""
        self._require("search_tools")
        return self._context._kernel.catalog(
            include_core=include_core, scope=self._context._state.scope
        )

    def view_tool(self, name: str, version: int | None = None) -> dict[str, Any]:
        """View a tool through the inspection capability."""
        self._require("view_tool")
        return self._context._kernel.view_tool(
            name, version=version, scope=self._context._state.scope
        )

    def write_tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        source: str,
        base_version: int | None = None,
    ) -> dict[str, Any]:
        """Store source through the write capability."""
        self._require("write_tool")
        return self._context._kernel.write_tool(
            name=name,
            description=description,
            input_schema=input_schema,
            source=source,
            base_version=base_version,
            session_id=self._context._state.session_id,
            parent_call_id=self._context._call_id,
            scope=self._context._state.scope,
        )

    def execute(
        self, name: str, args: dict[str, Any], version: int | None = None
    ) -> JsonValue:
        """Bottom out recursive dispatch through the call capability."""
        self._require("call_tool")
        return self._context._kernel._invoke(
            name,
            args,
            state=self._context._state,
            parent_call_id=self._context._call_id,
            depth=self._context._depth + 1,
            version=version,
        )


class Kernel:
    """Persistent host substrate below the five editable core operations."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        max_depth: int = 12,
        max_calls: int = 100,
        cwd: str | Path | None = None,
    ) -> None:
        self.cwd = self._canonical_cwd(cwd or Path.cwd())
        self.db_path = Path(db_path).expanduser().resolve()
        parent_existed = self.db_path.parent.exists()
        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not parent_existed:
            self.db_path.parent.chmod(0o700)
        try:
            descriptor = os.open(
                self.db_path,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
            )
        except FileExistsError:
            self.db_path.chmod(0o600)
        else:
            os.close(descriptor)
        self.max_depth = max_depth
        self.max_calls = max_calls
        self._initialize()

    @staticmethod
    def _canonical_cwd(path: str | Path) -> str:
        try:
            resolved = Path(path).expanduser().resolve(strict=True)
        except OSError as error:
            raise ToolboxError("invalid_cwd", f"invalid directory: {path}") from error
        if not resolved.is_dir():
            raise ToolboxError("invalid_cwd", f"not a directory: {resolved}")
        return str(resolved)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _tables(connection: sqlite3.Connection) -> set[str]:
        return {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                tables = self._tables(connection)
                if "tool_versions" not in tables:
                    self._create_schema(connection)
                    self._ensure_cwd_default(connection, self.cwd)
                elif "tool_lineages" not in tables:
                    self._migrate_legacy(connection)
                else:
                    version = int(
                        connection.execute("PRAGMA user_version").fetchone()[0]
                    )
                    if version == 2:
                        version = 3
                    if version == 3:
                        self._migrate_v3(connection)
                        version = 4
                    if version == 4:
                        self._migrate_v4(connection)
                        version = 5
                    if version == 5:
                        self._migrate_v5(connection)
                        version = 6
                    if version != _SCHEMA_VERSION:
                        raise ToolboxError(
                            "unsupported_schema",
                            f"unsupported toolbox schema version: {version}",
                        )
                self._refresh_core_defaults(connection)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _migrate_v3(connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(sessions)")
        }
        if "parent_session_id" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN parent_session_id TEXT "
                "REFERENCES sessions(id)"
            )
        if "kind" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN kind TEXT NOT NULL DEFAULT 'generic' "
                "CHECK(kind IN ('generic', 'conversation', 'completion'))"
            )
        if "origin_call_id" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN origin_call_id TEXT")
        Kernel._create_session_indexes_and_triggers(connection)

    @staticmethod
    def _migrate_v4(connection: sqlite3.Connection) -> None:
        """Rename the tool-inspection core slot while preserving its versions."""
        toolbox_rows = connection.execute("SELECT id FROM toolboxes").fetchall()
        for toolbox_row in toolbox_rows:
            toolbox_id = str(toolbox_row["id"])
            old = connection.execute(
                "SELECT id FROM tool_lineages WHERE toolbox_id = ? AND name = ?",
                (toolbox_id, "read_tool_source"),
            ).fetchone()
            if old is None:
                continue
            existing = connection.execute(
                "SELECT id FROM tool_lineages WHERE toolbox_id = ? AND name = ?",
                (toolbox_id, "view_tool"),
            ).fetchone()
            if existing is not None:
                suffix = 1
                legacy_name = "view_tool_legacy"
                while (
                    connection.execute(
                        "SELECT 1 FROM tool_lineages WHERE toolbox_id = ? AND name = ?",
                        (toolbox_id, legacy_name),
                    ).fetchone()
                    is not None
                ):
                    suffix += 1
                    legacy_name = f"view_tool_legacy_{suffix}"
                connection.execute(
                    "UPDATE tool_lineages SET name = ? WHERE id = ?",
                    (legacy_name, int(existing["id"])),
                )
                connection.execute(
                    "UPDATE bindings SET name = ? WHERE toolbox_id = ? AND name = ?",
                    (legacy_name, toolbox_id, "view_tool"),
                )
            connection.execute(
                "UPDATE tool_lineages SET name = ? WHERE id = ?",
                ("view_tool", int(old["id"])),
            )
            connection.execute(
                "UPDATE bindings SET name = ? WHERE toolbox_id = ? AND name = ?",
                ("view_tool", toolbox_id, "read_tool_source"),
            )

    @staticmethod
    def _migrate_v5(connection: sqlite3.Connection) -> None:
        """Remove agent-authored ratings and comments from existing databases."""
        connection.execute("DROP TABLE IF EXISTS tool_feedback")

    @staticmethod
    def _create_session_indexes_and_triggers(
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute(
            "CREATE INDEX IF NOT EXISTS sessions_parent_created "
            "ON sessions(parent_session_id, created_at DESC)"
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS sessions_lineage_immutable
            BEFORE UPDATE OF parent_session_id, kind, origin_call_id ON sessions
            WHEN OLD.parent_session_id IS NOT NEW.parent_session_id
              OR OLD.kind IS NOT NEW.kind
              OR OLD.origin_call_id IS NOT NEW.origin_call_id
            BEGIN
                SELECT RAISE(ABORT, 'session lineage is immutable');
            END
            """
        )

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE toolboxes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE cwd_defaults (
                cwd TEXT PRIMARY KEY,
                toolbox_id TEXT NOT NULL UNIQUE REFERENCES toolboxes(id)
            )
            """,
            """
            CREATE TABLE tool_lineages (
                id INTEGER PRIMARY KEY,
                toolbox_id TEXT NOT NULL REFERENCES toolboxes(id),
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(toolbox_id, name)
            )
            """,
            """
            CREATE TABLE tool_versions (
                id INTEGER PRIMARY KEY,
                lineage_id INTEGER NOT NULL REFERENCES tool_lineages(id),
                version INTEGER NOT NULL,
                description TEXT NOT NULL,
                schema_json TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(lineage_id, version)
            )
            """,
            """
            CREATE TABLE bindings (
                toolbox_id TEXT NOT NULL REFERENCES toolboxes(id),
                name TEXT NOT NULL,
                tool_version_id INTEGER NOT NULL REFERENCES tool_versions(id),
                PRIMARY KEY(toolbox_id, name)
            )
            """,
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                cwd TEXT,
                parent_session_id TEXT REFERENCES sessions(id),
                kind TEXT NOT NULL DEFAULT 'generic'
                    CHECK(kind IN ('generic', 'conversation', 'completion')),
                origin_call_id TEXT,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE session_toolboxes (
                session_id TEXT NOT NULL REFERENCES sessions(id),
                position INTEGER NOT NULL,
                toolbox_id TEXT NOT NULL REFERENCES toolboxes(id),
                PRIMARY KEY(session_id, position),
                UNIQUE(session_id, toolbox_id)
            )
            """,
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                seq INTEGER NOT NULL,
                kind TEXT NOT NULL,
                call_id TEXT,
                parent_call_id TEXT,
                toolbox_id TEXT REFERENCES toolboxes(id),
                tool_name TEXT,
                tool_version INTEGER,
                tool_version_id INTEGER REFERENCES tool_versions(id),
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(session_id, seq)
            )
            """,
            "CREATE INDEX events_tool_version_id ON events(tool_version_id)",
            "CREATE INDEX events_session_seq ON events(session_id, seq)",
        )
        for statement in statements:
            connection.execute(statement)
        Kernel._create_session_indexes_and_triggers(connection)

    def _migrate_legacy(self, connection: sqlite3.Connection) -> None:
        for table in ("tool_versions", "bindings", "sessions", "events"):
            connection.execute(f"ALTER TABLE {table} RENAME TO legacy_{table}")
        self._create_schema(connection)
        toolbox_id = uuid.uuid4().hex
        connection.execute(
            "INSERT INTO toolboxes (id, name, created_at) VALUES (?, ?, ?)",
            (toolbox_id, "legacy", _now()),
        )
        connection.execute(
            "INSERT INTO metadata (key, value) VALUES ('legacy_fallback', ?)",
            (toolbox_id,),
        )
        names = connection.execute(
            "SELECT DISTINCT name FROM legacy_tool_versions ORDER BY name"
        ).fetchall()
        lineages: dict[str, int] = {}
        for row in names:
            cursor = connection.execute(
                """
                INSERT INTO tool_lineages (toolbox_id, name, created_at)
                VALUES (?, ?, ?)
                """,
                (toolbox_id, row["name"], _now()),
            )
            lineages[str(row["name"])] = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO sessions (id, cwd, created_at)
            SELECT id, NULL, created_at FROM legacy_sessions
            """
        )
        for row in connection.execute(
            "SELECT * FROM legacy_tool_versions ORDER BY id"
        ).fetchall():
            connection.execute(
                """
                INSERT INTO tool_versions (
                    id, lineage_id, version, description, schema_json, source,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    lineages[str(row["name"])],
                    row["version"],
                    row["description"],
                    row["schema_json"],
                    row["source"],
                    row["created_at"],
                ),
            )
        connection.execute(
            """
            INSERT INTO bindings (toolbox_id, name, tool_version_id)
            SELECT ?, name, tool_id FROM legacy_bindings
            """,
            (toolbox_id,),
        )
        connection.execute(
            """
            INSERT INTO session_toolboxes (session_id, position, toolbox_id)
            SELECT id, 0, ? FROM sessions
            """,
            (toolbox_id,),
        )
        legacy_events = connection.execute(
            "SELECT * FROM legacy_events ORDER BY id"
        ).fetchall()
        for row in legacy_events:
            version_id = None
            if row["tool_name"] is not None and row["tool_version"] is not None:
                version_row = connection.execute(
                    """
                    SELECT v.id
                    FROM tool_versions AS v
                    JOIN tool_lineages AS l ON l.id = v.lineage_id
                    WHERE l.toolbox_id = ? AND l.name = ? AND v.version = ?
                    """,
                    (toolbox_id, row["tool_name"], row["tool_version"]),
                ).fetchone()
                version_id = int(version_row["id"]) if version_row else None
            connection.execute(
                """
                INSERT INTO events (
                    id, session_id, seq, kind, call_id, parent_call_id,
                    toolbox_id, tool_name, tool_version, tool_version_id,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["session_id"],
                    row["seq"],
                    row["kind"],
                    row["call_id"],
                    row["parent_call_id"],
                    toolbox_id if row["tool_name"] is not None else None,
                    row["tool_name"],
                    row["tool_version"],
                    version_id,
                    row["payload_json"],
                    row["created_at"],
                ),
            )
        for table in (
            "legacy_bindings",
            "legacy_tool_versions",
            "legacy_events",
            "legacy_sessions",
        ):
            connection.execute(f"DROP TABLE {table}")
        self._seed_missing_core(connection, toolbox_id)

    def _seed_missing_core(
        self, connection: sqlite3.Connection, toolbox_id: str
    ) -> None:
        for tool in BOOTSTRAP_TOOLS:
            exists = connection.execute(
                """
                SELECT 1 FROM tool_lineages
                WHERE toolbox_id = ? AND name = ?
                """,
                (toolbox_id, tool.name),
            ).fetchone()
            if exists is not None:
                continue
            lineage = connection.execute(
                """
                INSERT INTO tool_lineages (toolbox_id, name, created_at)
                VALUES (?, ?, ?)
                """,
                (toolbox_id, tool.name, _now()),
            )
            version = connection.execute(
                """
                INSERT INTO tool_versions (
                    lineage_id, version, description, schema_json, source, created_at
                ) VALUES (?, 1, ?, ?, ?, ?)
                """,
                (
                    int(lineage.lastrowid),
                    tool.description,
                    _json(tool.input_schema),
                    tool.source,
                    _now(),
                ),
            )
            connection.execute(
                """
                INSERT INTO bindings (toolbox_id, name, tool_version_id)
                VALUES (?, ?, ?)
                """,
                (toolbox_id, tool.name, int(version.lastrowid)),
            )

    @staticmethod
    def _refresh_core_defaults(connection: sqlite3.Connection) -> None:
        """Synchronize every toolbox's core version 1 with shipped source."""
        for tool in BOOTSTRAP_TOOLS:
            connection.execute(
                """
                UPDATE tool_versions
                SET description = ?, schema_json = ?, source = ?
                WHERE version = 1 AND lineage_id IN (
                    SELECT id FROM tool_lineages WHERE name = ?
                )
                """,
                (
                    tool.description,
                    _json(tool.input_schema),
                    tool.source,
                    tool.name,
                ),
            )

    def _create_toolbox_connection(
        self,
        connection: sqlite3.Connection,
        name: str,
    ) -> str:
        if not isinstance(name, str) or _TOOLBOX_NAME.fullmatch(name) is None:
            raise ToolboxError(
                "invalid_toolbox_name", f"invalid toolbox name: {name!r}"
            )
        toolbox_id = uuid.uuid4().hex
        try:
            connection.execute(
                "INSERT INTO toolboxes (id, name, created_at) VALUES (?, ?, ?)",
                (toolbox_id, name, _now()),
            )
        except sqlite3.IntegrityError as error:
            raise ToolboxError(
                "toolbox_exists", f"toolbox already exists: {name}"
            ) from error
        self._seed_missing_core(connection, toolbox_id)
        return toolbox_id

    def _ensure_cwd_default(self, connection: sqlite3.Connection, cwd: str) -> str:
        row = connection.execute(
            "SELECT toolbox_id FROM cwd_defaults WHERE cwd = ?", (cwd,)
        ).fetchone()
        if row is not None:
            return str(row["toolbox_id"])
        legacy = connection.execute(
            "SELECT value FROM metadata WHERE key = 'legacy_fallback'"
        ).fetchone()
        if legacy is not None:
            return str(legacy["value"])
        digest = hashlib.sha256(cwd.encode()).hexdigest()[:8]
        base = re.sub(r"[^A-Za-z0-9_.-]", "-", Path(cwd).name) or "root"
        name = f"cwd-{base[:40]}-{digest}"
        toolbox_id = self._create_toolbox_connection(connection, name)
        connection.execute(
            "INSERT INTO cwd_defaults (cwd, toolbox_id) VALUES (?, ?)",
            (cwd, toolbox_id),
        )
        return toolbox_id

    def create_toolbox(
        self,
        name: str,
        *,
        cwd: str | Path | None = None,
    ) -> dict[str, Any]:
        """Create a named toolbox with independent bootstrap bindings."""
        canonical = self._canonical_cwd(cwd) if cwd is not None else None
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                toolbox_id = self._create_toolbox_connection(connection, name)
                if canonical is not None:
                    connection.execute(
                        """
                        INSERT INTO cwd_defaults (cwd, toolbox_id) VALUES (?, ?)
                        ON CONFLICT(cwd) DO UPDATE SET toolbox_id = excluded.toolbox_id
                        """,
                        (canonical, toolbox_id),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {"id": toolbox_id, "name": name, "cwd": canonical}

    def list_toolboxes(self) -> list[dict[str, Any]]:
        """List registered toolboxes and cwd-default status."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT t.id, t.name, d.cwd, t.created_at,
                       d.cwd = ? AS is_default
                FROM toolboxes AS t
                LEFT JOIN cwd_defaults AS d ON d.toolbox_id = t.id
                ORDER BY t.name
                """,
                (self.cwd,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "cwd": row["cwd"],
                "default": bool(row["is_default"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def rename_toolbox(self, name: str, new_name: str) -> dict[str, Any]:
        """Rename a toolbox without changing its identity or associations."""
        if not isinstance(new_name, str) or _TOOLBOX_NAME.fullmatch(new_name) is None:
            raise ToolboxError(
                "invalid_toolbox_name", f"invalid toolbox name: {new_name!r}"
            )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                toolbox_id = self._toolbox_id(connection, name)
                try:
                    connection.execute(
                        "UPDATE toolboxes SET name = ? WHERE id = ?",
                        (new_name, toolbox_id),
                    )
                except sqlite3.IntegrityError as error:
                    raise ToolboxError(
                        "toolbox_exists", f"toolbox already exists: {new_name}"
                    ) from error
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {"id": toolbox_id, "name": new_name}

    @staticmethod
    def _toolbox_id(connection: sqlite3.Connection, name: str) -> str:
        row = connection.execute(
            "SELECT id FROM toolboxes WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise ToolboxError("unknown_toolbox", f"unknown toolbox: {name}")
        return str(row["id"])

    def set_cwd_default(
        self, name: str, *, cwd: str | Path | None = None
    ) -> dict[str, Any]:
        """Associate one canonical cwd with a registered toolbox."""
        canonical = self._canonical_cwd(cwd or self.cwd)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                toolbox_id = self._toolbox_id(connection, name)
                connection.execute(
                    "DELETE FROM cwd_defaults WHERE toolbox_id = ?", (toolbox_id,)
                )
                connection.execute(
                    """
                    INSERT INTO cwd_defaults (cwd, toolbox_id) VALUES (?, ?)
                    ON CONFLICT(cwd) DO UPDATE SET toolbox_id = excluded.toolbox_id
                    """,
                    (canonical, toolbox_id),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {"cwd": canonical, "toolbox": name}

    def create_session(
        self,
        session_id: str | None = None,
        *,
        kind: str = "generic",
    ) -> str:
        """Create a durable root session with a cwd-default selection."""
        if kind not in _SESSION_KINDS:
            raise ToolboxError("invalid_session_kind", f"invalid session kind: {kind}")
        identifier = session_id or uuid.uuid4().hex
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                created = connection.execute(
                    """
                    INSERT OR IGNORE INTO sessions (id, cwd, kind, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (identifier, self.cwd, kind, _now()),
                ).rowcount
                selected = connection.execute(
                    "SELECT 1 FROM session_toolboxes WHERE session_id = ?",
                    (identifier,),
                ).fetchone()
                if selected is None:
                    default_id = self._ensure_cwd_default(connection, self.cwd)
                    connection.execute(
                        """
                        INSERT INTO session_toolboxes (session_id, position, toolbox_id)
                        VALUES (?, 0, ?)
                        """,
                        (identifier, default_id),
                    )
                if not created:
                    exists = connection.execute(
                        "SELECT 1 FROM sessions WHERE id = ?", (identifier,)
                    ).fetchone()
                    if exists is None:
                        raise ToolboxError(
                            "unknown_session", f"unknown session: {identifier}"
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return identifier

    def create_child_session(
        self,
        scope: InvocationScope,
        *,
        kind: str,
        origin_call_id: str | None = None,
    ) -> str:
        """Create a host-identified child inheriting one frozen invocation scope."""
        if kind not in _SESSION_KINDS:
            raise ToolboxError("invalid_session_kind", f"invalid child kind: {kind}")
        identifier = uuid.uuid4().hex
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                parent = connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ?", (scope.session_id,)
                ).fetchone()
                if parent is None:
                    raise ToolboxError(
                        "unknown_session", f"unknown session: {scope.session_id}"
                    )
                if origin_call_id is not None:
                    origin = connection.execute(
                        """
                        SELECT 1
                        FROM events AS started
                        WHERE started.session_id = ?
                          AND started.call_id = ?
                          AND started.kind = 'call_started'
                          AND NOT EXISTS (
                              SELECT 1 FROM events AS terminal
                              WHERE terminal.session_id = started.session_id
                                AND terminal.call_id = started.call_id
                                AND terminal.kind IN ('call_succeeded', 'call_failed')
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM events AS child
                              WHERE child.session_id = started.session_id
                                AND child.parent_call_id = started.call_id
                                AND child.kind = 'call_started'
                                AND NOT EXISTS (
                                    SELECT 1 FROM events AS child_terminal
                                    WHERE child_terminal.session_id = child.session_id
                                      AND child_terminal.call_id = child.call_id
                                      AND child_terminal.kind IN (
                                          'call_succeeded', 'call_failed'
                                      )
                                )
                          )
                        """,
                        (scope.session_id, origin_call_id),
                    ).fetchone()
                    if origin is None:
                        raise ToolboxError(
                            "invalid_session_origin",
                            "origin call is not the active leaf in the parent session",
                        )
                connection.execute(
                    """
                    INSERT INTO sessions (
                        id, cwd, parent_session_id, kind, origin_call_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        scope.cwd,
                        scope.session_id,
                        kind,
                        origin_call_id,
                        _now(),
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO session_toolboxes (session_id, position, toolbox_id)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (identifier, position, toolbox_id)
                        for position, toolbox_id in enumerate(scope.toolbox_ids)
                    ],
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return identifier

    def active_toolboxes(self, session_id: str) -> list[dict[str, Any]]:
        """Return one session's ordered toolbox selection."""
        self.create_session(session_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT t.id, t.name, d.cwd, s.position
                FROM session_toolboxes AS s
                JOIN toolboxes AS t ON t.id = s.toolbox_id
                LEFT JOIN cwd_defaults AS d ON d.toolbox_id = t.id
                WHERE s.session_id = ?
                ORDER BY s.position
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "cwd": row["cwd"],
                "position": int(row["position"]),
                "primary": int(row["position"]) == 0,
            }
            for row in rows
        ]

    def _selection_ids(
        self, connection: sqlite3.Connection, session_id: str
    ) -> list[str]:
        rows = connection.execute(
            """
            SELECT toolbox_id FROM session_toolboxes
            WHERE session_id = ? ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        return [str(row["toolbox_id"]) for row in rows]

    def select_toolboxes(
        self,
        session_id: str,
        names: list[str],
        *,
        mode: str = "use",
    ) -> list[dict[str, Any]]:
        """Replace, append to, or remove from a session selection."""
        if mode not in {"use", "add", "remove"}:
            raise ToolboxError("invalid_selection_mode", f"invalid mode: {mode}")
        self.create_session(session_id)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                before = self._selection_ids(connection, session_id)
                requested = list(
                    dict.fromkeys(self._toolbox_id(connection, name) for name in names)
                )
                if mode == "use":
                    after = list(dict.fromkeys(requested))
                    if not after:
                        raise ToolboxError(
                            "empty_selection", "use requires at least one toolbox"
                        )
                elif mode == "add":
                    after = before + [item for item in requested if item not in before]
                else:
                    removed = set(requested)
                    after = [item for item in before if item not in removed]
                    if not after:
                        after = [self._session_default_id(connection, session_id)]
                self._replace_selection(connection, session_id, after)
                self._append_event_connection(
                    connection,
                    session_id,
                    "toolbox_selection_changed",
                    {"mode": mode, "before": before, "after": after},
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.active_toolboxes(session_id)

    @staticmethod
    def _replace_selection(
        connection: sqlite3.Connection, session_id: str, toolbox_ids: list[str]
    ) -> None:
        connection.execute(
            "DELETE FROM session_toolboxes WHERE session_id = ?", (session_id,)
        )
        connection.executemany(
            """
            INSERT INTO session_toolboxes (session_id, position, toolbox_id)
            VALUES (?, ?, ?)
            """,
            [
                (session_id, position, toolbox_id)
                for position, toolbox_id in enumerate(toolbox_ids)
            ],
        )

    def _session_default_id(
        self, connection: sqlite3.Connection, session_id: str
    ) -> str:
        row = connection.execute(
            "SELECT cwd FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise ToolboxError("unknown_session", f"unknown session: {session_id}")
        cwd = str(row["cwd"] or self.cwd)
        return self._ensure_cwd_default(connection, cwd)

    def reset_toolboxes(self, session_id: str) -> list[dict[str, Any]]:
        """Reset a session to the toolbox mapped to its launch cwd."""
        self.create_session(session_id)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                before = self._selection_ids(connection, session_id)
                after = [self._session_default_id(connection, session_id)]
                self._replace_selection(connection, session_id, after)
                self._append_event_connection(
                    connection,
                    session_id,
                    "toolbox_selection_changed",
                    {"mode": "default", "before": before, "after": after},
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.active_toolboxes(session_id)

    def snapshot_scope(self, session_id: str) -> InvocationScope:
        """Capture the immutable selection for one top-level call tree."""
        self.create_session(session_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT cwd FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            assert row is not None
            toolbox_ids = tuple(self._selection_ids(connection, session_id))
        if not toolbox_ids:
            raise ToolboxError("empty_selection", "session has no active toolbox")
        return InvocationScope(
            session_id=session_id,
            toolbox_ids=toolbox_ids,
            cwd=str(row["cwd"] or self.cwd),
        )

    def append_event(
        self,
        session_id: str,
        kind: str,
        payload: Any,
        *,
        call_id: str | None = None,
        parent_call_id: str | None = None,
        toolbox_id: str | None = None,
        tool_name: str | None = None,
        tool_version: int | None = None,
        tool_version_id: int | None = None,
    ) -> int:
        """Append one committed event and return its session-local sequence."""
        with closing(self._connect()) as connection, connection:
            return self._append_event_connection(
                connection,
                session_id,
                kind,
                payload,
                call_id=call_id,
                parent_call_id=parent_call_id,
                toolbox_id=toolbox_id,
                tool_name=tool_name,
                tool_version=tool_version,
                tool_version_id=tool_version_id,
            )

    def _append_event_connection(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        kind: str,
        payload: Any,
        *,
        call_id: str | None = None,
        parent_call_id: str | None = None,
        toolbox_id: str | None = None,
        tool_name: str | None = None,
        tool_version: int | None = None,
        tool_version_id: int | None = None,
    ) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq
            FROM events WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        assert row is not None
        sequence = int(row["next_seq"])
        connection.execute(
            """
            INSERT INTO events (
                session_id, seq, kind, call_id, parent_call_id, toolbox_id,
                tool_name, tool_version, tool_version_id, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                sequence,
                kind,
                call_id,
                parent_call_id,
                toolbox_id,
                tool_name,
                tool_version,
                tool_version_id,
                _json(payload),
                _now(),
            ),
        )
        return sequence

    def list_sessions(self, *, limit: int = 20, cursor: int = 0) -> dict[str, Any]:
        """List saved sessions with bounded pagination."""
        limit = min(100, max(1, limit))
        cursor = max(0, cursor)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT sessions.id, sessions.cwd, sessions.parent_session_id,
                       sessions.kind, sessions.origin_call_id, sessions.created_at,
                       COUNT(events.id) AS event_count
                FROM sessions
                LEFT JOIN events ON events.session_id = sessions.id
                GROUP BY sessions.id
                ORDER BY sessions.created_at DESC, sessions.id DESC
                LIMIT ? OFFSET ?
                """,
                (limit + 1, cursor),
            ).fetchall()
            sessions = []
            for row in rows[:limit]:
                session = dict(row)
                session["root_session_id"] = self._root_session_id(
                    connection, str(row["id"])
                )
                sessions.append(session)
        has_more = len(rows) > limit
        return {
            "sessions": sessions,
            "next_cursor": cursor + limit if has_more else None,
        }

    @staticmethod
    def _root_session_id(connection: sqlite3.Connection, session_id: str) -> str:
        row = connection.execute(
            """
            WITH RECURSIVE lineage(id, parent_session_id) AS (
                SELECT id, parent_session_id FROM sessions WHERE id = ?
                UNION ALL
                SELECT sessions.id, sessions.parent_session_id
                FROM sessions JOIN lineage ON sessions.id = lineage.parent_session_id
            )
            SELECT id FROM lineage WHERE parent_session_id IS NULL
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise ToolboxError("unknown_session", f"unknown session: {session_id}")
        return str(row["id"])

    def session_metadata(self, session_id: str) -> dict[str, Any]:
        """Return one session's durable identity and derived root."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT id, cwd, parent_session_id, kind, origin_call_id, created_at
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                raise ToolboxError("unknown_session", f"unknown session: {session_id}")
            metadata = dict(row)
            metadata["root_session_id"] = self._root_session_id(connection, session_id)
        return metadata

    def read_session(
        self, session_id: str, *, after: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        """Read committed events in stable sequence order."""
        limit = min(100, max(1, limit))
        after = max(0, after)
        with closing(self._connect()) as connection:
            session = connection.execute(
                """
                SELECT id, cwd, parent_session_id, kind, origin_call_id, created_at
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if session is None:
                raise ToolboxError("unknown_session", f"unknown session: {session_id}")
            metadata = dict(session)
            metadata["root_session_id"] = self._root_session_id(connection, session_id)
            rows = connection.execute(
                """
                SELECT seq, kind, call_id, parent_call_id, toolbox_id, tool_name,
                       tool_version, tool_version_id, payload_json, created_at
                FROM events
                WHERE session_id = ? AND seq > ?
                ORDER BY seq
                LIMIT ?
                """,
                (session_id, after, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        events = []
        for row in rows[:limit]:
            event = dict(row)
            event["payload"] = json.loads(event.pop("payload_json"))
            events.append(event)
        return {
            "session_id": session_id,
            "session": metadata,
            "events": events,
            "next_after": events[-1]["seq"] if has_more and events else None,
        }

    def _scope(
        self,
        *,
        session_id: str | None = None,
        scope: InvocationScope | None = None,
    ) -> InvocationScope:
        if scope is not None:
            return scope
        if session_id is not None:
            identifier = self.create_session(session_id)
            return self.snapshot_scope(identifier)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                toolbox_id = self._ensure_cwd_default(connection, self.cwd)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return InvocationScope(
            session_id="host",
            toolbox_ids=(toolbox_id,),
            cwd=self.cwd,
        )

    def _resolve(
        self,
        name: str,
        *,
        version: int | None = None,
        scope: InvocationScope,
        toolbox_id: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> _ResolvedTool:
        owned = connection is None
        db = connection or self._connect()
        try:
            toolbox_ids = (toolbox_id,) if toolbox_id is not None else scope.toolbox_ids
            binding = None
            for candidate in toolbox_ids:
                binding = db.execute(
                    """
                    SELECT b.name, b.tool_version_id, l.id AS lineage_id,
                           l.toolbox_id, t.name AS toolbox_name
                    FROM bindings AS b
                    JOIN tool_versions AS active ON active.id = b.tool_version_id
                    JOIN tool_lineages AS l ON l.id = active.lineage_id
                    JOIN toolboxes AS t ON t.id = b.toolbox_id
                    WHERE b.toolbox_id = ? AND b.name = ?
                    """,
                    (candidate, name),
                ).fetchone()
                if binding is not None:
                    break
            if binding is None:
                raise ToolboxError("unknown_tool", f"unknown tool {name!r}")
            if version is None:
                row = db.execute(
                    "SELECT * FROM tool_versions WHERE id = ?",
                    (binding["tool_version_id"],),
                ).fetchone()
            else:
                row = db.execute(
                    """
                    SELECT * FROM tool_versions
                    WHERE lineage_id = ? AND version = ?
                    """,
                    (binding["lineage_id"], version),
                ).fetchone()
            if row is None:
                raise ToolboxError(
                    "unknown_tool",
                    f"unknown tool {name!r} version {version}",
                )
            return _ResolvedTool(
                id=int(row["id"]),
                toolbox_id=str(binding["toolbox_id"]),
                toolbox_name=str(binding["toolbox_name"]),
                name=str(binding["name"]),
                version=int(row["version"]),
                description=str(row["description"]),
                input_schema=json.loads(row["schema_json"]),
                source=str(row["source"]),
            )
        finally:
            if owned:
                db.close()

    def tool_definitions(
        self, *, session_id: str | None = None, scope: InvocationScope | None = None
    ) -> list[dict[str, Any]]:
        """Return the five fixed model-facing operation definitions."""
        active_scope = self._scope(session_id=session_id, scope=scope)
        return [
            {
                "name": name,
                "description": self._resolve(name, scope=active_scope).description,
                "input_schema": CORE_SCHEMAS[name],
            }
            for name in CORE_NAMES
        ]

    def catalog(
        self,
        *,
        include_core: bool = True,
        session_id: str | None = None,
        scope: InvocationScope | None = None,
    ) -> list[dict[str, Any]]:
        """Return effective metadata for an editable search implementation."""
        active_scope = self._scope(session_id=session_id, scope=scope)
        with closing(self._connect()) as connection:
            rows = self._effective_rows(connection, active_scope)
        tools = []
        for row in rows:
            name = str(row["name"])
            if not include_core and name in CORE_NAMES:
                continue
            tools.append(
                {
                    "name": name,
                    "version": int(row["version"]),
                    "description": row["description"],
                    "input_schema": json.loads(row["schema_json"]),
                    "source": row["source"],
                    "kind": "core" if name in CORE_NAMES else "user",
                    "created_at": row["created_at"],
                    "toolbox": row["toolbox_name"],
                }
            )
        return tools

    @staticmethod
    def _effective_rows(
        connection: sqlite3.Connection, scope: InvocationScope
    ) -> list[sqlite3.Row]:
        seen: set[str] = set()
        rows: list[sqlite3.Row] = []
        for toolbox_id in scope.toolbox_ids:
            candidates = connection.execute(
                """
                SELECT b.name, v.id AS tool_version_id, v.lineage_id, v.version,
                       v.description, v.schema_json, v.source, v.created_at,
                       b.toolbox_id, t.name AS toolbox_name
                FROM bindings AS b
                JOIN tool_versions AS v ON v.id = b.tool_version_id
                JOIN toolboxes AS t ON t.id = b.toolbox_id
                WHERE b.toolbox_id = ?
                ORDER BY b.name
                """,
                (toolbox_id,),
            ).fetchall()
            for row in candidates:
                name = str(row["name"])
                if name not in seen:
                    seen.add(name)
                    rows.append(row)
        return sorted(rows, key=lambda row: str(row["name"]))

    def view_tool(
        self,
        name: str,
        *,
        version: int | None = None,
        session_id: str | None = None,
        scope: InvocationScope | None = None,
        toolbox: str | None = None,
    ) -> dict[str, Any]:
        """View an effective or explicitly scoped tool lineage in detail."""
        active_scope = self._scope(session_id=session_id, scope=scope)
        with closing(self._connect()) as connection:
            toolbox_id = self._toolbox_id(connection, toolbox) if toolbox else None
            tool = self._resolve(
                name,
                version=version,
                scope=active_scope,
                toolbox_id=toolbox_id,
                connection=connection,
            )
            active = self._resolve(
                name,
                scope=active_scope,
                toolbox_id=tool.toolbox_id,
                connection=connection,
            )
        return {
            "name": tool.name,
            "version": tool.version,
            "active": active.version == tool.version,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "source": tool.source,
            "kind": "core" if name in CORE_NAMES else "user",
            "toolbox": tool.toolbox_name,
        }

    def write_tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        source: str,
        base_version: int | None = None,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        scope: InvocationScope | None = None,
        toolbox: str | None = None,
    ) -> dict[str, Any]:
        """Compile, store, and bind one namespace-owned immutable version."""
        self._validate_tool(name, description, input_schema, source)
        if name in CORE_SCHEMAS and input_schema != CORE_SCHEMAS[name]:
            raise ToolboxError(
                "core_schema_pinned",
                f"the outer schema for {name} cannot change in this prototype",
            )
        active_scope = self._scope(session_id=session_id, scope=scope)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if toolbox is not None:
                    target_id = self._toolbox_id(connection, toolbox)
                else:
                    try:
                        target_id = self._resolve(
                            name, scope=active_scope, connection=connection
                        ).toolbox_id
                    except ToolboxError as error:
                        if error.code != "unknown_tool":
                            raise
                        target_id = active_scope.toolbox_ids[0]
                lineage = connection.execute(
                    """
                    SELECT id FROM tool_lineages
                    WHERE toolbox_id = ? AND name = ?
                    """,
                    (target_id, name),
                ).fetchone()
                if lineage is None:
                    lineage_id = int(
                        connection.execute(
                            """
                            INSERT INTO tool_lineages (toolbox_id, name, created_at)
                            VALUES (?, ?, ?)
                            """,
                            (target_id, name, _now()),
                        ).lastrowid
                    )
                else:
                    lineage_id = int(lineage["id"])
                active_row = connection.execute(
                    """
                    SELECT v.version
                    FROM bindings AS b
                    JOIN tool_versions AS v ON v.id = b.tool_version_id
                    WHERE b.toolbox_id = ? AND b.name = ?
                    """,
                    (target_id, name),
                ).fetchone()
                active_version = (
                    int(active_row["version"]) if active_row is not None else None
                )
                if base_version is not None and active_version != base_version:
                    raise ToolboxError(
                        "stale_base_version",
                        (
                            f"active version for {name} is {active_version}, "
                            f"not {base_version}"
                        ),
                        active_version=active_version,
                        base_version=base_version,
                    )
                row = connection.execute(
                    """
                    SELECT COALESCE(MAX(version), 0) + 1 AS next
                    FROM tool_versions WHERE lineage_id = ?
                    """,
                    (lineage_id,),
                ).fetchone()
                assert row is not None
                version = int(row["next"])
                cursor = connection.execute(
                    """
                    INSERT INTO tool_versions (
                        lineage_id, version, description, schema_json, source,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lineage_id,
                        version,
                        description,
                        _json(input_schema),
                        source,
                        _now(),
                    ),
                )
                version_id = int(cursor.lastrowid)
                connection.execute(
                    """
                    INSERT INTO bindings (toolbox_id, name, tool_version_id)
                    VALUES (?, ?, ?)
                    ON CONFLICT(toolbox_id, name)
                    DO UPDATE SET tool_version_id = excluded.tool_version_id
                    """,
                    (target_id, name, version_id),
                )
                if session_id is not None:
                    self._append_event_connection(
                        connection,
                        session_id,
                        "binding_changed",
                        {
                            "name": name,
                            "from_version": active_version,
                            "to_version": version,
                            "toolbox_id": target_id,
                        },
                        parent_call_id=parent_call_id,
                        toolbox_id=target_id,
                        tool_name=name,
                        tool_version=version,
                        tool_version_id=version_id,
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {
            "name": name,
            "version": version,
            "active": True,
            "previous_version": active_version,
        }

    def _validate_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        source: str,
    ) -> None:
        if not isinstance(name, str) or _TOOL_NAME.fullmatch(name) is None:
            raise ToolboxError("invalid_name", f"invalid tool name: {name!r}")
        if not isinstance(description, str) or not description.strip():
            raise ToolboxError("invalid_description", "description must be non-empty")
        if not isinstance(input_schema, dict):
            raise ToolboxError("invalid_schema", "input_schema must be an object")
        _json(input_schema)
        if not isinstance(source, str) or not source.strip():
            raise ToolboxError("invalid_source", "source must be non-empty")
        try:
            tree = ast.parse(source, filename=f"<tool:{name}>", mode="exec")
            compile(tree, filename=f"<tool:{name}>", mode="exec")
        except SyntaxError as error:
            raise ToolboxError(
                "invalid_source",
                f"source does not compile: {error.msg}",
                line=error.lineno,
            ) from error
        mains = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
        ]
        if not mains:
            raise ToolboxError(
                "missing_main", "source must define def main(input, ctx)"
            )
        main = mains[-1]
        if isinstance(main, ast.AsyncFunctionDef):
            raise ToolboxError(
                "async_main", "main must be synchronous in this prototype"
            )
        positional = len(main.args.posonlyargs) + len(main.args.args)
        if positional != 2 or main.args.vararg is not None:
            raise ToolboxError(
                "invalid_main", "main must accept exactly two positional arguments"
            )

    def call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session_id: str | None = None,
        version: int | None = None,
        scope: InvocationScope | None = None,
        model_provider: Any | None = None,
    ) -> JsonValue:
        """Start a top-level invocation in a durable session."""
        from mechagnome.model_provider import (
            ModelProvider,
            ModelSession,
            _bind_model_provider,
            _BoundedModelProvider,
        )

        if scope is None and session_id is None:
            identifier = self.create_session()
            active_scope = self.snapshot_scope(identifier)
        else:
            active_scope = self._scope(session_id=session_id, scope=scope)
        provider = model_provider
        if provider is not None and not isinstance(provider, _BoundedModelProvider):
            gateway = ModelProvider.from_completion_transport(self, provider)
            provider = ModelSession(
                gateway, active_scope.session_id
            ).completion_provider(active_scope)
        bound_provider = _bind_model_provider(provider)
        bound_provider = bound_provider.for_scope(active_scope)
        state = _InvocationState(
            scope=active_scope,
            max_depth=self.max_depth,
            max_calls=self.max_calls,
            model_provider=bound_provider,
        )
        self.append_event(
            active_scope.session_id,
            "invocation_scope",
            {"toolbox_ids": active_scope.toolbox_ids, "cwd": active_scope.cwd},
        )
        return self._invoke(name, args, state=state, depth=0, version=version)

    def _invoke(
        self,
        name: str,
        args: dict[str, Any],
        *,
        state: _InvocationState,
        depth: int,
        parent_call_id: str | None = None,
        version: int | None = None,
    ) -> JsonValue:
        if not isinstance(args, dict):
            raise ToolboxError("invalid_input", "tool args must be an object")
        _json(args)
        if depth > state.max_depth:
            raise ToolboxError(
                "max_depth", f"maximum nested call depth {state.max_depth} exceeded"
            )
        if state.calls >= state.max_calls:
            raise ToolboxError(
                "max_calls", f"maximum call count {state.max_calls} exceeded"
            )
        state.calls += 1
        tool = self._resolve(name, version=version, scope=state.scope)
        call_id = uuid.uuid4().hex
        self.append_event(
            state.session_id,
            "call_started",
            {"args": args},
            call_id=call_id,
            parent_call_id=parent_call_id,
            toolbox_id=tool.toolbox_id,
            tool_name=tool.name,
            tool_version=tool.version,
            tool_version_id=tool.id,
        )
        logical_slot = tool.name if tool.name in CORE_NAMES else None
        context = ToolContext(self, state, call_id, depth, logical_slot)
        started_at = time.perf_counter_ns()
        try:
            namespace: dict[str, Any] = {
                "__name__": f"toolbox_tool_{tool.id}",
                "__file__": f"<tool:{tool.name}@{tool.version}>",
            }
            code = compile(
                tool.source,
                filename=f"<tool:{tool.name}@{tool.version}>",
                mode="exec",
            )
            exec(code, namespace)
            main = namespace.get("main")
            if not callable(main):
                raise ToolboxError(
                    "missing_main", "executed source has no callable main"
                )
            result = main(args, context)
            if inspect.isawaitable(result):
                raise ToolboxError(
                    "async_result", "tool returned an awaitable; use synchronous main"
                )
            _json(result)
        except Exception as error:
            duration_ms = (time.perf_counter_ns() - started_at) / 1_000_000
            details = (
                error.to_dict()["error"]
                if isinstance(error, ToolboxError)
                else {"code": "tool_failed", "message": str(error)}
            )
            self.append_event(
                state.session_id,
                "call_failed",
                {**details, "duration_ms": duration_ms},
                call_id=call_id,
                parent_call_id=parent_call_id,
                toolbox_id=tool.toolbox_id,
                tool_name=tool.name,
                tool_version=tool.version,
                tool_version_id=tool.id,
            )
            raise
        duration_ms = (time.perf_counter_ns() - started_at) / 1_000_000
        self.append_event(
            state.session_id,
            "call_succeeded",
            {"result": result, "duration_ms": duration_ms},
            call_id=call_id,
            parent_call_id=parent_call_id,
            toolbox_id=tool.toolbox_id,
            tool_name=tool.name,
            tool_version=tool.version,
            tool_version_id=tool.id,
        )
        return result

    def bindings(
        self,
        *,
        recent_first: bool = False,
        session_id: str | None = None,
        scope: InvocationScope | None = None,
        include_origin: bool = False,
        toolbox: str | None = None,
    ) -> list[dict[str, Any]]:
        """Inspect effective bindings without calling editable code."""
        active_scope = self._scope(session_id=session_id, scope=scope)
        with closing(self._connect()) as connection:
            if toolbox is not None:
                active_scope = InvocationScope(
                    session_id=active_scope.session_id,
                    toolbox_ids=(self._toolbox_id(connection, toolbox),),
                    cwd=active_scope.cwd,
                )
            rows = self._effective_rows(connection, active_scope)
            items = []
            for row in rows:
                versions = connection.execute(
                    """
                    SELECT version FROM tool_versions
                    WHERE lineage_id = ? ORDER BY version
                    """,
                    (row["lineage_id"],),
                ).fetchall()
                used = connection.execute(
                    """
                    SELECT MAX(created_at) AS last_used
                    FROM events WHERE kind = 'call_started'
                    AND tool_version_id IN (
                        SELECT id FROM tool_versions WHERE lineage_id = ?
                    )
                    """,
                    (row["lineage_id"],),
                ).fetchone()
                item = {
                    "name": row["name"],
                    "active_version": int(row["version"]),
                    "description": row["description"],
                    "versions": [int(version["version"]) for version in versions],
                    "kind": "core" if row["name"] in CORE_NAMES else "user",
                }
                if include_origin:
                    item["toolbox"] = row["toolbox_name"]
                    item["toolbox_id"] = row["toolbox_id"]
                items.append((used["last_used"] if used else None, item))
        if recent_first:
            items.sort(
                key=lambda pair: (pair[0] is None, pair[0] or "", pair[1]["name"])
            )
            used_items = [pair for pair in items if pair[0] is not None]
            unused_items = [pair for pair in items if pair[0] is None]
            used_items.reverse()
            items = used_items + unused_items
        else:
            items.sort(key=lambda pair: pair[1]["name"])
        return [item for _, item in items]

    def _lineage_for_history(
        self,
        connection: sqlite3.Connection,
        name: str,
        scope: InvocationScope,
        toolbox: str | None,
    ) -> sqlite3.Row:
        if toolbox is not None:
            toolbox_ids = (self._toolbox_id(connection, toolbox),)
        else:
            toolbox_ids = scope.toolbox_ids
            try:
                resolved = self._resolve(name, scope=scope, connection=connection)
            except ToolboxError as error:
                if error.code != "unknown_tool":
                    raise
            else:
                toolbox_ids = (resolved.toolbox_id,)
        for toolbox_id in toolbox_ids:
            row = connection.execute(
                """
                SELECT l.id, l.toolbox_id, t.name AS toolbox_name
                FROM tool_lineages AS l
                JOIN toolboxes AS t ON t.id = l.toolbox_id
                WHERE l.toolbox_id = ? AND l.name = ?
                """,
                (toolbox_id, name),
            ).fetchone()
            if row is not None:
                return row
        raise ToolboxError("unknown_tool", f"unknown tool: {name}")

    def tool_inventory(
        self, *, session_id: str | None = None, scope: InvocationScope | None = None
    ) -> list[dict[str, Any]]:
        """Return visible and historical tools in the selected namespaces."""
        active_scope = self._scope(session_id=session_id, scope=scope)
        names: set[str] = set()
        with closing(self._connect()) as connection:
            for toolbox_id in active_scope.toolbox_ids:
                names.update(
                    str(row["name"])
                    for row in connection.execute(
                        "SELECT name FROM tool_lineages WHERE toolbox_id = ?",
                        (toolbox_id,),
                    )
                )
        inventory = []
        for name in sorted(names):
            history = self.tool_history(name, scope=active_scope)
            latest = history["versions"][0]
            inventory.append(
                {
                    "name": name,
                    "active_version": history["active_version"],
                    "latest_version": latest["version"],
                    "description": latest["description"],
                    "created_at": latest["created_at"],
                    "version_count": len(history["versions"]),
                    "call_count": history["call_count"],
                    "session_count": len(history["sessions"]),
                    "kind": history["kind"],
                }
            )
        return inventory

    def tool_history(
        self,
        name: str,
        *,
        session_id: str | None = None,
        scope: InvocationScope | None = None,
        toolbox: str | None = None,
    ) -> dict[str, Any]:
        """Return one namespace-owned lineage's versions and usage."""
        active_scope = self._scope(session_id=session_id, scope=scope)
        with closing(self._connect()) as connection:
            lineage = self._lineage_for_history(connection, name, active_scope, toolbox)
            version_rows = connection.execute(
                """
                SELECT id, version, description, schema_json, source, created_at
                FROM tool_versions WHERE lineage_id = ? ORDER BY version DESC
                """,
                (lineage["id"],),
            ).fetchall()
            active_row = connection.execute(
                """
                SELECT v.version FROM bindings AS b
                JOIN tool_versions AS v ON v.id = b.tool_version_id
                WHERE b.toolbox_id = ? AND b.name = ?
                """,
                (lineage["toolbox_id"], name),
            ).fetchone()
            ids = [int(row["id"]) for row in version_rows]
            placeholders = ",".join("?" for _ in ids)
            event_rows = (
                connection.execute(
                    f"""
                    SELECT tool_version_id, kind, session_id, created_at,
                           CASE WHEN kind IN ('call_succeeded', 'call_failed')
                                THEN json_type(payload_json, '$.duration_ms')
                           END AS duration_type,
                           CASE WHEN kind IN ('call_succeeded', 'call_failed')
                                THEN json_extract(payload_json, '$.duration_ms')
                           END AS duration_ms
                    FROM events WHERE tool_version_id IN ({placeholders})
                    """,
                    ids,
                ).fetchall()
                if ids
                else []
            )
        active_version = int(active_row["version"]) if active_row is not None else None
        grouped: dict[int, list[sqlite3.Row]] = {}
        for event in event_rows:
            grouped.setdefault(int(event["tool_version_id"]), []).append(event)
        versions = []
        sessions: dict[str, list[str]] = {}
        for row in version_rows:
            events = grouped.get(int(row["id"]), [])
            calls = [event for event in events if event["kind"] == "call_started"]
            durations = []
            for event in events:
                if event["duration_type"] not in {"integer", "real"}:
                    continue
                duration = float(event["duration_ms"])
                if math.isfinite(duration) and duration >= 0:
                    durations.append(duration)
            for event in calls:
                sessions.setdefault(str(event["session_id"]), []).append(
                    str(event["created_at"])
                )
            creator = next(
                (
                    str(event["session_id"])
                    for event in events
                    if event["kind"] == "binding_changed"
                ),
                None,
            )
            version = int(row["version"])
            versions.append(
                {
                    "version": version,
                    "tool_version_id": int(row["id"]),
                    "active": version == active_version,
                    "description": row["description"],
                    "input_schema": json.loads(row["schema_json"]),
                    "source": row["source"],
                    "created_at": row["created_at"],
                    "created_session_id": creator,
                    "call_count": len(calls),
                    "success_count": sum(
                        event["kind"] == "call_succeeded" for event in events
                    ),
                    "failure_count": sum(
                        event["kind"] == "call_failed" for event in events
                    ),
                    "session_count": len({event["session_id"] for event in calls}),
                    "last_called_at": max(
                        (str(event["created_at"]) for event in calls), default=None
                    ),
                    "timed_call_count": len(durations),
                    "average_duration_ms": (
                        sum(durations) / len(durations) if durations else None
                    ),
                }
            )
        session_items = [
            {
                "session_id": session_id,
                "call_count": len(timestamps),
                "last_called_at": max(timestamps),
            }
            for session_id, timestamps in sessions.items()
        ]
        session_items.sort(
            key=lambda item: (item["last_called_at"], item["session_id"]), reverse=True
        )
        return {
            "name": name,
            "kind": "core" if name in CORE_NAMES else "user",
            "active_version": active_version,
            "versions": versions,
            "sessions": session_items,
            "call_count": sum(item["call_count"] for item in versions),
            "success_count": sum(item["success_count"] for item in versions),
            "failure_count": sum(item["failure_count"] for item in versions),
            "toolbox": lineage["toolbox_name"],
        }

    def delete_tool(
        self,
        name: str,
        *,
        session_id: str | None = None,
        scope: InvocationScope | None = None,
        toolbox: str | None = None,
    ) -> dict[str, Any]:
        """Remove one namespace binding while retaining its lineage history."""
        if name in CORE_NAMES:
            raise ToolboxError("core_tool_required", f"cannot delete core tool: {name}")
        active_scope = self._scope(session_id=session_id, scope=scope)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                target = (
                    self._toolbox_id(connection, toolbox)
                    if toolbox
                    else self._resolve(
                        name, scope=active_scope, connection=connection
                    ).toolbox_id
                )
                tool = self._resolve(
                    name,
                    scope=active_scope,
                    toolbox_id=target,
                    connection=connection,
                )
                connection.execute(
                    "DELETE FROM bindings WHERE toolbox_id = ? AND name = ?",
                    (target, name),
                )
                if session_id is not None:
                    self._append_event_connection(
                        connection,
                        session_id,
                        "binding_deleted",
                        {
                            "name": name,
                            "from_version": tool.version,
                            "toolbox_id": target,
                        },
                        toolbox_id=target,
                        tool_name=name,
                        tool_version=tool.version,
                        tool_version_id=tool.id,
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {"name": name, "deleted_version": tool.version, "active": False}

    def rollback(
        self,
        name: str,
        *,
        version: int,
        session_id: str | None = None,
        scope: InvocationScope | None = None,
        toolbox: str | None = None,
    ) -> dict[str, Any]:
        """Move one namespace binding without editable tool code."""
        active_scope = self._scope(session_id=session_id, scope=scope)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if toolbox:
                    target_id = self._toolbox_id(connection, toolbox)
                    lineage = self._lineage_for_history(
                        connection, name, active_scope, toolbox
                    )
                    try:
                        current_tool = self._resolve(
                            name,
                            scope=active_scope,
                            toolbox_id=target_id,
                            connection=connection,
                        )
                    except ToolboxError as error:
                        if error.code != "unknown_tool":
                            raise
                        current_tool = None
                else:
                    current_tool = self._resolve(
                        name, scope=active_scope, connection=connection
                    )
                    target_id = current_tool.toolbox_id
                    lineage = self._lineage_for_history(
                        connection, name, active_scope, current_tool.toolbox_name
                    )
                row = connection.execute(
                    """
                    SELECT id, version FROM tool_versions
                    WHERE lineage_id = ? AND version = ?
                    """,
                    (lineage["id"], version),
                ).fetchone()
                if row is None:
                    raise ToolboxError(
                        "no_rollback",
                        f"no requested rollback version exists for {name}",
                    )
                connection.execute(
                    """
                    INSERT INTO bindings (toolbox_id, name, tool_version_id)
                    VALUES (?, ?, ?)
                    ON CONFLICT(toolbox_id, name)
                    DO UPDATE SET tool_version_id = excluded.tool_version_id
                    """,
                    (target_id, name, int(row["id"])),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {
            "name": name,
            "from_version": current_tool.version if current_tool else None,
            "to_version": int(row["version"]),
        }
