"""Immutable host kernel for persistent, dynamically authored tools."""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mechagnome.bootstrap import BOOTSTRAP_TOOLS, CORE_NAMES, CORE_SCHEMAS

JsonValue = Any

_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


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


@dataclass
class _InvocationState:
    session_id: str
    max_depth: int
    max_calls: int
    calls: int = 0


@dataclass(frozen=True)
class _ResolvedTool:
    id: int
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
        self.sessions = SessionAccess(kernel, state.session_id)

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
        """Invoke a tool through the currently active editable call_tool binding."""
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
        """Return active tool metadata to the search_tools implementation."""
        self._require("search_tools")
        return self._context._kernel.catalog(include_core=include_core)

    def read_tool_source(self, name: str, version: int | None = None) -> dict[str, Any]:
        """Read source through the read_tool_source capability."""
        self._require("read_tool_source")
        return self._context._kernel.read_tool_source(name, version=version)

    def write_tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        source: str,
        base_version: int | None = None,
    ) -> dict[str, Any]:
        """Store source through the write_tool capability."""
        self._require("write_tool")
        return self._context._kernel.write_tool(
            name=name,
            description=description,
            input_schema=input_schema,
            source=source,
            base_version=base_version,
            session_id=self._context._state.session_id,
            parent_call_id=self._context._call_id,
        )

    def execute(
        self, name: str, args: dict[str, Any], version: int | None = None
    ) -> JsonValue:
        """Bottom out recursive dispatch through the call_tool capability."""
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
        self, db_path: str | Path, *, max_depth: int = 12, max_calls: int = 100
    ) -> None:
        self.db_path = Path(db_path)
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tool_versions (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    schema_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(name, version)
                );

                CREATE TABLE IF NOT EXISTS bindings (
                    name TEXT PRIMARY KEY,
                    tool_id INTEGER NOT NULL REFERENCES tool_versions(id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    seq INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    call_id TEXT,
                    parent_call_id TEXT,
                    tool_name TEXT,
                    tool_version INTEGER,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, seq)
                );
                """
            )
            for tool in BOOTSTRAP_TOOLS:
                row = connection.execute(
                    "SELECT id FROM tool_versions WHERE name = ? AND version = 1",
                    (tool.name,),
                ).fetchone()
                if row is None:
                    cursor = connection.execute(
                        """
                        INSERT INTO tool_versions (
                            name, version, description, schema_json, source,
                            created_at
                        ) VALUES (?, 1, ?, ?, ?, ?)
                        """,
                        (
                            tool.name,
                            tool.description,
                            _json(tool.input_schema),
                            tool.source,
                            _now(),
                        ),
                    )
                    tool_id = int(cursor.lastrowid)
                else:
                    tool_id = int(row["id"])
                connection.execute(
                    """
                    INSERT OR IGNORE INTO bindings (name, tool_id)
                    VALUES (?, ?)
                    """,
                    (tool.name, tool_id),
                )

    def create_session(self, session_id: str | None = None) -> str:
        """Create and return a durable session identifier."""
        identifier = session_id or uuid.uuid4().hex
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
                (identifier, _now()),
            )
        return identifier

    def append_event(
        self,
        session_id: str,
        kind: str,
        payload: Any,
        *,
        call_id: str | None = None,
        parent_call_id: str | None = None,
        tool_name: str | None = None,
        tool_version: int | None = None,
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
                tool_name=tool_name,
                tool_version=tool_version,
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
        tool_name: str | None = None,
        tool_version: int | None = None,
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
                session_id, seq, kind, call_id, parent_call_id, tool_name,
                tool_version, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                sequence,
                kind,
                call_id,
                parent_call_id,
                tool_name,
                tool_version,
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
                SELECT sessions.id, sessions.created_at, COUNT(events.id) AS event_count
                FROM sessions
                LEFT JOIN events ON events.session_id = sessions.id
                GROUP BY sessions.id
                ORDER BY sessions.created_at DESC, sessions.id DESC
                LIMIT ? OFFSET ?
                """,
                (limit + 1, cursor),
            ).fetchall()
        has_more = len(rows) > limit
        items = [dict(row) for row in rows[:limit]]
        return {
            "sessions": items,
            "next_cursor": cursor + limit if has_more else None,
        }

    def read_session(
        self, session_id: str, *, after: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        """Read committed events in stable sequence order."""
        limit = min(100, max(1, limit))
        after = max(0, after)
        with closing(self._connect()) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                raise ToolboxError("unknown_session", f"unknown session: {session_id}")
            rows = connection.execute(
                """
                SELECT seq, kind, call_id, parent_call_id, tool_name,
                       tool_version, payload_json, created_at
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
            "events": events,
            "next_after": events[-1]["seq"] if has_more and events else None,
        }

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return the five fixed model-facing operation definitions."""
        definitions = []
        for name in CORE_NAMES:
            tool = self._resolve(name)
            definitions.append(
                {
                    "name": name,
                    "description": tool.description,
                    "input_schema": CORE_SCHEMAS[name],
                }
            )
        return definitions

    def catalog(self, *, include_core: bool = True) -> list[dict[str, Any]]:
        """Return active tool metadata for an editable search implementation."""
        query = """
            SELECT v.name, v.version, v.description, v.schema_json, v.source,
                   v.created_at
            FROM bindings AS b
            JOIN tool_versions AS v ON v.id = b.tool_id
            ORDER BY v.name
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(query).fetchall()
        tools = []
        for row in rows:
            if not include_core and row["name"] in CORE_NAMES:
                continue
            tools.append(
                {
                    "name": row["name"],
                    "version": row["version"],
                    "description": row["description"],
                    "input_schema": json.loads(row["schema_json"]),
                    "source": row["source"],
                    "kind": "core" if row["name"] in CORE_NAMES else "user",
                    "created_at": row["created_at"],
                }
            )
        return tools

    def read_tool_source(
        self, name: str, *, version: int | None = None
    ) -> dict[str, Any]:
        """Host-level source read used by the corresponding core capability."""
        tool = self._resolve(name, version=version)
        active = self._active_version(name)
        return {
            "name": tool.name,
            "version": tool.version,
            "active": active == tool.version,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "source": tool.source,
            "kind": "core" if name in CORE_NAMES else "user",
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
    ) -> dict[str, Any]:
        """Compile, store, and bind an immutable tool version."""
        self._validate_tool(name, description, input_schema, source)
        if name in CORE_SCHEMAS and input_schema != CORE_SCHEMAS[name]:
            raise ToolboxError(
                "core_schema_pinned",
                f"the outer schema for {name} cannot change in this prototype",
            )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                active_row = connection.execute(
                    """
                    SELECT v.version
                    FROM bindings AS b
                    JOIN tool_versions AS v ON v.id = b.tool_id
                    WHERE b.name = ?
                    """,
                    (name,),
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
                    FROM tool_versions WHERE name = ?
                    """,
                    (name,),
                ).fetchone()
                assert row is not None
                version = int(row["next"])
                cursor = connection.execute(
                    """
                    INSERT INTO tool_versions (
                        name, version, description, schema_json, source,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        version,
                        description,
                        _json(input_schema),
                        source,
                        _now(),
                    ),
                )
                tool_id = int(cursor.lastrowid)
                connection.execute(
                    """
                    INSERT INTO bindings (name, tool_id)
                    VALUES (?, ?)
                    ON CONFLICT(name) DO UPDATE SET tool_id = excluded.tool_id
                    """,
                    (name, tool_id),
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
                        },
                        parent_call_id=parent_call_id,
                        tool_name=name,
                        tool_version=version,
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
    ) -> JsonValue:
        """Start a top-level invocation in a durable session."""
        identifier = self.create_session(session_id)
        state = _InvocationState(
            session_id=identifier,
            max_depth=self.max_depth,
            max_calls=self.max_calls,
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
        tool = self._resolve(name, version=version)
        call_id = uuid.uuid4().hex
        self.append_event(
            state.session_id,
            "call_started",
            {"args": args},
            call_id=call_id,
            parent_call_id=parent_call_id,
            tool_name=tool.name,
            tool_version=tool.version,
        )
        logical_slot = tool.name if tool.name in CORE_NAMES else None
        context = ToolContext(self, state, call_id, depth, logical_slot)
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
            details = (
                error.to_dict()["error"]
                if isinstance(error, ToolboxError)
                else {"code": "tool_failed", "message": str(error)}
            )
            self.append_event(
                state.session_id,
                "call_failed",
                details,
                call_id=call_id,
                parent_call_id=parent_call_id,
                tool_name=tool.name,
                tool_version=tool.version,
            )
            raise
        self.append_event(
            state.session_id,
            "call_succeeded",
            {"result": result},
            call_id=call_id,
            parent_call_id=parent_call_id,
            tool_name=tool.name,
            tool_version=tool.version,
        )
        return result

    def _resolve(self, name: str, *, version: int | None = None) -> _ResolvedTool:
        with closing(self._connect()) as connection:
            if version is None:
                row = connection.execute(
                    """
                    SELECT v.*
                    FROM bindings AS b
                    JOIN tool_versions AS v ON v.id = b.tool_id
                    WHERE b.name = ?
                    """,
                    (name,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM tool_versions WHERE name = ? AND version = ?",
                    (name, version),
                ).fetchone()
        if row is None:
            qualifier = f" version {version}" if version is not None else ""
            raise ToolboxError("unknown_tool", f"unknown tool {name!r}{qualifier}")
        return _ResolvedTool(
            id=int(row["id"]),
            name=row["name"],
            version=int(row["version"]),
            description=row["description"],
            input_schema=json.loads(row["schema_json"]),
            source=row["source"],
        )

    def _active_version(self, name: str) -> int | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT v.version
                FROM bindings AS b
                JOIN tool_versions AS v ON v.id = b.tool_id
                WHERE b.name = ?
                """,
                (name,),
            ).fetchone()
        return int(row["version"]) if row is not None else None

    def bindings(self) -> list[dict[str, Any]]:
        """Inspect active bindings without calling editable code."""
        with closing(self._connect()) as connection:
            active_rows = connection.execute(
                """
                SELECT v.name, v.version, v.description
                FROM bindings AS b
                JOIN tool_versions AS v ON v.id = b.tool_id
                ORDER BY v.name
                """
            ).fetchall()
            version_rows = connection.execute(
                "SELECT name, version FROM tool_versions ORDER BY name, version"
            ).fetchall()
        versions: dict[str, list[int]] = {}
        for row in version_rows:
            versions.setdefault(row["name"], []).append(int(row["version"]))
        return [
            {
                "name": row["name"],
                "active_version": int(row["version"]),
                "description": row["description"],
                "versions": versions[row["name"]],
                "kind": "core" if row["name"] in CORE_NAMES else "user",
            }
            for row in active_rows
        ]

    def tool_inventory(self) -> list[dict[str, Any]]:
        """Return every known tool with active state and aggregate usage."""
        with closing(self._connect()) as connection:
            version_rows = connection.execute(
                """
                SELECT name, version, description, created_at
                FROM tool_versions
                ORDER BY name, version DESC
                """
            ).fetchall()
            active_rows = connection.execute(
                """
                SELECT v.name, v.version
                FROM bindings AS b
                JOIN tool_versions AS v ON v.id = b.tool_id
                """
            ).fetchall()
            usage_rows = connection.execute(
                """
                SELECT tool_name, COUNT(*) AS call_count,
                       COUNT(DISTINCT session_id) AS session_count
                FROM events
                WHERE kind = 'call_started' AND tool_name IS NOT NULL
                GROUP BY tool_name
                """
            ).fetchall()
        active = {row["name"]: int(row["version"]) for row in active_rows}
        usage = {row["tool_name"]: row for row in usage_rows}
        inventory: dict[str, dict[str, Any]] = {}
        for row in version_rows:
            name = str(row["name"])
            item = inventory.setdefault(
                name,
                {
                    "name": name,
                    "active_version": active.get(name),
                    "latest_version": int(row["version"]),
                    "description": row["description"],
                    "created_at": row["created_at"],
                    "version_count": 0,
                    "call_count": int(usage.get(name, {"call_count": 0})["call_count"]),
                    "session_count": int(
                        usage.get(name, {"session_count": 0})["session_count"]
                    ),
                    "kind": "core" if name in CORE_NAMES else "user",
                },
            )
            item["version_count"] += 1
        return list(inventory.values())

    def tool_history(self, name: str) -> dict[str, Any]:
        """Return immutable versions, provenance, and per-session call stats."""
        with closing(self._connect()) as connection:
            version_rows = connection.execute(
                """
                SELECT version, description, schema_json, source, created_at
                FROM tool_versions
                WHERE name = ?
                ORDER BY version DESC
                """,
                (name,),
            ).fetchall()
            if not version_rows:
                raise ToolboxError("unknown_tool", f"unknown tool: {name}")
            active_row = connection.execute(
                """
                SELECT v.version
                FROM bindings AS b
                JOIN tool_versions AS v ON v.id = b.tool_id
                WHERE b.name = ?
                """,
                (name,),
            ).fetchone()
            usage_rows = connection.execute(
                """
                SELECT tool_version,
                       SUM(kind = 'call_started') AS call_count,
                       SUM(kind = 'call_succeeded') AS success_count,
                       SUM(kind = 'call_failed') AS failure_count,
                       COUNT(DISTINCT CASE WHEN kind = 'call_started'
                                          THEN session_id END) AS session_count,
                       MAX(CASE WHEN kind = 'call_started'
                                THEN created_at END) AS last_called_at
                FROM events
                WHERE tool_name = ? AND tool_version IS NOT NULL
                GROUP BY tool_version
                """,
                (name,),
            ).fetchall()
            creator_rows = connection.execute(
                """
                SELECT tool_version, session_id
                FROM events
                WHERE kind = 'binding_changed' AND tool_name = ?
                      AND tool_version IS NOT NULL
                ORDER BY created_at, id
                """,
                (name,),
            ).fetchall()
            session_rows = connection.execute(
                """
                SELECT session_id, COUNT(*) AS call_count,
                       MAX(created_at) AS last_called_at
                FROM events
                WHERE kind = 'call_started' AND tool_name = ?
                GROUP BY session_id
                ORDER BY last_called_at DESC, session_id
                """,
                (name,),
            ).fetchall()
        active_version = int(active_row["version"]) if active_row is not None else None
        usage = {
            int(row["tool_version"]): row
            for row in usage_rows
            if row["tool_version"] is not None
        }
        creators: dict[int, str] = {}
        for row in creator_rows:
            creators.setdefault(int(row["tool_version"]), str(row["session_id"]))
        versions = []
        for row in version_rows:
            version = int(row["version"])
            stats = usage.get(version)
            versions.append(
                {
                    "version": version,
                    "active": version == active_version,
                    "description": row["description"],
                    "input_schema": json.loads(row["schema_json"]),
                    "source": row["source"],
                    "created_at": row["created_at"],
                    "created_session_id": creators.get(version),
                    "call_count": int(stats["call_count"] or 0) if stats else 0,
                    "success_count": int(stats["success_count"] or 0) if stats else 0,
                    "failure_count": int(stats["failure_count"] or 0) if stats else 0,
                    "session_count": int(stats["session_count"] or 0) if stats else 0,
                    "last_called_at": stats["last_called_at"] if stats else None,
                }
            )
        sessions = [
            {
                "session_id": row["session_id"],
                "call_count": int(row["call_count"]),
                "last_called_at": row["last_called_at"],
            }
            for row in session_rows
        ]
        return {
            "name": name,
            "kind": "core" if name in CORE_NAMES else "user",
            "active_version": active_version,
            "versions": versions,
            "sessions": sessions,
            "call_count": sum(item["call_count"] for item in versions),
            "success_count": sum(item["success_count"] for item in versions),
            "failure_count": sum(item["failure_count"] for item in versions),
        }

    def delete_tool(
        self, name: str, *, session_id: str | None = None
    ) -> dict[str, Any]:
        """Remove a user tool from the active toolbox while retaining its history."""
        if name in CORE_NAMES:
            raise ToolboxError("core_tool_required", f"cannot delete core tool: {name}")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT v.version
                    FROM bindings AS b
                    JOIN tool_versions AS v ON v.id = b.tool_id
                    WHERE b.name = ?
                    """,
                    (name,),
                ).fetchone()
                if row is None:
                    raise ToolboxError(
                        "unknown_active_tool", f"unknown active tool: {name}"
                    )
                version = int(row["version"])
                connection.execute("DELETE FROM bindings WHERE name = ?", (name,))
                if session_id is not None:
                    self._append_event_connection(
                        connection,
                        session_id,
                        "binding_deleted",
                        {"name": name, "from_version": version},
                        tool_name=name,
                        tool_version=version,
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {"name": name, "deleted_version": version, "active": False}

    def rollback(self, name: str, *, version: int) -> dict[str, Any]:
        """Move a binding without relying on any editable tool implementation."""
        current = self._active_version(name)
        if current is None:
            raise ToolboxError("unknown_tool", f"unknown active tool: {name}")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT id, version FROM tool_versions
                    WHERE name = ? AND version = ?
                    """,
                    (name, version),
                ).fetchone()
                if row is None:
                    raise ToolboxError(
                        "no_rollback",
                        f"no requested rollback version exists for {name}",
                    )
                target = int(row["version"])
                connection.execute(
                    "UPDATE bindings SET tool_id = ? WHERE name = ?",
                    (int(row["id"]), name),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {"name": name, "from_version": current, "to_version": target}
