"""Immutable host kernel for persistent, dynamically authored tools."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import math
import os
import re
import sqlite3
import time
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mechagnome.bootstrap import (
    BOOTSTRAP_TOOLS,
    CORE_NAMES,
    CORE_SCHEMAS,
    NAMESPACE_PATH_MAX,
    NAMESPACE_PATH_PATTERN,
)

if TYPE_CHECKING:
    from mechagnome.model_provider import _BoundedModelProvider

JsonValue = Any

_SCHEMA_VERSION = 13
_SESSION_KINDS = frozenset({"generic", "conversation", "completion"})
_SESSION_METADATA_LIMITS = {"title": 256, "description": 4096}
_SESSION_ANNOTATION_FIELDS = frozenset(_SESSION_METADATA_LIMITS)
_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_TOOLBOX_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
_NAMESPACE_PATH = re.compile(NAMESPACE_PATH_PATTERN)


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
    legacy_executor: ThreadPoolExecutor
    calls: int = 0

    @property
    def session_id(self) -> str:
        return self.scope.session_id


@dataclass(frozen=True)
class _ResolvedTool:
    id: int
    lineage_id: int
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


def _normalize_session_metadata(changes: Any) -> dict[str, str | None]:
    """Validate and normalize mutable session display metadata."""
    if not isinstance(changes, dict) or not changes:
        raise ToolboxError(
            "invalid_session_metadata",
            "session metadata update requires title or description",
        )
    if not all(isinstance(name, str) for name in changes):
        raise ToolboxError(
            "invalid_session_metadata",
            "session metadata keys must be strings",
        )
    unknown = set(changes) - _SESSION_METADATA_LIMITS.keys()
    if unknown:
        raise ToolboxError(
            "invalid_session_metadata",
            f"unsupported session metadata: {', '.join(sorted(unknown))}",
        )
    normalized: dict[str, str | None] = {}
    for name, value in changes.items():
        if value is not None and not isinstance(value, str):
            raise ToolboxError(
                "invalid_session_metadata",
                f"session {name} must be a string or null",
            )
        if value is not None:
            try:
                size = len(value.encode("utf-8"))
            except UnicodeError as error:
                raise ToolboxError(
                    "invalid_session_metadata",
                    f"session {name} is not valid UTF-8",
                ) from error
            if size > _SESSION_METADATA_LIMITS[name]:
                raise ToolboxError(
                    "invalid_session_metadata",
                    f"session {name} exceeds {_SESSION_METADATA_LIMITS[name]} bytes",
                )
        normalized[name] = value
    return normalized


class ToolSession:
    """One durable session exposed through the authored-tool context."""

    def __init__(
        self,
        kernel: Kernel,
        provider: _BoundedModelProvider,
        metadata: dict[str, Any],
        caller_session_id: str,
        actor_call_id: str | None = None,
    ) -> None:
        self._kernel = kernel
        self._provider = provider
        self._metadata = metadata
        self._caller_session_id = caller_session_id
        self._actor_call_id = actor_call_id
        self.id = str(metadata["id"])

    @property
    def metadata(self) -> dict[str, Any]:
        """Return durable identity, lineage, and inherited-context metadata."""
        return dict(self._metadata)

    @property
    def kind(self) -> str:
        return str(self.metadata["kind"])

    @property
    def parent_id(self) -> str | None:
        value = self.metadata["parent_session_id"]
        return None if value is None else str(value)

    @property
    def root_id(self) -> str:
        return str(self.metadata["root_session_id"])

    @property
    def origin_session_id(self) -> str | None:
        value = self.metadata["origin_session_id"]
        return None if value is None else str(value)

    @property
    def origin_call_id(self) -> str | None:
        value = self.metadata["origin_call_id"]
        return None if value is None else str(value)

    @property
    def title(self) -> str | None:
        value = self.metadata["title"]
        return None if value is None else str(value)

    @property
    def description(self) -> str | None:
        value = self.metadata["description"]
        return None if value is None else str(value)

    def update_metadata(
        self,
        *,
        expected_revision: int | None = None,
        **changes: str | None,
    ) -> dict[str, Any]:
        """Update display metadata; keep a supplied title to four words or fewer."""
        self._metadata = self._kernel._update_session_metadata(
            self.id,
            changes,
            caller_session_id=self._caller_session_id,
            actor_call_id=self._actor_call_id,
            expected_revision=expected_revision,
        )
        return self.metadata

    def read(self, after: int = 0, limit: int = 50) -> dict[str, Any]:
        """Read a page of committed events from this session."""
        return self._kernel.read_session(self.id, after=after, limit=limit)

    async def prompt(
        self,
        prompt: str,
        *,
        mode: str = "continue",
        detach: bool = False,
        metadata: dict[str, str | None] | None = None,
    ) -> JsonValue:
        """Prompt this conversation; keep metadata titles to four words or fewer."""
        request: dict[str, Any] = {
            "session_id": self.id,
            "prompt": prompt,
            "mode": mode,
            "detach": detach,
        }
        if metadata is not None:
            request["metadata"] = metadata
        return await asyncio.to_thread(
            self._provider.prompt_session,
            request,
        )


class SessionAccess:
    """Bounded durable-session access exposed to every authored tool."""

    def __init__(
        self,
        kernel: Kernel,
        session_id: str,
        call_id: str,
        provider: _BoundedModelProvider,
    ) -> None:
        self._kernel = kernel
        self._provider = provider
        self._call_id = call_id
        self.id = session_id

    def get(self, session_id: str | None = None) -> ToolSession:
        """Return a stable handle for the current or selected durable session."""
        identifier = self.id if session_id is None else session_id
        if not isinstance(identifier, str) or not identifier:
            raise ToolboxError("unknown_session", f"unknown session: {identifier}")
        metadata = self._kernel.session_metadata(identifier)
        return ToolSession(
            self._kernel,
            self._provider,
            metadata,
            self.id,
            self._call_id,
        )

    async def inspect(self, job_id: str) -> JsonValue:
        """Inspect one detached session-prompt job visible to this call tree."""
        return await asyncio.to_thread(
            self._provider.prompt_session,
            {"job_id": job_id},
        )

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

    def set_title(
        self,
        title: str | None,
        *,
        session_id: str | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Set or clear a short title, using no more than four words."""
        return self._kernel._set_session_annotation(
            self.id if session_id is None else session_id,
            field="title",
            value=title,
            expected_revision=expected_revision,
            actor_session_id=self.id,
            actor_call_id=self._call_id,
        )

    def set_description(
        self,
        description: str | None,
        *,
        session_id: str | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Set or clear one saved session's description."""
        return self._kernel._set_session_annotation(
            self.id if session_id is None else session_id,
            field="description",
            value=description,
            expected_revision=expected_revision,
            actor_session_id=self.id,
            actor_call_id=self._call_id,
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
        from mechagnome.model_provider import ToolModelProvider

        provider = state.model_provider.for_origin(call_id)
        self.sessions = SessionAccess(kernel, state.session_id, call_id, provider)
        self.model_provider = ToolModelProvider(provider)

    @property
    def kernel(self) -> _KernelCapability:
        """Return the narrow capability belonging to a distinguished core slot."""
        if self._logical_slot is None:
            raise ToolboxError(
                "capability_denied",
                "ordinary tools do not receive a core kernel capability",
            )
        return _KernelCapability(self)

    async def call_tool(
        self, name: str, args: dict[str, Any], version: int | None = None
    ) -> JsonValue:
        """Invoke a tool through the snapshotted editable dispatcher."""
        envelope: dict[str, Any] = {"name": name, "args": args}
        if version is not None:
            envelope["version"] = version
        return await self._kernel._invoke_async(
            "call_tool",
            envelope,
            state=self._state,
            parent_call_id=self._call_id,
            depth=self._depth + 1,
        )


class _LegacyToolModelProvider:
    """Synchronous facade for tools persisted before the async ABI."""

    def __init__(self, provider: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._provider = provider
        self._loop = loop

    def complete(self, messages: Any) -> str:
        return asyncio.run_coroutine_threadsafe(
            self._provider.complete(messages), self._loop
        ).result()

    def run_agent(self, prompt: str) -> str:
        return asyncio.run_coroutine_threadsafe(
            self._provider.run_agent(prompt), self._loop
        ).result()


class _LegacyKernelCapability:
    """Preserve the pre-async core capability contract for stored sources."""

    def __init__(
        self, capability: _KernelCapability, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._capability = capability
        self._loop = loop

    def execute(
        self, name: str, args: dict[str, Any], version: int | None = None
    ) -> JsonValue:
        return asyncio.run_coroutine_threadsafe(
            self._capability.execute(name, args, version), self._loop
        ).result()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._capability, name)


class _LegacyToolContext:
    """Compatibility context for synchronous tools already stored on disk."""

    def __init__(self, context: ToolContext, loop: asyncio.AbstractEventLoop) -> None:
        self._context = context
        self._loop = loop
        self.caller_session_id = context.caller_session_id
        self.sessions = context.sessions
        self.model_provider = _LegacyToolModelProvider(context.model_provider, loop)

    @property
    def kernel(self) -> _LegacyKernelCapability:
        return _LegacyKernelCapability(self._context.kernel, self._loop)

    def call_tool(
        self, name: str, args: dict[str, Any], version: int | None = None
    ) -> JsonValue:
        return asyncio.run_coroutine_threadsafe(
            self._context.call_tool(name, args, version), self._loop
        ).result()


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

    def catalog(
        self, include_core: bool = True, namespace: str | None = None
    ) -> list[dict[str, Any]]:
        """Return effective metadata to the search implementation."""
        self._require("search_tools")
        return self._context._kernel.catalog(
            include_core=include_core,
            namespace=namespace,
            scope=self._context._state.scope,
        )

    def list_tools(self, namespace: str | None = None) -> list[dict[str, Any]]:
        """Return effective metadata to the paged tool listing."""
        self._require("list_tools")
        return self._context._kernel.catalog(
            namespace=namespace,
            scope=self._context._state.scope,
        )

    def list_tool_namespaces(
        self, namespace: str | None = None
    ) -> list[dict[str, Any]]:
        """Return hierarchical namespace counts to the paged listing."""
        self._require("list_tool_namespaces")
        return self._context._kernel.list_tool_namespaces(
            namespace=namespace,
            scope=self._context._state.scope,
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
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        source: str | None = None,
        base_version: int | None = None,
        namespaces: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store source through the write capability."""
        self._require("write_tool")
        return self._context._kernel.write_tool(
            name=name,
            description=description,
            input_schema=input_schema,
            source=source,
            base_version=base_version,
            namespaces=namespaces,
            session_id=self._context._state.session_id,
            parent_call_id=self._context._call_id,
            scope=self._context._state.scope,
        )

    def delete_tool(self, name: str, version: int | None = None) -> dict[str, Any]:
        """Remove a tool binding or specific version through the delete capability."""
        self._require("delete_tool")
        return self._context._kernel.delete_tool(
            name,
            version=version,
            session_id=self._context._state.session_id,
            scope=self._context._state.scope,
        )

    async def execute(
        self, name: str, args: dict[str, Any], version: int | None = None
    ) -> JsonValue:
        """Bottom out recursive dispatch through the call capability."""
        self._require("call_tool")
        return await self._context._kernel._invoke_async(
            name,
            args,
            state=self._context._state,
            parent_call_id=self._context._call_id,
            depth=self._context._depth + 1,
            version=version,
        )


class Kernel:
    """Persistent host substrate below the editable core operations."""

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
    def _is_sqlite_contention(error: sqlite3.OperationalError) -> bool:
        """Return whether an SQLite failure is a busy/locked contention result."""
        code = getattr(error, "sqlite_errorcode", None)
        if not isinstance(code, int):
            return False
        primary_code = code & 0xFF
        return primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}

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
                    if version == 6:
                        self._migrate_v6(connection)
                        version = 7
                    if version == 7:
                        self._migrate_v7(connection)
                        version = 8
                    if version == 8:
                        self._migrate_v8(connection)
                        version = 9
                    if version == 9:
                        self._migrate_v9(connection)
                        version = 10
                    if version == 10:
                        self._migrate_v10(connection)
                        version = 11
                    if version == 11:
                        self._migrate_v11(connection)
                        version = 12
                    if version == 12:
                        self._migrate_v12(connection)
                        version = 13
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
    def _migrate_v6(connection: sqlite3.Connection) -> None:
        """Add hierarchical namespace memberships to every tool lineage."""
        Kernel._create_namespace_schema(connection)
        Kernel._backfill_namespaces(connection)

    def _migrate_v7(self, connection: sqlite3.Connection) -> None:
        """Reserve and seed the two tool-listing core slots."""
        self._reserve_new_core_names(connection, ("list_tools", "list_tool_namespaces"))
        for row in connection.execute("SELECT id FROM toolboxes").fetchall():
            self._seed_missing_core(connection, str(row["id"]))

    def _migrate_v8(self, connection: sqlite3.Connection) -> None:
        """Reserve and seed the delete_tool core slot."""
        self._reserve_new_core_names(connection, ("delete_tool",))
        for row in connection.execute("SELECT id FROM toolboxes").fetchall():
            self._seed_missing_core(connection, str(row["id"]))

    def _migrate_v9(self, connection: sqlite3.Connection) -> None:
        """Reserve and seed the run_agent core slot."""
        self._reserve_new_core_names(connection, ("run_agent",))
        for row in connection.execute("SELECT id FROM toolboxes").fetchall():
            self._seed_missing_core(connection, str(row["id"]))

    def _migrate_v10(self, connection: sqlite3.Connection) -> None:
        """Add durable origin and fork-context metadata to sessions."""
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(sessions)")
        }
        if "origin_session_id" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN origin_session_id TEXT "
                "REFERENCES sessions(id)"
            )
        if "context_source_session_id" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN context_source_session_id TEXT "
                "REFERENCES sessions(id)"
            )
        if "context_through_seq" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN context_through_seq INTEGER"
            )
        connection.execute(
            "UPDATE sessions SET origin_session_id = parent_session_id "
            "WHERE origin_call_id IS NOT NULL AND origin_session_id IS NULL"
        )
        connection.execute("DROP TRIGGER IF EXISTS sessions_lineage_immutable")
        self._create_session_indexes_and_triggers(connection)

    @staticmethod
    def _migrate_v11(connection: sqlite3.Connection) -> None:
        """Add mutable, human-facing session metadata."""
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(sessions)")
        }
        if "title" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
        if "description" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN description TEXT")

    def _migrate_v12(self, connection: sqlite3.Connection) -> None:
        """Add revisioned annotations and ensure the run_agent core slot."""
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(sessions)")
        }
        if "title" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
        if "description" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN description TEXT")
        if "annotation_revision" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN annotation_revision INTEGER NOT NULL "
                "DEFAULT 0 CHECK(typeof(annotation_revision) = 'integer' "
                "AND annotation_revision >= 0)"
            )
        self._reserve_new_core_names(connection, ("run_agent",))
        for row in connection.execute("SELECT id FROM toolboxes").fetchall():
            self._seed_missing_core(connection, str(row["id"]))

    @staticmethod
    def _reserve_new_core_names(
        connection: sqlite3.Connection, names: tuple[str, ...]
    ) -> None:
        """Keep pre-core user lineages callable under collision-free names."""
        defaults = {tool.name: tool for tool in BOOTSTRAP_TOOLS}
        for toolbox_row in connection.execute("SELECT id FROM toolboxes").fetchall():
            toolbox_id = str(toolbox_row["id"])
            for name in names:
                lineage = connection.execute(
                    "SELECT id FROM tool_lineages WHERE toolbox_id = ? AND name = ?",
                    (toolbox_id, name),
                ).fetchone()
                if lineage is None:
                    continue
                version_one = connection.execute(
                    "SELECT description, schema_json, source FROM tool_versions "
                    "WHERE lineage_id = ? AND version = 1",
                    (int(lineage["id"]),),
                ).fetchone()
                default = defaults[name]
                if version_one is not None and (
                    str(version_one["description"]) == default.description
                    and str(version_one["schema_json"]) == _json(default.input_schema)
                    and str(version_one["source"]) == default.source
                ):
                    continue
                suffix = 1
                legacy_name = f"{name}_legacy"
                while (
                    connection.execute(
                        "SELECT 1 FROM tool_lineages WHERE toolbox_id = ? AND name = ?",
                        (toolbox_id, legacy_name),
                    ).fetchone()
                    is not None
                ):
                    suffix += 1
                    legacy_name = f"{name}_legacy_{suffix}"
                connection.execute(
                    "UPDATE tool_lineages SET name = ? WHERE id = ?",
                    (legacy_name, int(lineage["id"])),
                )
                connection.execute(
                    "UPDATE bindings SET name = ? WHERE toolbox_id = ? AND name = ?",
                    (legacy_name, toolbox_id, name),
                )

    @staticmethod
    def _create_namespace_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_namespaces (
                lineage_id INTEGER NOT NULL
                    REFERENCES tool_lineages(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                PRIMARY KEY(lineage_id, path)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS tool_namespaces_path_lineage "
            "ON tool_namespaces(path, lineage_id)"
        )

    @staticmethod
    def _backfill_namespaces(connection: sqlite3.Connection) -> None:
        for row in connection.execute("SELECT id, name FROM tool_lineages"):
            path = "core" if str(row["name"]) in CORE_NAMES else "uncategorized"
            connection.execute(
                "INSERT OR IGNORE INTO tool_namespaces (lineage_id, path) "
                "VALUES (?, ?)",
                (int(row["id"]), path),
            )

    @staticmethod
    def _create_session_indexes_and_triggers(
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute(
            "CREATE INDEX IF NOT EXISTS sessions_parent_created "
            "ON sessions(parent_session_id, created_at DESC)"
        )
        available = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(sessions)")
        }
        candidates = (
            "parent_session_id",
            "kind",
            "origin_session_id",
            "origin_call_id",
            "context_source_session_id",
            "context_through_seq",
        )
        immutable = tuple(name for name in candidates if name in available)
        updated = ", ".join(immutable)
        changed = " OR ".join(f"OLD.{name} IS NOT NEW.{name}" for name in immutable)
        connection.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS sessions_lineage_immutable
            BEFORE UPDATE OF {updated} ON sessions
            WHEN {changed}
            BEGIN
                SELECT RAISE(ABORT, 'session lineage is immutable');
            END
            """
        )
        if {
            "parent_session_id",
            "context_source_session_id",
            "context_through_seq",
        } <= available:
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS sessions_context_valid
                BEFORE INSERT ON sessions
                WHEN (
                    NEW.context_source_session_id IS NULL
                    AND NEW.context_through_seq IS NOT NULL
                ) OR (
                    NEW.context_source_session_id IS NOT NULL
                    AND NEW.context_through_seq IS NULL
                ) OR (
                    NEW.context_source_session_id IS NOT NULL
                    AND NEW.context_source_session_id IS NOT NEW.parent_session_id
                )
                BEGIN
                    SELECT RAISE(ABORT, 'invalid session context');
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
            CREATE TABLE tool_namespaces (
                lineage_id INTEGER NOT NULL
                    REFERENCES tool_lineages(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                PRIMARY KEY(lineage_id, path)
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
                origin_session_id TEXT REFERENCES sessions(id),
                origin_call_id TEXT,
                context_source_session_id TEXT REFERENCES sessions(id),
                context_through_seq INTEGER
                    CHECK(context_through_seq IS NULL OR context_through_seq >= 0),
                title TEXT,
                description TEXT,
                annotation_revision INTEGER NOT NULL DEFAULT 0
                    CHECK(typeof(annotation_revision) = 'integer'
                        AND annotation_revision >= 0),
                created_at TEXT NOT NULL,
                CHECK(
                    (context_source_session_id IS NULL) =
                    (context_through_seq IS NULL)
                ),
                CHECK(
                    context_source_session_id IS NULL
                    OR context_source_session_id = parent_session_id
                )
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
            "CREATE INDEX tool_namespaces_path_lineage "
            "ON tool_namespaces(path, lineage_id)",
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
        self._reserve_new_core_names(
            connection,
            ("list_tools", "list_tool_namespaces", "delete_tool", "run_agent"),
        )
        self._seed_missing_core(connection, toolbox_id)
        self._backfill_namespaces(connection)

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
            connection.execute(
                "INSERT INTO tool_namespaces (lineage_id, path) VALUES (?, 'core')",
                (int(lineage.lastrowid),),
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
        parent_scope: InvocationScope,
        *,
        kind: str,
        origin_scope: InvocationScope | None = None,
        origin_call_id: str | None = None,
        context_source_session_id: str | None = None,
        context_through_seq: int | None = None,
    ) -> str:
        """Create a host-identified child inheriting one frozen invocation scope."""
        if kind not in _SESSION_KINDS:
            raise ToolboxError("invalid_session_kind", f"invalid child kind: {kind}")
        if (context_source_session_id is None) != (context_through_seq is None):
            raise ToolboxError(
                "invalid_session_context",
                "fork context requires both a source session and event sequence",
            )
        if context_through_seq is not None and (
            isinstance(context_through_seq, bool)
            or not isinstance(context_through_seq, int)
            or context_through_seq < 0
        ):
            raise ToolboxError(
                "invalid_session_context",
                "fork context sequence must be a non-negative integer",
            )
        caller_scope = origin_scope or parent_scope
        identifier = uuid.uuid4().hex
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                parent = connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ?", (parent_scope.session_id,)
                ).fetchone()
                if parent is None:
                    raise ToolboxError(
                        "unknown_session", f"unknown session: {parent_scope.session_id}"
                    )
                if context_source_session_id is not None:
                    if context_source_session_id != parent_scope.session_id:
                        raise ToolboxError(
                            "invalid_session_context",
                            "fork context must come from the child session's parent",
                        )
                    latest = self._latest_completed_sequence(
                        connection, context_source_session_id
                    )
                    if context_through_seq != latest:
                        raise ToolboxError(
                            "invalid_session_context",
                            "fork context must end at the latest completed turn",
                            latest_completed_seq=latest,
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
                        (caller_scope.session_id, origin_call_id),
                    ).fetchone()
                    if origin is None:
                        raise ToolboxError(
                            "invalid_session_origin",
                            "origin call is not the active leaf in the parent session",
                        )
                connection.execute(
                    """
                    INSERT INTO sessions (
                        id, cwd, parent_session_id, kind, origin_session_id,
                        origin_call_id, context_source_session_id,
                        context_through_seq, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        parent_scope.cwd,
                        parent_scope.session_id,
                        kind,
                        caller_scope.session_id if origin_call_id is not None else None,
                        origin_call_id,
                        context_source_session_id,
                        context_through_seq,
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
                        for position, toolbox_id in enumerate(parent_scope.toolbox_ids)
                    ],
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return identifier

    @staticmethod
    def _latest_completed_sequence(
        connection: sqlite3.Connection, session_id: str
    ) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(seq), 0) AS seq FROM events "
            "WHERE session_id = ? AND kind IN ('final', 'cancelled')",
            (session_id,),
        ).fetchone()
        return int(row["seq"])

    def latest_completed_sequence(self, session_id: str) -> int:
        """Return the durable boundary of the latest completed conversation turn."""
        with closing(self._connect()) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                raise ToolboxError("unknown_session", f"unknown session: {session_id}")
            return self._latest_completed_sequence(connection, session_id)

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
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                sequence = self._append_event_connection(
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
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return sequence

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
                       sessions.kind, sessions.origin_session_id,
                       sessions.origin_call_id, sessions.context_source_session_id,
                       sessions.context_through_seq, sessions.title,
                       sessions.description, sessions.annotation_revision,
                       sessions.created_at,
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
            return self._session_metadata_connection(connection, session_id)

    def _session_metadata_connection(
        self, connection: sqlite3.Connection, session_id: str
    ) -> dict[str, Any]:
        """Return session metadata from an existing connection snapshot."""
        row = connection.execute(
            """
            SELECT id, cwd, parent_session_id, kind, origin_session_id,
                   origin_call_id, context_source_session_id, context_through_seq,
                   title, description, annotation_revision, created_at
            FROM sessions WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise ToolboxError("unknown_session", f"unknown session: {session_id}")
        metadata = dict(row)
        metadata["root_session_id"] = self._root_session_id(connection, session_id)
        return metadata

    def _set_session_annotation(
        self,
        session_id: str,
        *,
        field: str,
        value: str | None,
        expected_revision: int | None,
        actor_session_id: str,
        actor_call_id: str,
    ) -> dict[str, Any]:
        """Atomically set one mutable session field with optional revision checking."""
        if field not in _SESSION_ANNOTATION_FIELDS:
            raise AssertionError(f"unsupported session annotation field: {field}")
        if value is not None and not isinstance(value, str):
            raise ToolboxError(
                "invalid_session_annotation",
                f"session {field} must be a string or None",
                field=field,
            )
        return self._update_session_metadata(
            session_id,
            {field: value},
            caller_session_id=actor_session_id,
            actor_call_id=actor_call_id,
            expected_revision=expected_revision,
        )

    def _update_session_metadata(
        self,
        session_id: str,
        changes: dict[str, Any],
        *,
        caller_session_id: str,
        actor_call_id: str | None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Atomically update revisioned display metadata within one session tree."""
        normalized = _normalize_session_metadata(changes)
        if expected_revision is not None and (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ToolboxError(
                "invalid_annotation_revision",
                "expected_revision must be a non-negative integer",
            )

        with closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                target_root = self._root_session_id(connection, session_id)
                caller_root = self._root_session_id(connection, caller_session_id)
                if target_root != caller_root:
                    raise ToolboxError(
                        "session_access_denied",
                        "session metadata updates are limited to the caller's "
                        "session tree",
                    )
                row = connection.execute(
                    "SELECT title, description, annotation_revision "
                    "FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    raise ToolboxError(
                        "unknown_session", f"unknown session: {session_id}"
                    )
                actual_revision = int(row["annotation_revision"])
                if (
                    expected_revision is not None
                    and expected_revision != actual_revision
                ):
                    raise ToolboxError(
                        "session_annotation_conflict",
                        "session annotations changed since they were read",
                        session_id=session_id,
                        expected_revision=expected_revision,
                        actual_revision=actual_revision,
                    )
                changed = [
                    (field, row[field], value)
                    for field, value in normalized.items()
                    if row[field] != value
                ]
                if not changed:
                    metadata = self._session_metadata_connection(connection, session_id)
                    connection.commit()
                    return metadata

                revision = actual_revision
                for field, old_value, value in changed:
                    revision += 1
                    connection.execute(
                        f"UPDATE sessions SET {field} = ?, annotation_revision = ? "
                        "WHERE id = ?",
                        (value, revision, session_id),
                    )
                    self._append_event_connection(
                        connection,
                        session_id,
                        "session_annotation_changed",
                        {
                            "field": field,
                            "old_value": old_value,
                            "new_value": value,
                            "annotation_revision": revision,
                            "actor_session_id": caller_session_id,
                            "actor_call_id": actor_call_id,
                        },
                    )
                metadata = self._session_metadata_connection(connection, session_id)
                connection.commit()
                return metadata
            except sqlite3.OperationalError as error:
                if connection.in_transaction:
                    connection.rollback()
                if self._is_sqlite_contention(error):
                    raise ToolboxError(
                        "session_annotation_busy",
                        "session annotations are busy; retry the update",
                        session_id=session_id,
                        retryable=True,
                    ) from error
                raise
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def read_session(
        self, session_id: str, *, after: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        """Read committed events in stable sequence order."""
        limit = min(100, max(1, limit))
        after = max(0, after)
        with closing(self._connect()) as connection:
            metadata = self._session_metadata_connection(connection, session_id)
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

    def latest_event_sequence(self, session_id: str) -> int:
        """Return the latest committed sequence for one existing session."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT sessions.id, COALESCE(MAX(events.seq), 0) AS latest_seq
                FROM sessions
                LEFT JOIN events ON events.session_id = sessions.id
                WHERE sessions.id = ?
                GROUP BY sessions.id
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            raise ToolboxError("unknown_session", f"unknown session: {session_id}")
        return int(row["latest_seq"])

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
                lineage_id=int(binding["lineage_id"]),
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
        """Return the fixed model-facing operation definitions."""
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
        namespace: str | None = None,
        session_id: str | None = None,
        scope: InvocationScope | None = None,
    ) -> list[dict[str, Any]]:
        """Return effective metadata for an editable search implementation."""
        namespace_filter = (
            self._normalize_namespace_path(namespace) if namespace is not None else None
        )
        active_scope = self._scope(session_id=session_id, scope=scope)
        with closing(self._connect()) as connection:
            rows = self._effective_rows(connection, active_scope)
            namespaces = self._lineage_namespace_map(
                connection, (int(row["lineage_id"]) for row in rows)
            )
        tools = []
        for row in rows:
            name = str(row["name"])
            if not include_core and name in CORE_NAMES:
                continue
            paths = namespaces[int(row["lineage_id"])]
            if namespace_filter is not None and not any(
                path == namespace_filter or path.startswith(f"{namespace_filter}/")
                for path in paths
            ):
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
                    "namespaces": list(paths),
                }
            )
        return tools

    def list_tool_namespaces(
        self,
        *,
        namespace: str | None = None,
        session_id: str | None = None,
        scope: InvocationScope | None = None,
    ) -> list[dict[str, Any]]:
        """Return namespace paths with recursive de-duplicated tool counts."""
        namespace_filter = (
            self._normalize_namespace_path(namespace) if namespace is not None else None
        )
        tools = self.catalog(session_id=session_id, scope=scope)
        memberships: dict[str, set[str]] = {}
        for tool in tools:
            for assigned_path in tool["namespaces"]:
                if namespace_filter is not None and not (
                    assigned_path == namespace_filter
                    or assigned_path.startswith(f"{namespace_filter}/")
                ):
                    continue
                parts = assigned_path.split("/")
                for depth in range(1, len(parts) + 1):
                    path = "/".join(parts[:depth])
                    if namespace_filter is not None and not (
                        path == namespace_filter
                        or path.startswith(f"{namespace_filter}/")
                    ):
                        continue
                    memberships.setdefault(path, set()).add(tool["name"])
        return [
            {"namespace": path, "tool_count": len(names)}
            for path, names in sorted(memberships.items())
        ]

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
            namespaces = self._lineage_namespaces(connection, tool.lineage_id)
        return {
            "name": tool.name,
            "version": tool.version,
            "active": active.version == tool.version,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "source": tool.source,
            "kind": "core" if name in CORE_NAMES else "user",
            "toolbox": tool.toolbox_name,
            "namespaces": list(namespaces),
        }

    def write_tool(
        self,
        *,
        name: str,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        source: str | None = None,
        base_version: int | None = None,
        namespaces: list[str] | None = None,
        session_id: str | None = None,
        parent_call_id: str | None = None,
        scope: InvocationScope | None = None,
        toolbox: str | None = None,
    ) -> dict[str, Any]:
        """Create a tool version or replace its discovery namespaces."""
        authoring_values = (description, input_schema, source)
        full_write = all(value is not None for value in authoring_values)
        if any(value is not None for value in authoring_values) and not full_write:
            raise ToolboxError(
                "invalid_write",
                "description, input_schema, and source must be supplied together",
            )
        if not full_write and namespaces is None:
            raise ToolboxError(
                "invalid_write",
                "write_tool requires tool source or namespace assignments",
            )
        normalized_namespaces = (
            self._normalize_namespaces(namespaces) if namespaces is not None else None
        )
        if full_write:
            assert description is not None
            assert input_schema is not None
            assert source is not None
            self._validate_tool(name, description, input_schema, source)
        elif not isinstance(name, str) or _TOOL_NAME.fullmatch(name) is None:
            raise ToolboxError("invalid_name", f"invalid tool name: {name!r}")
        if full_write and name in CORE_SCHEMAS and input_schema != CORE_SCHEMAS[name]:
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
                        if not full_write:
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
                    if not full_write:
                        raise ToolboxError("unknown_tool", f"unknown tool {name!r}")
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
                    SELECT v.id, v.version
                    FROM bindings AS b
                    JOIN tool_versions AS v ON v.id = b.tool_version_id
                    WHERE b.toolbox_id = ? AND b.name = ?
                    """,
                    (target_id, name),
                ).fetchone()
                active_version = (
                    int(active_row["version"]) if active_row is not None else None
                )
                if not full_write and active_row is None:
                    raise ToolboxError("unknown_tool", f"unknown tool {name!r}")
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
                previous_namespaces = self._lineage_namespaces(connection, lineage_id)
                next_namespaces = normalized_namespaces
                if next_namespaces is None:
                    next_namespaces = previous_namespaces or (
                        "core" if name in CORE_NAMES else "uncategorized",
                    )
                if full_write:
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
                else:
                    version = active_version
                    assert version is not None
                    version_id = int(active_row["id"])
                self._replace_namespaces(connection, lineage_id, next_namespaces)
                if session_id is not None and previous_namespaces != next_namespaces:
                    self._append_event_connection(
                        connection,
                        session_id,
                        "tool_namespaces_changed",
                        {
                            "name": name,
                            "before": previous_namespaces,
                            "after": next_namespaces,
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
        result = {
            "name": name,
            "version": version,
            "active": True,
            "namespaces": list(next_namespaces),
        }
        if full_write:
            result["previous_version"] = active_version
        else:
            result["metadata_only"] = True
        return result

    @staticmethod
    def _normalize_namespace_path(path: Any) -> str:
        if (
            not isinstance(path, str)
            or len(path) > NAMESPACE_PATH_MAX
            or _NAMESPACE_PATH.fullmatch(path) is None
        ):
            raise ToolboxError("invalid_namespace", f"invalid namespace path: {path!r}")
        return path

    @classmethod
    def _normalize_namespaces(cls, namespaces: Any) -> tuple[str, ...]:
        if not isinstance(namespaces, list) or not namespaces:
            raise ToolboxError(
                "invalid_namespace", "namespaces must be a non-empty array"
            )
        return tuple(
            sorted({cls._normalize_namespace_path(path) for path in namespaces})
        )

    @staticmethod
    def _lineage_namespaces(
        connection: sqlite3.Connection, lineage_id: int
    ) -> tuple[str, ...]:
        return Kernel._lineage_namespace_map(connection, (lineage_id,))[lineage_id]

    @staticmethod
    def _lineage_namespace_map(
        connection: sqlite3.Connection, lineage_ids: Iterable[int]
    ) -> dict[int, tuple[str, ...]]:
        identifiers = tuple(sorted(set(lineage_ids)))
        if not identifiers:
            return {}
        grouped: dict[int, list[str]] = {lineage_id: [] for lineage_id in identifiers}
        placeholders = ",".join("?" for _ in identifiers)
        for row in connection.execute(
            f"SELECT lineage_id, path FROM tool_namespaces "
            f"WHERE lineage_id IN ({placeholders}) ORDER BY lineage_id, path",
            identifiers,
        ):
            grouped[int(row["lineage_id"])].append(str(row["path"]))
        return {lineage_id: tuple(paths) for lineage_id, paths in grouped.items()}

    @staticmethod
    def _replace_namespaces(
        connection: sqlite3.Connection,
        lineage_id: int,
        namespaces: tuple[str, ...],
    ) -> None:
        connection.execute(
            "DELETE FROM tool_namespaces WHERE lineage_id = ?", (lineage_id,)
        )
        connection.executemany(
            "INSERT INTO tool_namespaces (lineage_id, path) VALUES (?, ?)",
            ((lineage_id, path) for path in namespaces),
        )

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
                "missing_main", "source must define async def main(input, ctx)"
            )
        main = mains[-1]
        if not isinstance(main, ast.AsyncFunctionDef):
            raise ToolboxError(
                "sync_main", "main must be async; define async def main(input, ctx)"
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
        """Synchronously run an async top-level invocation in a durable session."""
        return asyncio.run(
            self.call_async(
                name,
                args,
                session_id=session_id,
                version=version,
                scope=scope,
                model_provider=model_provider,
            )
        )

    async def call_async(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session_id: str | None = None,
        version: int | None = None,
        scope: InvocationScope | None = None,
        model_provider: Any | None = None,
    ) -> JsonValue:
        """Start an async top-level invocation in a durable session."""
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
        legacy_executor = ThreadPoolExecutor(max_workers=max(1, self.max_calls))
        try:
            state = _InvocationState(
                scope=active_scope,
                max_depth=self.max_depth,
                max_calls=self.max_calls,
                model_provider=bound_provider,
                legacy_executor=legacy_executor,
            )
            self.append_event(
                active_scope.session_id,
                "invocation_scope",
                {"toolbox_ids": active_scope.toolbox_ids, "cwd": active_scope.cwd},
            )
            return await self._invoke_async(
                name, args, state=state, depth=0, version=version
            )
        finally:
            legacy_executor.shutdown(wait=False, cancel_futures=True)

    async def _invoke_async(
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
            if inspect.iscoroutinefunction(main):
                result = await main(args, context)
            else:
                legacy_context = _LegacyToolContext(context, asyncio.get_running_loop())
                result = await asyncio.get_running_loop().run_in_executor(
                    state.legacy_executor, main, args, legacy_context
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
        """Return visible and historical tools in the selected toolboxes."""
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
        """Return one toolbox-owned lineage's versions and usage."""
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
            lineage_namespaces = self._lineage_namespaces(
                connection, int(lineage["id"])
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
            "namespaces": list(lineage_namespaces),
        }

    def delete_tool(
        self,
        name: str,
        *,
        version: int | None = None,
        session_id: str | None = None,
        scope: InvocationScope | None = None,
        toolbox: str | None = None,
    ) -> dict[str, Any]:
        """Remove one toolbox binding or a specific version, retaining lineage."""
        if name in CORE_NAMES and (version is None or version == 1):
            raise ToolboxError(
                "core_tool_required",
                f"cannot delete core tool {name}"
                + (" version 1" if version == 1 else ""),
            )
        active_scope = self._scope(session_id=session_id, scope=scope)
        if version is not None:
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
                    version_row = connection.execute(
                        """
                        SELECT id, version FROM tool_versions
                        WHERE lineage_id = ? AND version = ?
                        """,
                        (tool.lineage_id, version),
                    ).fetchone()
                    if version_row is None:
                        raise ToolboxError(
                            "unknown_version",
                            f"version {version} does not exist for tool {name!r}",
                        )
                    # Check it's not the last version
                    count_row = connection.execute(
                        "SELECT COUNT(*) AS count FROM tool_versions"
                        " WHERE lineage_id = ?",
                        (tool.lineage_id,),
                    ).fetchone()
                    if int(count_row["count"]) <= 1:
                        raise ToolboxError(
                            "last_version",
                            f"cannot delete the last remaining version of {name!r}",
                        )

                    # Check if the deleted version is the active one
                    active_row = connection.execute(
                        """
                        SELECT tool_version_id FROM bindings
                        WHERE toolbox_id = ? AND name = ?
                        """,
                        (target, name),
                    ).fetchone()
                    rolled_back_to = None
                    if active_row is not None and int(
                        active_row["tool_version_id"]
                    ) == int(version_row["id"]):
                        # Find the next-highest remaining version
                        prev_row = connection.execute(
                            """
                            SELECT id, version FROM tool_versions
                            WHERE lineage_id = ? AND version < ?
                            ORDER BY version DESC LIMIT 1
                            """,
                            (tool.lineage_id, version),
                        ).fetchone()
                        if prev_row is not None:
                            # Update the binding to point to the previous version
                            connection.execute(
                                """
                                UPDATE bindings SET tool_version_id = ?
                                WHERE toolbox_id = ? AND name = ?
                                """,
                                (int(prev_row["id"]), target, name),
                            )
                            rolled_back_to = int(prev_row["version"])
                        else:
                            # No lower version exists; try the next-higher one
                            next_row = connection.execute(
                                """
                                SELECT id, version FROM tool_versions
                                WHERE lineage_id = ? AND version > ?
                                ORDER BY version ASC LIMIT 1
                                """,
                                (tool.lineage_id, version),
                            ).fetchone()
                            if next_row is not None:
                                connection.execute(
                                    """
                                    UPDATE bindings SET tool_version_id = ?
                                    WHERE toolbox_id = ? AND name = ?
                                    """,
                                    (int(next_row["id"]), target, name),
                                )
                                rolled_back_to = int(next_row["version"])

                    # Null out event references to the deleted version,
                    # log the deletion event, then delete the version row.
                    connection.execute(
                        "UPDATE events SET tool_version_id = NULL"
                        " WHERE tool_version_id = ?",
                        (int(version_row["id"]),),
                    )
                    if session_id is not None:
                        event_payload = {
                            "name": name,
                            "deleted_version": version,
                            "toolbox_id": target,
                        }
                        if rolled_back_to is not None:
                            event_payload["rolled_back_to"] = rolled_back_to
                        self._append_event_connection(
                            connection,
                            session_id,
                            "version_deleted",
                            event_payload,
                            toolbox_id=target,
                            tool_name=name,
                            tool_version=version,
                            tool_version_id=None,
                        )
                    connection.execute(
                        "DELETE FROM tool_versions WHERE id = ?",
                        (int(version_row["id"]),),
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
            result = {"name": name, "deleted_version": version}
            if rolled_back_to is not None:
                result["rolled_back_to"] = rolled_back_to
            return result
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
        """Move one toolbox binding without editable tool code."""
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
