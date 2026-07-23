"""Behavioral tests for the metaprogrammable toolbox proof."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sqlite3
import stat
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from mechagnome import Harness, Kernel, ModelProvider, ModelTurn, ToolboxError, ToolCall
from mechagnome import __main__ as cli
from mechagnome import harness as harness_module
from mechagnome import isolation as isolation_module
from mechagnome import kernel as kernel_module
from mechagnome.bootstrap import CORE_NAMES, CORE_SCHEMAS, HELP_SOURCE
from mechagnome.harness import (
    MODEL_ACTION_NAMES,
    Conversation,
    _parse_run_agent_request,
)
from mechagnome.isolation import IsolatedToolRunner
from mechagnome.model_provider import ModelTransportError


def kernel_at(tmp_path: Path, **kwargs: Any) -> Kernel:
    """Create a kernel with isolated persistent state."""
    return Kernel(tmp_path / "toolbox.db", **kwargs)


def write(
    kernel: Kernel,
    name: str,
    source: str,
    *,
    session_id: str | None = None,
    description: str | None = None,
    input_schema: dict[str, Any] | None = None,
    base_version: int | None = None,
    namespaces: list[str] | None = None,
) -> dict[str, Any]:
    """Write through the editable model-facing operation."""
    args: dict[str, Any] = {
        "name": name,
        "description": description or f"Test tool {name}.",
        "input_schema": input_schema or {"type": "object"},
        "source": source,
    }
    if base_version is not None:
        args["base_version"] = base_version
    if namespaces is not None:
        args["namespaces"] = namespaces
    result = kernel.call("write_tool", args, session_id=session_id)
    assert isinstance(result, dict)
    return result


def call_tool(
    kernel: Kernel,
    name: str,
    args: dict[str, Any],
    *,
    session_id: str | None = None,
    version: int | None = None,
) -> Any:
    """Call through the editable universal dispatcher."""
    envelope: dict[str, Any] = {"name": name, "args": args}
    if version is not None:
        envelope["version"] = version
    return kernel.call("call_tool", envelope, session_id=session_id)


def wait_for_detached(
    runner: IsolatedToolRunner,
    job_id: str,
    session_id: str,
    *,
    status: str | None = None,
    output: str | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Wait for deterministic detached state without assuming a fixed duration."""
    deadline = time.monotonic() + timeout
    while True:
        snapshot = runner.inspect_detached(job_id, session_id=session_id)
        if (status is None or snapshot["status"] == status) and (
            output is None or output in snapshot["output_tail"]
        ):
            return snapshot
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"detached job did not reach expected state: {snapshot}"
            )
        time.sleep(0.01)


def wait_for_agent(
    harness: Harness,
    job_id: str,
    session_id: str,
    *,
    status: str,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Wait for one supervised detached agent to reach a stable state."""
    deadline = time.monotonic() + timeout
    while True:
        snapshot = harness._agent_coordinator.inspect(job_id, session_id=session_id)
        if snapshot["status"] == status:
            return snapshot
        if time.monotonic() >= deadline:
            raise AssertionError(f"detached agent did not reach {status}: {snapshot}")
        time.sleep(0.01)


def test_fresh_bootstrap_is_exact_and_idempotent(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)

    assert tuple(item["name"] for item in kernel.tool_definitions()) == CORE_NAMES
    assert {item["name"] for item in kernel.bindings()} == set(CORE_NAMES)
    assert all(item["kind"] == "core" for item in kernel.bindings())

    reopened = kernel_at(tmp_path)
    assert reopened.bindings() == kernel.bindings()
    assert all(item["versions"] == [1] for item in reopened.bindings())


def test_reopen_refreshes_code_shipped_core_version_one(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    kernel.create_toolbox("secondary")
    stale_source = "async def main(input, ctx):\n    return 'stale default'\n"
    with closing(kernel._connect()) as connection, connection:
        connection.execute(
            """
            UPDATE tool_versions
            SET description = 'stale', schema_json = '{}', source = ?
            WHERE version = 1 AND lineage_id IN (
                SELECT id FROM tool_lineages WHERE name = 'help'
            )
            """,
            (stale_source,),
        )

    reopened = kernel_at(tmp_path)

    assert reopened.call("help", {}).startswith("# Mechagnome help\n")
    refreshed = reopened.view_tool("help", version=1)
    assert refreshed["description"] == (
        "Read progressive documentation for using and extending the toolbox."
    )
    assert refreshed["input_schema"] == CORE_SCHEMAS["help"]
    assert refreshed["source"] == HELP_SOURCE
    assert (
        reopened.view_tool("help", version=1, toolbox="secondary")["source"]
        == HELP_SOURCE
    )


def test_reopen_preserves_active_core_version_two_override(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    replacement_source = "async def main(input, ctx):\n    return 'custom help'\n"
    write(
        kernel,
        "help",
        replacement_source,
        input_schema=CORE_SCHEMAS["help"],
        base_version=1,
    )

    reopened = kernel_at(tmp_path)

    assert reopened.call("help", {}) == "custom help"
    assert reopened.call("help", {}, version=1).startswith("# Mechagnome help\n")
    binding = next(item for item in reopened.bindings() if item["name"] == "help")
    assert binding["active_version"] == 2
    assert binding["versions"] == [1, 2]


def test_bindings_can_order_by_recent_usage_without_changing_payload(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)

    kernel.call("help", {"topic": "quickstart"})
    kernel.call("search_tools", {"query": "missing"})

    assert [item["name"] for item in kernel.bindings()] == sorted(CORE_NAMES)
    recent = kernel.bindings(recent_first=True)
    assert [item["name"] for item in recent[:2]] == ["search_tools", "help"]
    assert set(recent[0]) == {
        "name",
        "active_version",
        "description",
        "versions",
        "kind",
    }


def test_help_returns_complete_packaged_markdown_documents(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)

    toc = kernel.call("help", {})
    assert toc.startswith("# Mechagnome help\n")
    assert "`quickstart`" in toc

    headings = {
        "quickstart": "# Quickstart\n",
        "authoring": "# Authoring tools\n",
        "composition": "# Composing tools\n",
        "sessions": "# Sessions\n",
        "namespaces": "# Hierarchical tool namespaces\n",
        "toolboxes": "# Toolbox stacks\n",
        "versioning": "# Versioning\n",
        "core": "# Core operations\n",
    }
    for topic, heading in headings.items():
        document = kernel.call("help", {"topic": topic})
        assert isinstance(document, str)
        assert document.startswith(heading)

    authoring = kernel.call("help", {"topic": "authoring"})
    assert "```python" in authoring
    assert "## The `ctx` context object" in authoring
    for public_api in (
        "await ctx.call_tool(name, args, version=None)",
        "ctx.caller_session_id",
        "ctx.sessions.current(after=0, limit=50)",
        "ctx.sessions.read(session_id, after=0, limit=50)",
        "ctx.sessions.list(limit=20, cursor=0)",
        "await ctx.model_provider.complete([",
        "await ctx.model_provider.run_agent(",
        "ctx.kernel.catalog(include_core=True)",
        "ctx.kernel.view_tool(name, version=None)",
        "ctx.kernel.write_tool(...)",
        "await ctx.kernel.execute(name, args, version=None)",
    ):
        assert public_api in authoring

    composition = kernel.call("help", {"topic": "composition"})
    assert "Each nested invocation receives its own context object" in composition
    assert "base_version" in composition
    assert "eight delegated-model attempts per top-level call tree" in composition
    assert "remains an awaited foreground call" in composition

    assert "model_provider_unavailable" in authoring
    assert "model_provider_limit" in authoring
    assert "model_provider_protocol" in authoring
    assert "except ToolboxError as error" in authoring
    assert "error.code" in authoring
    assert "1 MiB" in authoring
    assert "256 KiB" in authoring
    assert "bounded JSON transport frame" in authoring
    assert "authenticated data-egress capability" in authoring
    assert "not a monetary" in authoring

    sessions = kernel.call("help", {"topic": "sessions"})
    assert "ctx.sessions.id" in sessions
    assert "next_after" in sessions
    assert "parent_call_id" in sessions
    assert "call_succeeded" in sessions
    assert "call_finished" not in authoring + sessions
    assert "not isinstance(value, bool)" in authoring


def test_help_lists_topics_for_an_unknown_document(tmp_path: Path) -> None:
    result = kernel_at(tmp_path).call("help", {"topic": "missing"})

    assert result["error"] == "unknown help topic: missing"
    assert result["topics"][0] == "toc"


def test_write_immediately_search_view_and_call(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    source = "async def main(input, ctx):\n    return {'echo': input['value']}\n"

    result = write(
        kernel,
        "echo",
        source,
        description="Echo a supplied value for later reuse.",
    )

    assert result == {
        "name": "echo",
        "version": 1,
        "active": True,
        "previous_version": None,
        "namespaces": ["uncategorized"],
    }
    assert call_tool(kernel, "echo", {"value": "hello"}) == {"echo": "hello"}
    search = kernel.call("search_tools", {"query": "echo"})
    assert search["items"][0]["name"] == "echo"
    viewed = kernel.call("view_tool", {"name": "echo"})
    assert viewed["source"] == source
    assert viewed["active"] is True


def test_search_uses_ranked_matches_across_tool_metadata(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "send_email",
        "async def main(input, ctx):\n    return input['recipient_address']\n",
        description="Deliver an electronic message.",
        input_schema={
            "type": "object",
            "properties": {
                "recipient_address": {"type": "string"},
                "subject": {"type": "string"},
            },
        },
    )
    write(
        kernel,
        "weatherForecast",
        "async def main(input, ctx):\n    return 'sunny'\n",
        description="Look up atmospheric conditions.",
    )
    write(
        kernel,
        "archive",
        "async def main(input, ctx):\n    return 'delivery_manifest'\n",
        description="Store a record for later.",
    )
    write(
        kernel,
        "HTTPClient",
        "async def main(input, ctx):\n    return None\n",
        description="Make a remote request.",
    )
    write(
        kernel,
        "notify_user",
        "async def main(input, ctx):\n    return None\n",
        description="Отправить сообщение получателю.",
    )

    partial = kernel.call("search_tools", {"query": "send urgent email"})
    schema = kernel.call("search_tools", {"query": "recipient address"})
    camel_case = kernel.call("search_tools", {"query": "weather forecast"})
    source = kernel.call("search_tools", {"query": "delivery manifest"})
    acronym = kernel.call("search_tools", {"query": "http client"})
    description = kernel.call("search_tools", {"query": "atmospheric conditions"})
    unicode_description = kernel.call("search_tools", {"query": "сообщение"})

    assert partial["items"][0]["name"] == "send_email"
    assert schema["items"][0]["name"] == "send_email"
    assert camel_case["items"][0]["name"] == "weatherForecast"
    assert source["items"][0]["name"] == "archive"
    assert acronym["items"][0]["name"] == "HTTPClient"
    assert description["items"][0]["name"] == "weatherForecast"
    assert unicode_description["items"][0]["name"] == "notify_user"
    assert partial["items"][0] == {
        "name": "send_email",
        "description": "Deliver an electronic message.",
        "namespaces": ["uncategorized"],
    }


def test_search_weights_names_above_other_fields(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel, "rankingmarker_tool", "async def main(input, ctx):\n    return None\n"
    )
    write(
        kernel,
        "description_candidate",
        "async def main(input, ctx):\n    return None\n",
        description="rankingmarker",
    )
    write(
        kernel,
        "schema_candidate",
        "async def main(input, ctx):\n    return None\n",
        input_schema={
            "type": "object",
            "properties": {"rankingmarker": {"type": "string"}},
        },
    )
    write(
        kernel,
        "source_candidate",
        "async def main(input, ctx):\n    return 'rankingmarker'\n",
    )

    result = kernel.call(
        "search_tools", {"query": "rankingmarker", "include_core": False}
    )

    assert [item["name"] for item in result["items"]] == [
        "rankingmarker_tool",
        "description_candidate",
        "schema_candidate",
        "source_candidate",
    ]


def test_search_exact_name_priority_filtering_and_pagination(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    common_source = "async def main(input, ctx):\n    return 'needle needle needle'\n"
    write(kernel, "needle", "async def main(input, ctx):\n    return None\n")
    write(kernel, "haystack", common_source)

    exact = kernel.call(
        "search_tools",
        {"query": "needle", "include_core": False, "limit": 1},
    )
    second = kernel.call(
        "search_tools",
        {
            "query": "needle",
            "include_core": False,
            "limit": 1,
            "cursor": exact["next_cursor"],
        },
    )

    assert exact["items"][0]["name"] == "needle"
    assert exact["total"] == 2
    assert exact["next_cursor"] == 1
    assert second["items"][0]["name"] == "haystack"
    assert second["next_cursor"] is None

    empty = kernel.call(
        "search_tools",
        {"query": "   ", "include_core": False, "limit": 1},
    )
    assert [item["name"] for item in empty["items"]] == ["haystack"]
    assert set(empty["items"][0]) == {"name", "description", "namespaces"}
    assert empty["total"] == 2
    assert empty["next_cursor"] == 1


def test_hierarchical_namespaces_are_mutable_lineage_metadata(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    session_id = kernel.create_session()
    original_source = "async def main(input, ctx):\n    return 'v1'\n"

    created = write(
        kernel,
        "formatter",
        original_source,
        session_id=session_id,
        namespaces=["development/python", "shared", "development/python"],
    )
    assert created["namespaces"] == ["development/python", "shared"]
    before = kernel.tool_history("formatter", session_id=session_id)

    reassigned = kernel.call(
        "write_tool",
        {
            "name": "formatter",
            "namespaces": ["quality/formatting", "development/python"],
            "base_version": 1,
        },
        session_id=session_id,
    )

    assert reassigned == {
        "name": "formatter",
        "version": 1,
        "active": True,
        "namespaces": ["development/python", "quality/formatting"],
        "metadata_only": True,
    }
    after = kernel.tool_history("formatter", session_id=session_id)
    assert after["namespaces"] == ["development/python", "quality/formatting"]
    assert after["versions"] == before["versions"]
    assert kernel.view_tool("formatter", version=1)["source"] == original_source
    assert call_tool(kernel, "formatter", {}, session_id=session_id) == "v1"
    namespace_events = [
        event
        for event in kernel.read_session(session_id, limit=100)["events"]
        if event["kind"] == "tool_namespaces_changed"
    ]
    assert namespace_events[-1]["payload"] == {
        "name": "formatter",
        "before": ["development/python", "shared"],
        "after": ["development/python", "quality/formatting"],
        "toolbox_id": kernel.list_toolboxes()[0]["id"],
    }

    updated = write(
        kernel,
        "formatter",
        "async def main(input, ctx):\n    return 'v2'\n",
        session_id=session_id,
        base_version=1,
    )
    assert updated["namespaces"] == ["development/python", "quality/formatting"]
    assert kernel.tool_history("formatter")["namespaces"] == updated["namespaces"]


@pytest.mark.parametrize(
    "namespaces",
    [[], ["/development"], ["development/"], ["development//python"], ["dev space"]],
)
def test_namespace_assignments_reject_invalid_paths(
    tmp_path: Path, namespaces: list[str]
) -> None:
    kernel = kernel_at(tmp_path)
    write(kernel, "organized", "async def main(input, ctx):\n    return True\n")

    with pytest.raises(ToolboxError) as error:
        kernel.call("write_tool", {"name": "organized", "namespaces": namespaces})

    assert error.value.code == "invalid_namespace"
    assert kernel.view_tool("organized")["namespaces"] == ["uncategorized"]


def test_write_tool_rejects_partial_authoring_and_unknown_reassignment(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)

    with pytest.raises(ToolboxError) as partial:
        kernel.call(
            "write_tool",
            {"name": "partial", "description": "Incomplete.", "namespaces": ["x"]},
        )
    assert partial.value.code == "invalid_write"

    with pytest.raises(ToolboxError) as unknown:
        kernel.call("write_tool", {"name": "missing", "namespaces": ["x"]})
    assert unknown.value.code == "unknown_tool"


def test_search_browses_namespace_subtrees_without_duplicates(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "python_format",
        "async def main(input, ctx):\n    return True\n",
        namespaces=["development/python", "quality/python"],
    )
    write(
        kernel,
        "release",
        "async def main(input, ctx):\n    return True\n",
        namespaces=["development"],
    )
    write(
        kernel,
        "device_info",
        "async def main(input, ctx):\n    return True\n",
        namespaces=["device/python"],
    )

    browsed = kernel.call(
        "search_tools",
        {"query": "", "namespace": "development", "include_core": False},
    )
    assert [item["name"] for item in browsed["items"]] == [
        "python_format",
        "release",
    ]
    assert browsed["total"] == 2
    assert browsed["items"][0]["namespaces"] == [
        "development/python",
        "quality/python",
    ]

    namespace_match = kernel.call(
        "search_tools", {"query": "quality", "include_core": False}
    )
    assert [item["name"] for item in namespace_match["items"]] == ["python_format"]
    exact = kernel.call(
        "search_tools",
        {"query": "", "namespace": "development/python", "include_core": False},
    )
    assert [item["name"] for item in exact["items"]] == ["python_format"]


def test_namespace_filter_respects_winning_toolbox_binding(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db", cwd=tmp_path)
    kernel.create_toolbox("alpha")
    kernel.create_toolbox("beta")
    session_id = kernel.create_session()

    kernel.select_toolboxes(session_id, ["alpha"], mode="use")
    write(
        kernel,
        "same",
        "async def main(input, ctx):\n    return 'alpha'\n",
        session_id=session_id,
        namespaces=["operations"],
    )
    kernel.select_toolboxes(session_id, ["beta"], mode="use")
    write(
        kernel,
        "same",
        "async def main(input, ctx):\n    return 'beta'\n",
        session_id=session_id,
        namespaces=["development/python"],
    )

    kernel.select_toolboxes(session_id, ["alpha", "beta"], mode="use")
    assert kernel.catalog(session_id=session_id, namespace="development") == []
    kernel.select_toolboxes(session_id, ["beta", "alpha"], mode="use")
    assert [
        item["name"]
        for item in kernel.catalog(session_id=session_id, namespace="development")
    ] == ["same"]


def test_core_namespace_is_discovery_metadata_not_capability_identity(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)

    kernel.call(
        "write_tool", {"name": "help", "namespaces": ["platform/documentation"]}
    )

    assert kernel.view_tool("help")["namespaces"] == ["platform/documentation"]
    assert "#" in kernel.call("help", {"topic": "quickstart"})
    assert "help" not in {item["name"] for item in kernel.catalog(include_core=False)}


def test_active_feedback_surface_is_absent(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    write(kernel, "observed", "async def main(input, ctx):\n    return True\n")
    call_tool(kernel, "observed", {})

    search_schema = CORE_SCHEMAS["search_tools"]
    assert search_schema["required"] == ["query"]
    assert "feedback" not in search_schema["properties"]
    assert not hasattr(kernel_module._KernelCapability, "submit_tool_feedback")
    assert "feedback" not in next(
        item for item in kernel.catalog() if item["name"] == "observed"
    )
    assert "feedback" not in kernel.view_tool("observed")
    assert "feedback" not in kernel.tool_history("observed")["versions"][0]


def test_schema_two_database_migrates_without_feedback_storage(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    with closing(kernel._connect()) as connection, connection:
        connection.execute("PRAGMA user_version = 2")

    reopened = kernel_at(tmp_path)
    with closing(reopened._connect()) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'tool_feedback'"
        ).fetchone()
    assert version == 7
    assert table is None


def test_schema_five_database_removes_feedback_storage(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    with closing(kernel._connect()) as connection, connection:
        connection.execute(
            """
            CREATE TABLE tool_feedback (
                id INTEGER PRIMARY KEY,
                tool_version_id INTEGER NOT NULL REFERENCES tool_versions(id),
                session_id TEXT NOT NULL REFERENCES sessions(id),
                rating INTEGER NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute("PRAGMA user_version = 5")

    reopened = kernel_at(tmp_path)
    with closing(reopened._connect()) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'tool_feedback'"
        ).fetchone()
    assert version == 7
    assert table is None


def test_schema_six_backfills_all_lineage_namespaces_idempotently(
    tmp_path: Path,
) -> None:
    database = tmp_path / "toolbox.db"
    kernel = Kernel(database, cwd=tmp_path)
    write(kernel, "active_user", "async def main(input, ctx):\n    return True\n")
    write(kernel, "retired_user", "async def main(input, ctx):\n    return False\n")
    retired_version_id = kernel.view_tool("retired_user")["version"]
    kernel.delete_tool("retired_user")
    with closing(kernel._connect()) as connection, connection:
        lineage_ids = {
            str(row["name"]): int(row["id"])
            for row in connection.execute("SELECT id, name FROM tool_lineages")
        }
        connection.execute("DROP TABLE tool_namespaces")
        connection.execute("PRAGMA user_version = 6")

    reopened = Kernel(database, cwd=tmp_path)
    with closing(reopened._connect()) as connection:
        rows = connection.execute(
            "SELECT l.name, n.path FROM tool_lineages AS l "
            "JOIN tool_namespaces AS n ON n.lineage_id = l.id "
            "ORDER BY l.name, n.path"
        ).fetchall()
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
    paths = {str(row["name"]): str(row["path"]) for row in rows}
    assert all(paths[name] == "core" for name in CORE_NAMES)
    assert paths["active_user"] == "uncategorized"
    assert paths["retired_user"] == "uncategorized"
    assert reopened.tool_history("retired_user")["namespaces"] == ["uncategorized"]
    assert reopened.tool_history("retired_user")["versions"][0]["version"] == (
        retired_version_id
    )
    assert foreign_key_errors == []

    second_reopen = Kernel(database, cwd=tmp_path)
    with closing(second_reopen._connect()) as connection:
        counts = connection.execute(
            "SELECT lineage_id, COUNT(*) AS count FROM tool_namespaces "
            "GROUP BY lineage_id"
        ).fetchall()
    assert {int(row["lineage_id"]) for row in counts} == set(lineage_ids.values())
    assert all(int(row["count"]) == 1 for row in counts)


def test_schema_three_database_migrates_real_pre_lineage_sessions(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    root = kernel.create_session("legacy-root")
    kernel.append_event(root, "legacy_event", {"kept": True})
    selected = kernel.active_toolboxes(root)
    with closing(kernel._connect()) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "CREATE TABLE sessions_v3 ("
            "id TEXT PRIMARY KEY, cwd TEXT, created_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO sessions_v3 (id, cwd, created_at) "
            "SELECT id, cwd, created_at FROM sessions"
        )
        connection.execute("DROP TABLE sessions")
        connection.execute("ALTER TABLE sessions_v3 RENAME TO sessions")
        connection.execute("PRAGMA user_version = 3")
        connection.commit()

    reopened = kernel_at(tmp_path)
    metadata = reopened.session_metadata(root)
    assert metadata["kind"] == "generic"
    assert metadata["parent_session_id"] is None
    assert metadata["root_session_id"] == root
    assert reopened.read_session(root, limit=100)["events"][0]["payload"] == {
        "kept": True
    }
    assert reopened.active_toolboxes(root) == selected
    with closing(reopened._connect()) as connection:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(sessions)")
        }
        index = connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'index' AND name = 'sessions_parent_created'"
        ).fetchone()
        trigger = connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'trigger' AND name = 'sessions_lineage_immutable'"
        ).fetchone()
    assert {"parent_session_id", "kind", "origin_call_id"} <= columns
    assert index is not None
    assert trigger is not None
    with (
        closing(reopened._connect()) as connection,
        pytest.raises(sqlite3.IntegrityError, match="session lineage is immutable"),
    ):
        connection.execute(
            "UPDATE sessions SET kind = 'conversation' WHERE id = ?", (root,)
        )

    reopened_again = kernel_at(tmp_path)
    assert reopened_again.session_metadata(root) == metadata


def test_schema_four_database_renames_view_core_slot(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    replacement_source = "async def main(input, ctx):\n    return {'custom': True}\n"
    write(
        kernel,
        "view_tool",
        replacement_source,
        input_schema=CORE_SCHEMAS["view_tool"],
        base_version=1,
    )
    with closing(kernel._connect()) as connection, connection:
        toolbox_id = connection.execute("SELECT id FROM toolboxes").fetchone()["id"]
        connection.execute(
            "UPDATE tool_lineages SET name = 'read_tool_source' "
            "WHERE toolbox_id = ? AND name = 'view_tool'",
            (toolbox_id,),
        )
        connection.execute(
            "UPDATE bindings SET name = 'read_tool_source' "
            "WHERE toolbox_id = ? AND name = 'view_tool'",
            (toolbox_id,),
        )
        connection.execute("PRAGMA user_version = 4")

    reopened = kernel_at(tmp_path)

    assert "read_tool_source" not in {item["name"] for item in reopened.bindings()}
    current = reopened.view_tool("view_tool")
    assert current["version"] == 2
    assert current["source"] == replacement_source


def test_versions_exact_calls_and_stale_base(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    version_one = "async def main(input, ctx):\n    return 1\n"
    version_two = "async def main(input, ctx):\n    return 2\n"

    first = write(kernel, "number", version_one)
    assert first["active"] is True
    assert call_tool(kernel, "number", {}) == 1
    assert call_tool(kernel, "number", {}, version=1) == 1

    second = write(kernel, "number", version_two, base_version=1)
    assert second["version"] == 2
    assert call_tool(kernel, "number", {}) == 2
    assert call_tool(kernel, "number", {}, version=1) == 1
    with pytest.raises(ToolboxError) as error:
        write(kernel, "number", version_one, base_version=1)
    assert error.value.code == "stale_base_version"
    assert call_tool(kernel, "number", {}) == 2


def test_tool_management_history_usage_provenance_and_deletion(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    creator_one = kernel.create_session()
    creator_two = kernel.create_session()
    caller = kernel.create_session()
    write(
        kernel,
        "number",
        "async def main(input, ctx):\n    return 1\n",
        session_id=creator_one,
    )
    call_tool(kernel, "number", {}, session_id=caller)
    write(
        kernel,
        "number",
        "async def main(input, ctx):\n    return 2\n",
        session_id=creator_two,
        base_version=1,
    )
    call_tool(kernel, "number", {}, session_id=caller)
    call_tool(kernel, "number", {}, session_id=creator_two)

    inventory = next(
        item for item in kernel.tool_inventory() if item["name"] == "number"
    )
    assert inventory["active_version"] == 2
    assert inventory["version_count"] == 2
    assert inventory["call_count"] == 3
    assert inventory["session_count"] == 2

    history = kernel.tool_history("number")
    assert history["active_version"] == 2
    assert history["call_count"] == 3
    assert history["success_count"] == 3
    assert history["failure_count"] == 0
    assert [version["version"] for version in history["versions"]] == [2, 1]
    assert history["versions"][0]["created_session_id"] == creator_two
    assert history["versions"][0]["call_count"] == 2
    assert history["versions"][1]["created_session_id"] == creator_one
    assert history["versions"][1]["call_count"] == 1
    assert (
        history["versions"][0]["tool_version_id"]
        != history["versions"][1]["tool_version_id"]
    )
    assert history["versions"][0]["timed_call_count"] == 2
    assert history["versions"][1]["timed_call_count"] == 1
    assert history["versions"][0]["average_duration_ms"] >= 0
    assert history["versions"][1]["average_duration_ms"] >= 0
    assert history["sessions"][0]["session_id"] in {caller, creator_two}

    deleted = kernel.delete_tool("number", session_id=creator_two)
    assert deleted == {"name": "number", "deleted_version": 2, "active": False}
    assert "number" not in {item["name"] for item in kernel.bindings()}
    assert (
        next(item for item in kernel.tool_inventory() if item["name"] == "number")[
            "active_version"
        ]
        is None
    )
    assert kernel.tool_history("number")["versions"][0]["source"].endswith("return 2\n")
    assert (
        kernel.read_session(creator_two, limit=100)["events"][-1]["kind"]
        == "binding_deleted"
    )

    with pytest.raises(ToolboxError) as core_error:
        kernel.delete_tool("help")
    assert core_error.value.code == "core_tool_required"


def test_terminal_timings_are_persisted_and_aggregated_by_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel = kernel_at(tmp_path)
    session_id = kernel.create_session()
    write(
        kernel,
        "timed",
        "async def main(input, ctx):\n    return 'v1'\n",
    )

    with monkeypatch.context() as patch:
        ticks = iter((1_000_000_000, 1_002_000_000))
        patch.setattr(kernel_module.time, "perf_counter_ns", lambda: next(ticks))
        assert kernel.call("timed", {}, session_id=session_id, version=1) == "v1"

    write(
        kernel,
        "timed",
        "async def main(input, ctx):\n"
        "    if input.get('fail'):\n"
        "        raise RuntimeError('broken')\n"
        "    return 'v2'\n",
        base_version=1,
    )
    with monkeypatch.context() as patch:
        ticks = iter(
            (
                2_000_000_000,
                2_004_000_000,
                3_000_000_000,
                3_008_000_000,
            )
        )
        patch.setattr(kernel_module.time, "perf_counter_ns", lambda: next(ticks))
        assert kernel.call("timed", {}, session_id=session_id) == "v2"
        with pytest.raises(RuntimeError, match="broken"):
            kernel.call("timed", {"fail": True}, session_id=session_id)

    history = kernel.tool_history("timed")
    version_two, version_one = history["versions"]
    assert version_one["timed_call_count"] == 1
    assert version_one["average_duration_ms"] == 2.0
    assert version_two["timed_call_count"] == 2
    assert version_two["average_duration_ms"] == 6.0
    assert version_one["tool_version_id"] != version_two["tool_version_id"]

    terminal_events = [
        event
        for event in kernel.read_session(session_id, limit=100)["events"]
        if event["kind"] in {"call_succeeded", "call_failed"}
        and event["tool_name"] == "timed"
    ]
    assert [event["payload"]["duration_ms"] for event in terminal_events] == [
        2.0,
        4.0,
        8.0,
    ]
    assert [event["tool_version_id"] for event in terminal_events] == [
        version_one["tool_version_id"],
        version_two["tool_version_id"],
        version_two["tool_version_id"],
    ]

    for invalid in (None, True, "3", -1):
        payload = {"result": None}
        if invalid is not None:
            payload["duration_ms"] = invalid
        kernel.append_event(
            session_id,
            "call_succeeded",
            payload,
            tool_name="timed",
            tool_version=1,
            tool_version_id=version_one["tool_version_id"],
        )

    reopened = kernel_at(tmp_path).tool_history("timed")
    reopened_v1 = next(item for item in reopened["versions"] if item["version"] == 1)
    assert reopened_v1["tool_version_id"] == version_one["tool_version_id"]
    assert reopened_v1["timed_call_count"] == 1
    assert reopened_v1["average_duration_ms"] == 2.0


def test_result_validation_failures_record_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel = kernel_at(tmp_path)
    session_id = kernel.create_session()
    write(kernel, "not_json", "async def main(input, ctx):\n    return {1}\n")

    with monkeypatch.context() as patch:
        ticks = iter((5_000_000_000, 5_003_000_000))
        patch.setattr(kernel_module.time, "perf_counter_ns", lambda: next(ticks))
        with pytest.raises(ToolboxError) as error:
            kernel.call("not_json", {}, session_id=session_id)

    assert error.value.code == "not_json"
    terminal = kernel.read_session(session_id, limit=100)["events"][-1]
    assert terminal["kind"] == "call_failed"
    assert terminal["payload"]["duration_ms"] == 3.0


def test_tools_compose_and_read_the_live_session(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    session_id = kernel.create_session()
    session_help = kernel.call("help", {"topic": "sessions"})
    assert "ctx.caller_session_id" in session_help
    write(
        kernel,
        "add",
        "async def main(input, ctx):\n    return input['a'] + input['b']\n",
    )
    write(
        kernel,
        "double",
        "async def main(input, ctx):\n"
        "    return await ctx.call_tool('add', {'a': input['x'], 'b': input['x']})\n",
    )
    write(
        kernel,
        "recall",
        "async def main(input, ctx):\n"
        "    events = ctx.sessions.current(limit=100)['events']\n"
        "    return {\n"
        "        'session_id': ctx.caller_session_id,\n"
        "        'session_access_id': ctx.sessions.id,\n"
        "        'saw_start': any(e['kind'] == 'call_started' and "
        "e['tool_name'] == 'recall' for e in events),\n"
        "    }\n",
    )

    assert call_tool(kernel, "double", {"x": 9}, session_id=session_id) == 18
    recall = call_tool(kernel, "recall", {}, session_id=session_id)
    assert recall == {
        "session_id": session_id,
        "session_access_id": session_id,
        "saw_start": True,
    }

    events = kernel.read_session(session_id, limit=100)["events"]
    add_starts = [
        event
        for event in events
        if event["kind"] == "call_started" and event["tool_name"] == "add"
    ]
    assert len(add_starts) == 1
    assert add_starts[0]["parent_call_id"] is not None


def test_model_provider_propagates_through_nested_tool_calls(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "ask",
        "async def main(input, ctx):\n"
        "    return await ctx.model_provider.complete(input['messages'])\n",
    )
    write(
        kernel,
        "ask_many",
        "async def main(input, ctx):\n"
        "    return [\n"
        "        await ctx.call_tool('ask', {'messages': input['messages']})\n"
        "        for _ in range(input['count'])\n"
        "    ]\n",
    )

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Any) -> str:
            self.calls += 1
            return f"answer {self.calls}: {messages[-1]['content']}"

    provider = Provider()
    envelope = {
        "name": "ask_many",
        "args": {
            "messages": [{"role": "user", "content": "hello"}],
            "count": 8,
        },
    }
    assert kernel.call("call_tool", envelope, model_provider=provider) == [
        f"answer {index}: hello" for index in range(1, 9)
    ]

    envelope["args"]["count"] = 9
    with pytest.raises(ToolboxError) as exhausted:
        kernel.call("call_tool", envelope, model_provider=provider)
    assert exhausted.value.code == "model_provider_limit"
    assert provider.calls == 16

    single = {
        "name": "ask",
        "args": {"messages": [{"role": "user", "content": "fresh"}]},
    }
    assert kernel.call("call_tool", single, model_provider=provider).endswith("fresh")
    assert provider.calls == 17


def test_child_sessions_inherit_frozen_scope_and_expose_lineage(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    kernel.create_toolbox("alpha")
    kernel.create_toolbox("beta")
    root = kernel.create_session(kind="conversation")
    kernel.select_toolboxes(root, ["alpha", "beta"], mode="use")
    frozen = kernel.snapshot_scope(root)
    kernel.select_toolboxes(root, ["beta"], mode="use")

    child = kernel.create_child_session(frozen, kind="completion")

    assert [item["name"] for item in kernel.active_toolboxes(child)] == [
        "alpha",
        "beta",
    ]
    metadata = kernel.session_metadata(child)
    assert metadata["parent_session_id"] == root
    assert metadata["root_session_id"] == root
    assert metadata["kind"] == "completion"
    assert kernel.read_session(child)["session"] == metadata
    listed = {item["id"]: item for item in kernel.list_sessions(limit=100)["sessions"]}
    assert listed[child]["parent_session_id"] == root
    assert listed[child]["root_session_id"] == root

    with (
        closing(kernel._connect()) as connection,
        pytest.raises(sqlite3.IntegrityError, match="session lineage is immutable"),
    ):
        connection.execute(
            "UPDATE sessions SET parent_session_id = NULL WHERE id = ?", (child,)
        )


def test_nested_tool_completion_records_inner_origin_and_child_log(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "ask",
        "async def main(input, ctx):\n"
        "    return await ctx.model_provider.complete([\n"
        "        {'role': 'user', 'content': input['prompt']},\n"
        "    ])\n",
    )
    write(
        kernel,
        "delegate",
        "async def main(input, ctx):\n"
        "    return await ctx.call_tool('ask', {'prompt': input['prompt']})\n",
    )
    root = kernel.create_session(kind="conversation")
    provider = ProviderUsingModel()

    assert (
        kernel.call(
            "call_tool",
            {"name": "delegate", "args": {"prompt": "nested"}},
            session_id=root,
            model_provider=provider,
        )
        == "nested completion"
    )

    root_events = kernel.read_session(root, limit=100)["events"]
    ask_call = next(
        event
        for event in root_events
        if event["kind"] == "call_started" and event["tool_name"] == "ask"
    )
    children = [
        session
        for session in kernel.list_sessions(limit=100)["sessions"]
        if session["parent_session_id"] == root
    ]
    assert len(children) == 1
    child = children[0]
    assert child["origin_call_id"] == ask_call["call_id"]
    assert child["kind"] == "completion"
    assert [
        event["kind"] for event in kernel.read_session(child["id"], limit=100)["events"]
    ] == ["model_input", "model", "final"]


def test_provider_sessions_form_recursive_parent_chain(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    provider = ModelProvider(kernel, ProviderUsingModel())
    root = provider.start_session()
    child = provider.start_session(parent_scope=kernel.snapshot_scope(root.session_id))

    assert (
        child.completion_provider().complete(
            [{"role": "user", "content": "grandchild"}]
        )
        == "nested completion"
    )

    sessions = kernel.list_sessions(limit=100)["sessions"]
    grandchild = next(
        session
        for session in sessions
        if session["parent_session_id"] == child.session_id
    )
    assert kernel.session_metadata(child.session_id)["parent_session_id"] == (
        root.session_id
    )
    assert grandchild["root_session_id"] == root.session_id
    assert grandchild["kind"] == "completion"


def test_direct_model_session_traffic_is_durably_logged(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    provider = ModelProvider(kernel, ProviderUsingModel())
    session = provider.start_session()

    turn = session.respond([{"role": "user", "content": "direct"}], [])

    assert turn.calls
    events = kernel.read_session(session.session_id, limit=100)["events"]
    assert [event["kind"] for event in events] == ["model_input", "model"]
    assert events[0]["payload"]["messages"] == [{"role": "user", "content": "direct"}]


def test_isolated_tools_can_run_recursive_first_class_agents(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "delegate",
        "async def main(input, ctx):\n"
        "    return await ctx.model_provider.run_agent('child')\n",
    )
    write(
        kernel,
        "child_delegate",
        "async def main(input, ctx):\n"
        "    return {\n"
        "        'session_id': ctx.sessions.id,\n"
        "        'parent_id': ctx.sessions.metadata()['parent_session_id'],\n"
        "        'answer': await ctx.model_provider.run_agent('grandchild'),\n"
        "    }\n",
    )

    class RecursiveAgentTransport:
        def __init__(self) -> None:
            self.child_observation: dict[str, Any] | None = None

        def respond(self, messages: Any, tools: Any) -> ModelTurn:
            prompt = next(
                message["content"]
                for message in reversed(messages)
                if message["role"] == "user"
            )
            observations = [
                message for message in messages if message["role"] == "tool"
            ]
            if prompt == "root" and not observations:
                return ModelTurn(
                    calls=(
                        ToolCall("call_tool", {"name": "delegate", "args": {}}, "d"),
                    )
                )
            if prompt == "child" and not observations:
                return ModelTurn(
                    calls=(
                        ToolCall(
                            "call_tool",
                            {"name": "child_delegate", "args": {}},
                            "cd",
                        ),
                    )
                )
            if prompt == "grandchild":
                return ModelTurn(text="grandchild answer")
            if prompt == "child":
                self.child_observation = observations[-1]["content"]
                return ModelTurn(text="child answer")
            return ModelTurn(text="root answer")

        def complete(self, messages: Any) -> str:
            return "unused"

        def cancel_current(self) -> None:
            pass

        def reset_cancellation(self) -> None:
            pass

    transport = RecursiveAgentTransport()
    provider = ModelProvider(kernel, transport)
    result = Harness(kernel).run(provider, "root")

    sessions = kernel.list_sessions(limit=100)["sessions"]
    root = kernel.session_metadata(result.session_id)
    child = next(
        session
        for session in sessions
        if session["parent_session_id"] == root["id"]
        and session["kind"] == "conversation"
    )
    grandchild = next(
        session
        for session in sessions
        if session["parent_session_id"] == child["id"]
        and session["kind"] == "conversation"
    )
    assert result.answer == "root answer"
    assert child["root_session_id"] == root["id"]
    assert grandchild["root_session_id"] == root["id"]
    assert transport.child_observation == {
        "session_id": child["id"],
        "parent_id": root["id"],
        "answer": "grandchild answer",
    }
    child_calls = kernel.read_session(child["id"], limit=100)["events"]
    child_call = next(
        event
        for event in child_calls
        if event["kind"] == "call_started" and event["tool_name"] == "child_delegate"
    )
    assert grandchild["origin_call_id"] == child_call["call_id"]
    assert [
        event["kind"]
        for event in kernel.read_session(grandchild["id"], limit=100)["events"]
    ] == ["user", "model", "final"]


class UniformAgentTransport:
    """Session-bound transport that exercises direct recursive agent actions."""

    def __init__(self) -> None:
        self.action_names: dict[str, tuple[str, ...]] = {}
        self.bound_keys: list[str] = []
        self.cancelled = threading.Event()

    def for_session(self, session_id: str) -> UniformBoundAgentTransport:
        self.bound_keys.append(session_id)
        return UniformBoundAgentTransport(self, session_id)

    def respond(self, messages: Any, tools: Any) -> ModelTurn:
        raise AssertionError("unbound agent transport received model traffic")

    def complete(self, messages: Any) -> str:
        raise AssertionError("unbound agent transport received completion traffic")

    def cancel_current(self) -> None:
        raise AssertionError("unbound agent transport was cancelled")

    def reset_cancellation(self) -> None:
        raise AssertionError("unbound agent transport was reset")


class UniformBoundAgentTransport:
    def __init__(self, parent: UniformAgentTransport, session_id: str) -> None:
        self.parent = parent
        self.session_id = session_id

    def respond(self, messages: Any, tools: Any) -> ModelTurn:
        self.parent.action_names[self.session_id] = tuple(
            tool["name"] for tool in tools
        )
        prompt = next(
            message["content"]
            for message in reversed(messages)
            if message["role"] == "user"
        )
        observations = [message for message in messages if message["role"] == "tool"]
        if prompt == "root" and not observations:
            return ModelTurn(
                calls=(ToolCall("run_agent", {"prompt": "child"}, "agent"),)
            )
        if prompt == "child" and not observations:
            return ModelTurn(calls=(ToolCall("help", {"topic": "toc"}, "help"),))
        if prompt == "child":
            assert str(observations[-1]["content"]).startswith("# Mechagnome")
            return ModelTurn(text="child answer")
        assert observations[-1]["content"] == "child answer"
        return ModelTurn(text="root answer")

    def complete(self, messages: Any) -> str:
        return "completion"

    def cancel_current(self) -> None:
        self.parent.cancelled.set()

    def reset_cancellation(self) -> None:
        self.parent.cancelled.clear()


def test_direct_agents_share_one_conversation_action_surface_and_lineage(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    transport = UniformAgentTransport()
    provider = ModelProvider(kernel, transport)

    result = Harness(kernel).run(provider, "root")

    sessions = kernel.list_sessions(limit=100)["sessions"]
    child = next(
        session
        for session in sessions
        if session["parent_session_id"] == result.session_id
        and session["kind"] == "conversation"
    )
    assert result.answer == "root answer"
    assert child["origin_call_id"] is None
    assert child["root_session_id"] == result.session_id
    assert transport.action_names == {
        result.session_id: MODEL_ACTION_NAMES,
        child["id"]: MODEL_ACTION_NAMES,
    }
    assert provider._session_transports == {}


class RecursiveDetachedAgentTransport(UniformAgentTransport):
    """Have a foreground child detach a grandchild through model actions."""

    def __init__(self) -> None:
        super().__init__()
        self.job_id: str | None = None

    def for_session(self, session_id: str) -> RecursiveDetachedBoundTransport:
        self.bound_keys.append(session_id)
        return RecursiveDetachedBoundTransport(self, session_id)


class RecursiveDetachedBoundTransport(UniformBoundAgentTransport):
    parent: RecursiveDetachedAgentTransport

    def respond(self, messages: Any, tools: Any) -> ModelTurn:
        self.parent.action_names[self.session_id] = tuple(
            tool["name"] for tool in tools
        )
        prompt = next(
            message["content"]
            for message in reversed(messages)
            if message["role"] == "user"
        )
        observations = [message for message in messages if message["role"] == "tool"]
        if prompt == "root" and messages[-1]["role"] == "user":
            return ModelTurn(
                calls=(ToolCall("run_agent", {"prompt": "child"}, "child"),)
            )
        if prompt == "root":
            assert observations[-1]["content"] == "child detached a grandchild"
            return ModelTurn(text="root continued")
        if prompt == "child" and messages[-1]["role"] == "user":
            return ModelTurn(
                calls=(
                    ToolCall(
                        "run_agent",
                        {"prompt": "grandchild", "detach": True},
                        "grandchild",
                    ),
                )
            )
        if prompt == "child":
            handle = observations[-1]["content"]
            self.parent.job_id = handle["job_id"]
            return ModelTurn(text="child detached a grandchild")
        if prompt == "grandchild":
            return ModelTurn(text="grandchild answer")
        if prompt == "inspect descendant" and messages[-1]["role"] == "user":
            assert self.parent.job_id is not None
            return ModelTurn(
                calls=(
                    ToolCall(
                        "run_agent",
                        {"job_id": self.parent.job_id},
                        "inspect",
                    ),
                )
            )
        snapshot = observations[-1]["content"]
        assert snapshot["status"] == "succeeded"
        assert snapshot["result"] == "grandchild answer"
        return ModelTurn(text="ancestor inspected descendant")


def test_recursive_agent_can_detach_and_ancestor_can_inspect(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    transport = RecursiveDetachedAgentTransport()
    harness = Harness(kernel)
    conversation = harness.start(ModelProvider(kernel, transport))

    assert conversation.send("root").answer == "root continued"
    assert transport.job_id is not None
    wait_for_agent(
        harness,
        transport.job_id,
        conversation.session_id,
        status="succeeded",
    )
    assert (
        conversation.send("inspect descendant").answer
        == "ancestor inspected descendant"
    )

    sessions = kernel.list_sessions(limit=100)["sessions"]
    child = next(
        session
        for session in sessions
        if session["parent_session_id"] == conversation.session_id
    )
    assert kernel.session_metadata(transport.job_id)["parent_session_id"] == child["id"]
    assert transport.action_names == {
        conversation.session_id: MODEL_ACTION_NAMES,
        child["id"]: MODEL_ACTION_NAMES,
        transport.job_id: MODEL_ACTION_NAMES,
    }
    harness.close()


@pytest.mark.parametrize(
    "args",
    (
        {},
        {"prompt": ""},
        {"prompt": "work", "detach": 1},
        {"prompt": "work", "extra": True},
        {"job_id": ""},
        {"job_id": "job", "prompt": "mixed"},
        {"prompt": "x" * (256 * 1024 + 1)},
    ),
)
def test_run_agent_request_validation_is_strict(args: dict[str, Any]) -> None:
    with pytest.raises(ToolboxError) as error:
        _parse_run_agent_request(args)
    assert error.value.code == "invalid_run_agent_request"


def test_raw_transport_cannot_implicitly_authorize_direct_agent_runs(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)

    class UnauthorizedTransport:
        def respond(self, messages: Any, tools: Any) -> ModelTurn:
            observations = [
                message for message in messages if message["role"] == "tool"
            ]
            assert tuple(tool["name"] for tool in tools) == MODEL_ACTION_NAMES
            if not observations:
                return ModelTurn(
                    calls=(ToolCall("run_agent", {"prompt": "child"}, "agent"),)
                )
            assert observations[-1]["content"]["error"]["code"] == (
                "model_provider_unavailable"
            )
            return ModelTurn(text="not authorized")

    result = Harness(kernel).run(UnauthorizedTransport(), "root")

    assert result.answer == "not authorized"
    assert all(
        session["parent_session_id"] is None
        for session in kernel.list_sessions(limit=100)["sessions"]
    )


class DetachedAgentTransport(UniformAgentTransport):
    """Hold one detached child while its root conversation continues."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.job_id: str | None = None

    def for_session(self, session_id: str) -> DetachedBoundAgentTransport:
        self.bound_keys.append(session_id)
        return DetachedBoundAgentTransport(self, session_id)


class DetachedBoundAgentTransport(UniformBoundAgentTransport):
    parent: DetachedAgentTransport

    def respond(self, messages: Any, tools: Any) -> ModelTurn:
        self.parent.action_names[self.session_id] = tuple(
            tool["name"] for tool in tools
        )
        prompt = next(
            message["content"]
            for message in reversed(messages)
            if message["role"] == "user"
        )
        observations = [message for message in messages if message["role"] == "tool"]
        if prompt == "root" and not observations:
            return ModelTurn(
                calls=(
                    ToolCall(
                        "run_agent",
                        {"prompt": "detached child", "detach": True},
                        "detach-agent",
                    ),
                )
            )
        if prompt == "root":
            handle = observations[-1]["content"]
            self.parent.job_id = handle["job_id"]
            assert handle["status"] == "running"
            return ModelTurn(text="root continued")
        if prompt == "inspect":
            assert self.parent.job_id is not None
            if messages[-1]["role"] == "user":
                return ModelTurn(
                    calls=(
                        ToolCall(
                            "run_agent",
                            {"job_id": self.parent.job_id},
                            "inspect-agent",
                        ),
                    )
                )
            assert observations[-1]["content"]["status"] == "succeeded"
            assert observations[-1]["content"]["result"] == "detached answer"
            return ModelTurn(text="inspection complete")
        self.parent.started.set()
        assert self.parent.release.wait(timeout=5)
        return ModelTurn(text="detached answer")


def test_detached_agent_continues_and_is_inspectable_by_parent(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    transport = DetachedAgentTransport()
    harness = Harness(kernel)
    conversation = harness.start(ModelProvider(kernel, transport))

    first = conversation.send("root")

    assert first.answer == "root continued"
    assert transport.started.wait(timeout=2)
    assert transport.job_id is not None
    running = harness._agent_coordinator.inspect(
        transport.job_id, session_id=conversation.session_id
    )
    assert running == {"job_id": transport.job_id, "status": "running"}
    child = kernel.session_metadata(transport.job_id)
    assert child["kind"] == "conversation"
    assert child["parent_session_id"] == conversation.session_id
    assert transport.action_names[transport.job_id] == MODEL_ACTION_NAMES

    transport.release.set()
    completed = wait_for_agent(
        harness,
        transport.job_id,
        conversation.session_id,
        status="succeeded",
    )
    assert completed["result"] == "detached answer"
    assert conversation.send("inspect").answer == "inspection complete"
    harness.close()


def test_detached_agent_handle_is_visible_to_ancestors_but_not_siblings(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    transport = DetachedAgentTransport()
    provider = ModelProvider(kernel, transport)
    harness = Harness(kernel)
    root = harness.start(provider)
    child = harness._agent_coordinator._create_child(root)
    sibling_session = provider.start_session(
        parent_scope=kernel.snapshot_scope(root.session_id)
    )

    handle = harness._agent_coordinator.start_detached(
        child, "detached descendant", sink=None
    )
    assert transport.started.wait(timeout=2)

    assert (
        harness._agent_coordinator.inspect(
            handle["job_id"], session_id=root.session_id
        )["status"]
        == "running"
    )
    with pytest.raises(ToolboxError) as hidden:
        harness._agent_coordinator.inspect(
            handle["job_id"], session_id=sibling_session.session_id
        )
    assert hidden.value.code == "unknown_detached_agent"

    transport.release.set()
    wait_for_agent(
        harness,
        handle["job_id"],
        root.session_id,
        status="succeeded",
    )
    child.close()
    sibling_session.close()
    harness.close()


class FailingAgentTransport(UniformAgentTransport):
    """Raise provider-controlled failures inside foreground and detached children."""

    def __init__(self) -> None:
        super().__init__()
        self.detached_job_id: str | None = None
        self.foreground_error: dict[str, Any] | None = None

    def for_session(self, session_id: str) -> FailingBoundAgentTransport:
        self.bound_keys.append(session_id)
        return FailingBoundAgentTransport(self, session_id)


class FailingBoundAgentTransport(UniformBoundAgentTransport):
    parent: FailingAgentTransport

    def respond(self, messages: Any, tools: Any) -> ModelTurn:
        prompt = next(
            message["content"]
            for message in reversed(messages)
            if message["role"] == "user"
        )
        observations = [message for message in messages if message["role"] == "tool"]
        if prompt == "foreground failure" and messages[-1]["role"] == "user":
            return ModelTurn(
                calls=(ToolCall("run_agent", {"prompt": "fail foreground"}, "f"),)
            )
        if prompt == "foreground failure":
            self.parent.foreground_error = observations[-1]["content"]["error"]
            return ModelTurn(text="foreground failure handled")
        if prompt == "detached failure" and messages[-1]["role"] == "user":
            return ModelTurn(
                calls=(
                    ToolCall(
                        "run_agent",
                        {"prompt": "fail detached", "detach": True},
                        "d",
                    ),
                )
            )
        if prompt == "detached failure":
            self.parent.detached_job_id = observations[-1]["content"]["job_id"]
            return ModelTurn(text="detached failure started")
        if prompt == "fail foreground":
            raise RuntimeError("FOREGROUND_SENTINEL_SECRET")
        raise ModelTransportError(
            "provider_internal",
            "DETACHED_SENTINEL_SECRET",
        )


def test_agent_failures_are_sanitized_for_foreground_and_detached_results(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    transport = FailingAgentTransport()
    harness = Harness(kernel)
    conversation = harness.start(ModelProvider(kernel, transport))

    assert conversation.send("foreground failure").answer == (
        "foreground failure handled"
    )
    assert transport.foreground_error == {
        "code": "model_provider_failed",
        "message": "model provider request failed",
        "details": {},
    }
    assert conversation.send("detached failure").answer == "detached failure started"
    assert transport.detached_job_id is not None
    failed = wait_for_agent(
        harness,
        transport.detached_job_id,
        conversation.session_id,
        status="failed",
    )
    assert failed["error"] == {
        "code": "model_provider_failed",
        "message": "model provider request failed",
        "details": {},
    }
    assert "SENTINEL_SECRET" not in json.dumps(failed)
    harness.close()


class CancellationDomainTransport:
    """Block selected conversations and record session-local cancellation."""

    def __init__(self) -> None:
        self.events: dict[str, threading.Event] = {}
        self.cancelled_keys: list[str] = []
        self.detached_started = threading.Event()
        self.foreground_started = threading.Event()
        self.root_blocked = threading.Event()
        self.release_detached = threading.Event()
        self.detached_id: str | None = None

    def for_session(self, session_id: str) -> CancellationBoundTransport:
        self.events.setdefault(session_id, threading.Event())
        return CancellationBoundTransport(self, session_id)

    def respond(self, messages: Any, tools: Any) -> ModelTurn:
        raise AssertionError("unbound cancellation transport received traffic")

    def complete(self, messages: Any) -> str:
        raise AssertionError("unbound cancellation transport received completion")

    def cancel_current(self) -> None:
        raise AssertionError("unbound cancellation transport was cancelled")

    def reset_cancellation(self) -> None:
        raise AssertionError("unbound cancellation transport was reset")


class CancellationBoundTransport:
    def __init__(self, parent: CancellationDomainTransport, session_id: str) -> None:
        self.parent = parent
        self.session_id = session_id

    def respond(self, messages: Any, tools: Any) -> ModelTurn:
        prompt = next(
            message["content"]
            for message in reversed(messages)
            if message["role"] == "user"
        )
        observations = [message for message in messages if message["role"] == "tool"]
        if prompt == "start detached" and messages[-1]["role"] == "user":
            return ModelTurn(
                calls=(
                    ToolCall(
                        "run_agent",
                        {"prompt": "detached", "detach": True},
                        "detach",
                    ),
                )
            )
        if prompt == "start detached":
            self.parent.detached_id = observations[-1]["content"]["job_id"]
            return ModelTurn(text="detached started")
        if prompt == "detached":
            self.parent.detached_started.set()
            while not self.parent.release_detached.wait(timeout=0.01):
                if self.parent.events[self.session_id].is_set():
                    raise ToolboxError("cancelled", "detached cancelled")
            return ModelTurn(text="detached complete")
        if prompt == "block root":
            self.parent.root_blocked.set()
            while not self.parent.events[self.session_id].wait(timeout=0.01):
                pass
            raise ToolboxError("cancelled", "root cancelled")
        if prompt == "start foreground" and messages[-1]["role"] == "user":
            return ModelTurn(
                calls=(
                    ToolCall(
                        "run_agent",
                        {"prompt": "foreground child"},
                        "foreground",
                    ),
                )
            )
        if prompt == "foreground child":
            self.parent.foreground_started.set()
            while not self.parent.events[self.session_id].wait(timeout=0.01):
                pass
            raise ToolboxError("cancelled", "foreground child cancelled")
        return ModelTurn(text="unexpected")

    def complete(self, messages: Any) -> str:
        return "completion"

    def cancel_current(self) -> None:
        self.parent.cancelled_keys.append(self.session_id)
        self.parent.events[self.session_id].set()

    def reset_cancellation(self) -> None:
        self.parent.events[self.session_id].clear()


def test_parent_cancellation_does_not_stop_detached_agent(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    transport = CancellationDomainTransport()
    harness = Harness(kernel)
    conversation = harness.start(ModelProvider(kernel, transport))

    assert conversation.send("start detached").answer == "detached started"
    assert transport.detached_started.wait(timeout=2)
    assert transport.detached_id is not None
    failure: list[Exception] = []

    def block_root() -> None:
        try:
            conversation.send("block root")
        except Exception as error:
            failure.append(error)

    thread = threading.Thread(target=block_root)
    thread.start()
    assert transport.root_blocked.wait(timeout=2)
    assert conversation.cancel() is True
    thread.join(timeout=2)

    assert len(failure) == 1
    assert transport.cancelled_keys == [conversation.session_id]
    assert (
        harness._agent_coordinator.inspect(
            transport.detached_id, session_id=conversation.session_id
        )["status"]
        == "running"
    )

    transport.release_detached.set()
    wait_for_agent(
        harness,
        transport.detached_id,
        conversation.session_id,
        status="succeeded",
    )
    harness.close()


def test_harness_close_stops_detached_agents_idempotently(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    transport = CancellationDomainTransport()
    harness = Harness(kernel)
    conversation = harness.start(ModelProvider(kernel, transport))

    conversation.send("start detached")
    assert transport.detached_started.wait(timeout=2)
    assert transport.detached_id is not None

    harness.close()
    harness.close()

    stopped = harness._agent_coordinator.inspect(
        transport.detached_id, session_id=conversation.session_id
    )
    assert stopped["status"] == "failed"
    assert stopped["error"]["code"] == "detached_agent_shutdown"


def test_detached_agent_limit_is_separate_and_bounded(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    transport = CancellationDomainTransport()
    harness = Harness(kernel)
    conversation = harness.start(ModelProvider(kernel, transport))

    handles = [
        harness._agent_coordinator.start_detached(conversation, "detached", sink=None)
        for _ in range(4)
    ]
    with pytest.raises(ToolboxError) as error:
        harness._agent_coordinator.start_detached(conversation, "detached", sink=None)

    assert len({handle["job_id"] for handle in handles}) == 4
    assert error.value.code == "detached_agent_limit"
    harness.close()


class BoundaryAgentTransport(UniformAgentTransport):
    def for_session(self, session_id: str) -> BoundaryBoundAgentTransport:
        self.bound_keys.append(session_id)
        return BoundaryBoundAgentTransport(self, session_id)


class BoundaryBoundAgentTransport(UniformBoundAgentTransport):
    def respond(self, messages: Any, tools: Any) -> ModelTurn:
        prompt = next(
            message["content"]
            for message in reversed(messages)
            if message["role"] == "user"
        )
        if prompt == "oversized":
            return ModelTurn(text="x" * (1024 * 1024))
        if prompt == "surrogate":
            return ModelTurn(text="\ud800")
        return ModelTurn(text=f"answer:{prompt}")


def test_detached_agent_result_limit_and_unicode_are_terminal(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    harness = Harness(kernel)
    conversation = harness.start(ModelProvider(kernel, BoundaryAgentTransport()))

    oversized = harness._agent_coordinator.start_detached(
        conversation, "oversized", sink=None
    )
    failure = wait_for_agent(
        harness,
        oversized["job_id"],
        conversation.session_id,
        status="failed",
    )
    assert failure["error"]["code"] == "detached_agent_result_too_large"

    surrogate = harness._agent_coordinator.start_detached(
        conversation, "surrogate", sink=None
    )
    success = wait_for_agent(
        harness,
        surrogate["job_id"],
        conversation.session_id,
        status="succeeded",
    )
    assert success["result"] == "\ud800"
    harness.close()


def test_detached_agent_retention_evicts_oldest_terminal_handle(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    harness = Harness(kernel)
    conversation = harness.start(ModelProvider(kernel, BoundaryAgentTransport()))
    handles: list[str] = []

    for index in range(65):
        handle = harness._agent_coordinator.start_detached(
            conversation, f"job-{index}", sink=None
        )
        handles.append(handle["job_id"])
        wait_for_agent(
            harness,
            handle["job_id"],
            conversation.session_id,
            status="succeeded",
        )

    deadline = time.monotonic() + 2
    while True:
        try:
            harness._agent_coordinator.inspect(
                handles[0], session_id=conversation.session_id
            )
        except ToolboxError as error:
            assert error.code == "unknown_detached_agent"
            break
        if time.monotonic() >= deadline:
            raise AssertionError("oldest detached agent handle was not evicted")
        time.sleep(0.01)
    newest = harness._agent_coordinator.inspect(
        handles[-1], session_id=conversation.session_id
    )
    assert newest["result"] == "answer:job-64"
    harness.close()


def test_parent_cancellation_cascades_to_foreground_agent(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    transport = CancellationDomainTransport()
    harness = Harness(kernel)
    conversation = harness.start(ModelProvider(kernel, transport))
    failure: list[Exception] = []

    def run_foreground() -> None:
        try:
            conversation.send("start foreground")
        except Exception as error:
            failure.append(error)

    thread = threading.Thread(target=run_foreground)
    thread.start()
    assert transport.foreground_started.wait(timeout=2)
    assert conversation.cancel() is True
    thread.join(timeout=2)

    child = next(
        session
        for session in kernel.list_sessions(limit=100)["sessions"]
        if session["parent_session_id"] == conversation.session_id
    )
    assert len(failure) == 1
    assert conversation.session_id in transport.cancelled_keys
    assert child["id"] in transport.cancelled_keys
    harness.close()


def test_parent_cancellation_before_child_send_prevents_child_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = kernel_at(tmp_path)
    transport = UniformAgentTransport()
    harness = Harness(kernel)
    conversation = harness.start(ModelProvider(kernel, transport))
    child_send_entered = threading.Event()
    release_child_send = threading.Event()
    original_send = Conversation.send

    def gated_send(
        target: Conversation,
        prompt: str,
        *,
        on_event: Any = None,
        _agent_budget: Any = None,
    ) -> Any:
        if target is not conversation and prompt == "child":
            child_send_entered.set()
            assert release_child_send.wait(timeout=2)
        return original_send(
            target,
            prompt,
            on_event=on_event,
            _agent_budget=_agent_budget,
        )

    monkeypatch.setattr(Conversation, "send", gated_send)
    failures: list[Exception] = []

    def run_parent() -> None:
        try:
            conversation.send("root")
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=run_parent)
    thread.start()
    assert child_send_entered.wait(timeout=2)
    assert conversation.cancel() is True
    release_child_send.set()
    thread.join(timeout=2)

    child = next(
        session
        for session in kernel.list_sessions(limit=100)["sessions"]
        if session["parent_session_id"] == conversation.session_id
    )
    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], ToolboxError)
    assert kernel.read_session(child["id"], limit=100)["events"] == []
    harness.close()


def test_recursive_foreground_agents_share_an_active_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harness_module, "_MAX_ACTIVE_FOREGROUND_AGENTS", 1)
    kernel = kernel_at(tmp_path)
    transport = CancellationDomainTransport()
    harness = Harness(kernel)
    conversation = harness.start(ModelProvider(kernel, transport))
    failures: list[Exception] = []

    def run_first() -> None:
        try:
            harness._agent_coordinator.run_foreground(conversation, "foreground child")
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert transport.foreground_started.wait(timeout=2)

    with pytest.raises(ToolboxError) as error:
        harness._agent_coordinator.run_foreground(conversation, "second child")
    assert error.value.code == "foreground_agent_limit"

    conversation.close()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(failures) == 1
    harness.close()


def test_recursive_agent_launches_share_a_cumulative_rollout_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harness_module, "_MAX_AGENT_LAUNCHES_PER_ROLLOUT", 2)
    kernel = kernel_at(tmp_path)

    class BudgetTransport(UniformAgentTransport):
        def for_session(self, session_id: str) -> BudgetBoundTransport:
            self.bound_keys.append(session_id)
            return BudgetBoundTransport(self, session_id)

    class BudgetBoundTransport(UniformBoundAgentTransport):
        def respond(self, messages: Any, tools: Any) -> ModelTurn:
            prompt = next(
                message["content"]
                for message in reversed(messages)
                if message["role"] == "user"
            )
            if prompt.startswith("child-"):
                return ModelTurn(text=f"answer:{prompt}")
            observations = [
                message for message in messages if message["role"] == "tool"
            ]
            if len(observations) < 3:
                return ModelTurn(
                    calls=(
                        ToolCall(
                            "run_agent",
                            {"prompt": f"child-{len(observations)}"},
                            f"launch-{len(observations)}",
                        ),
                    )
                )
            assert observations[0]["content"] == "answer:child-0"
            assert observations[1]["content"] == "answer:child-1"
            assert observations[2]["content"]["error"]["code"] == ("agent_launch_limit")
            return ModelTurn(text="budget enforced")

    result = Harness(kernel).run(ModelProvider(kernel, BudgetTransport()), "root")

    assert result.answer == "budget enforced"


def test_child_agent_tools_see_child_identity_and_create_grandchildren(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "child_probe",
        "async def main(input, ctx):\n"
        "    return {\n"
        "        'id': ctx.sessions.id,\n"
        "        'caller': ctx.caller_session_id,\n"
        "        'metadata': ctx.sessions.metadata(),\n"
        "        'answer': await ctx.model_provider.complete([\n"
        "            {'role': 'user', 'content': 'nested'},\n"
        "        ]),\n"
        "    }\n",
    )

    class ChildAgentTransport:
        def __init__(self) -> None:
            self.observation: dict[str, Any] | None = None

        def respond(self, messages: Any, tools: Any) -> ModelTurn:
            observations = [item for item in messages if item["role"] == "tool"]
            if not observations:
                return ModelTurn(
                    calls=(
                        ToolCall(
                            "call_tool",
                            {"name": "child_probe", "args": {}},
                            "probe",
                        ),
                    )
                )
            self.observation = observations[-1]["content"]
            return ModelTurn(text="done")

        def complete(self, messages: Any) -> str:
            return "grandchild answer"

        def cancel_current(self) -> None:
            pass

        def reset_cancellation(self) -> None:
            pass

    transport = ChildAgentTransport()
    provider = ModelProvider(kernel, transport)
    root = provider.start_session()
    child = provider.start_session(parent_scope=kernel.snapshot_scope(root.session_id))

    Harness(kernel).start(provider, session_id=child.session_id).send("probe")

    assert transport.observation is not None
    assert transport.observation["id"] == child.session_id
    assert transport.observation["caller"] == child.session_id
    assert transport.observation["metadata"]["parent_session_id"] == root.session_id
    grandchildren = [
        session
        for session in kernel.list_sessions(limit=100)["sessions"]
        if session["parent_session_id"] == child.session_id
        and session["kind"] == "completion"
    ]
    assert len(grandchildren) == 1
    grandchild = grandchildren[0]
    assert grandchild["root_session_id"] == root.session_id
    assert grandchild["origin_call_id"] is not None


def test_model_provider_is_predictably_unavailable_for_direct_calls(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "ask",
        "async def main(input, ctx):\n"
        "    return await ctx.model_provider.complete([\n"
        "        {'role': 'user', 'content': 'hello'},\n"
        "    ])\n",
    )

    with pytest.raises(ToolboxError) as error:
        call_tool(kernel, "ask", {})

    assert error.value.code == "model_provider_unavailable"


def test_all_core_details_are_viewable_and_schema_is_pinned(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)

    for name in CORE_NAMES:
        result = kernel.call("view_tool", {"name": name})
        assert result["source"].startswith('"""')
        assert result["input_schema"] == CORE_SCHEMAS[name]

    with pytest.raises(ToolboxError) as error:
        write(
            kernel,
            "search_tools",
            "async def main(input, ctx):\n    return {}\n",
            input_schema={"type": "object"},
            base_version=1,
        )
    assert error.value.code == "core_schema_pinned"
    assert kernel.bindings()[3]["active_version"] == 1


def test_core_replacement_gets_slot_capability_and_host_rollback(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    catalog_source = (
        "async def main(input, ctx):\n"
        "    return {'names': [item['name'] for item in ctx.kernel.catalog()]}\n"
    )

    write(kernel, "impostor", catalog_source)
    with pytest.raises(ToolboxError) as denied:
        call_tool(kernel, "impostor", {})
    assert denied.value.code == "capability_denied"

    replacement = write(
        kernel,
        "search_tools",
        catalog_source,
        input_schema=CORE_SCHEMAS["search_tools"],
        base_version=1,
    )
    assert replacement["version"] == 2
    assert "impostor" in kernel.call("search_tools", {"query": "ignored"})["names"]

    assert kernel.rollback("search_tools", version=1) == {
        "name": "search_tools",
        "from_version": 2,
        "to_version": 1,
    }
    restored = kernel.call("search_tools", {"query": "impostor"})
    assert restored["items"][0]["name"] == "impostor"


def test_nested_calls_route_through_replaced_call_tool(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "delegator",
        "async def main(input, ctx):\n    return await ctx.call_tool('missing', {})\n",
    )
    replacement_source = (
        "async def main(input, ctx):\n"
        "    return {'intercepted': input['name'], 'args': input['args']}\n"
    )
    write(
        kernel,
        "call_tool",
        replacement_source,
        input_schema=CORE_SCHEMAS["call_tool"],
        base_version=1,
    )

    assert kernel.call("call_tool", {"name": "delegator", "args": {}}) == {
        "intercepted": "delegator",
        "args": {},
    }
    # Exact invocation bypasses the replaced outer binding only for this host test.
    assert kernel.call("delegator", {}) == {"intercepted": "missing", "args": {}}
    kernel.rollback("call_tool", version=1)
    with pytest.raises(ToolboxError, match="unknown tool"):
        call_tool(kernel, "delegator", {})


def test_source_and_sessions_survive_restart(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    session_id = kernel.create_session()
    write(
        kernel,
        "constant",
        "async def main(input, ctx):\n    return {'value': 42}\n",
        session_id=session_id,
    )
    assert call_tool(kernel, "constant", {}, session_id=session_id) == {"value": 42}

    reopened = kernel_at(tmp_path)
    assert call_tool(reopened, "constant", {}) == {"value": 42}
    listed = reopened.list_sessions(limit=100)["sessions"]
    assert session_id in {item["id"] for item in listed}
    assert reopened.read_session(session_id, limit=100)["events"]


def test_database_is_owner_only(tmp_path: Path) -> None:
    database = tmp_path / "private" / "toolbox.db"

    Kernel(database)

    assert stat.S_IMODE(database.stat().st_mode) == 0o600
    assert stat.S_IMODE(database.parent.stat().st_mode) == 0o700


def test_recursive_composition_is_bounded_and_recorded(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path, max_depth=4, max_calls=20)
    session_id = kernel.create_session()
    write(
        kernel,
        "forever",
        "async def main(input, ctx):\n    return await ctx.call_tool('forever', {})\n",
    )

    with pytest.raises(ToolboxError) as error:
        call_tool(kernel, "forever", {}, session_id=session_id)
    assert error.value.code == "max_depth"
    events = kernel.read_session(session_id, limit=100)["events"]
    assert any(event["kind"] == "call_failed" for event in events)


def test_invalid_source_never_moves_the_binding(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    write(kernel, "stable", "async def main(input, ctx):\n    return 'old'\n")

    with pytest.raises(ToolboxError) as error:
        write(
            kernel,
            "stable",
            "async def main(input, ctx)\n    return 'broken'\n",
            base_version=1,
        )
    assert error.value.code == "invalid_source"
    assert call_tool(kernel, "stable", {}) == "old"
    assert next(item for item in kernel.bindings() if item["name"] == "stable")[
        "versions"
    ] == [1]


def test_tool_main_must_be_async_and_supports_async_host_calls(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)

    with pytest.raises(ToolboxError) as error:
        write(kernel, "sync_tool", "def main(input, ctx):\n    return input\n")
    assert error.value.code == "sync_main"

    write(
        kernel,
        "async_tool",
        "import asyncio\n\n"
        "async def main(input, ctx):\n"
        "    await asyncio.sleep(0)\n"
        "    return input['value']\n",
    )

    result = asyncio.run(kernel.call_async("async_tool", {"value": 42}))

    assert result == 42


def test_persisted_sync_tool_keeps_legacy_context_compatibility(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "legacy_nested",
        "async def main(input, ctx):\n    return 'placeholder'\n",
    )
    with closing(kernel._connect()) as connection, connection:
        connection.execute(
            "UPDATE tool_versions SET source = ? WHERE id = ("
            "SELECT tool_version_id FROM bindings WHERE name = 'legacy_nested'"
            ")",
            (
                "def main(input, ctx):\n"
                "    page = ctx.call_tool('help', {'topic': 'quickstart'})\n"
                "    return page.splitlines()[0]\n",
            ),
        )

    assert kernel.call("legacy_nested", {}) == "# Quickstart"


def test_nested_legacy_tools_do_not_depend_on_default_executor_capacity(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    for name in ("legacy_outer", "legacy_inner"):
        write(
            kernel,
            name,
            "async def main(input, ctx):\n    return 'placeholder'\n",
        )
    with closing(kernel._connect()) as connection, connection:
        connection.execute(
            "UPDATE tool_versions SET source = ? WHERE id = ("
            "SELECT tool_version_id FROM bindings WHERE name = 'legacy_inner'"
            ")",
            ("def main(input, ctx):\n    return 'nested result'\n",),
        )
        connection.execute(
            "UPDATE tool_versions SET source = ? WHERE id = ("
            "SELECT tool_version_id FROM bindings WHERE name = 'legacy_outer'"
            ")",
            ("def main(input, ctx):\n    return ctx.call_tool('legacy_inner', {})\n",),
        )

    async def invoke_with_one_default_worker() -> Any:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(1))
        return await asyncio.wait_for(kernel.call_async("legacy_outer", {}), timeout=2)

    assert asyncio.run(invoke_with_one_default_worker()) == "nested result"


def test_cancelling_legacy_tool_does_not_block_event_loop(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    started = tmp_path / "legacy-started"
    release = tmp_path / "legacy-release"
    finished = tmp_path / "legacy-finished"
    write(
        kernel,
        "legacy_blocking",
        "async def main(input, ctx):\n    return 'placeholder'\n",
    )
    with closing(kernel._connect()) as connection, connection:
        connection.execute(
            "UPDATE tool_versions SET source = ? WHERE id = ("
            "SELECT tool_version_id FROM bindings WHERE name = 'legacy_blocking'"
            ")",
            (
                "import time\n"
                "from pathlib import Path\n\n"
                "def main(input, ctx):\n"
                f"    Path({str(started)!r}).write_text('yes')\n"
                f"    while not Path({str(release)!r}).exists():\n"
                "        time.sleep(0.01)\n"
                f"    Path({str(finished)!r}).write_text('yes')\n"
                "    return 'done'\n",
            ),
        )

    async def cancel_after_start() -> float:
        task = asyncio.create_task(kernel.call_async("legacy_blocking", {}))
        deadline = asyncio.get_running_loop().time() + 2
        while not started.exists():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        began = asyncio.get_running_loop().time()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        elapsed = asyncio.get_running_loop().time() - began
        release.write_text("go")
        deadline = asyncio.get_running_loop().time() + 2
        while not finished.exists():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        return elapsed

    assert asyncio.run(cancel_after_start()) < 0.25


def test_a_tool_can_write_and_call_a_new_tool_before_returning(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    builder_source = """\
async def main(input, ctx):
    written = await ctx.call_tool("write_tool", {
        "name": "born_now",
        "description": "Created during another tool's invocation.",
        "input_schema": {"type": "object"},
        "source": "async def main(input, ctx):\\n    return {'ready': True}\\n",
    })
    called = await ctx.call_tool("born_now", {})
    return {"written": written, "called": called}
"""
    write(kernel, "builder", builder_source)

    result = call_tool(kernel, "builder", {})

    assert result["written"]["active"] is True
    assert result["called"] == {"ready": True}


def test_toolbox_selection_replace_union_remove_and_cwd_default(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db", cwd=tmp_path)
    cwd_default = kernel.list_toolboxes()[0]["name"]
    kernel.create_toolbox("alpha", cwd=tmp_path)
    kernel.create_toolbox("beta")
    session_id = kernel.create_session()

    selected = kernel.select_toolboxes(session_id, ["alpha", "beta"], mode="use")
    assert [item["name"] for item in selected] == ["alpha", "beta"]
    assert selected[0]["primary"] is True

    selected = kernel.select_toolboxes(
        session_id, ["beta", "beta", cwd_default], mode="add"
    )
    assert [item["name"] for item in selected] == [
        "alpha",
        "beta",
        cwd_default,
    ]
    assert [
        item["name"]
        for item in kernel.select_toolboxes(session_id, ["alpha"], mode="remove")
    ] == ["beta", cwd_default]

    kernel.set_cwd_default("alpha", cwd=tmp_path)
    assert [item["name"] for item in kernel.reset_toolboxes(session_id)] == ["alpha"]
    assert [
        item["name"]
        for item in kernel.select_toolboxes(session_id, ["alpha"], mode="remove")
    ] == ["alpha"]
    events = kernel.read_session(session_id, limit=100)["events"]
    assert sum(event["kind"] == "toolbox_selection_changed" for event in events) == 5


def test_toolbox_rename_preserves_identity_tools_and_cwd_default(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db", cwd=tmp_path)
    created = kernel.create_toolbox("alpha", cwd=tmp_path)
    kernel.create_toolbox("beta")
    session_id = kernel.create_session()
    write(
        kernel,
        "kept",
        "async def main(input, ctx):\n    return 'still here'\n",
        session_id=session_id,
    )

    renamed = kernel.rename_toolbox("alpha", "renamed")

    assert renamed == {"id": created["id"], "name": "renamed"}
    assert [item["name"] for item in kernel.active_toolboxes(session_id)] == ["renamed"]
    assert call_tool(kernel, "kept", {}, session_id=session_id) == "still here"
    registered = {item["name"]: item for item in kernel.list_toolboxes()}
    assert "alpha" not in registered
    assert registered["renamed"]["id"] == created["id"]
    assert registered["renamed"]["default"] is True
    assert registered["renamed"]["cwd"] == str(tmp_path.resolve())

    with pytest.raises(ToolboxError) as duplicate:
        kernel.rename_toolbox("renamed", "beta")
    assert duplicate.value.code == "toolbox_exists"

    with pytest.raises(ToolboxError) as invalid:
        kernel.rename_toolbox("renamed", "not a namespace")
    assert invalid.value.code == "invalid_toolbox_name"


def test_sessionless_host_operations_do_not_create_saved_sessions(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db", cwd=tmp_path)

    kernel.bindings()
    kernel.catalog()
    kernel.tool_definitions()
    kernel.write_tool(
        name="host_tool",
        description="Written without a conversation.",
        input_schema={"type": "object"},
        source="async def main(input, ctx):\n    return True\n",
    )

    assert kernel.list_sessions()["sessions"] == []


def test_cwd_association_is_the_default_routing_source(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    database = tmp_path / "toolbox.db"
    kernel = Kernel(database, cwd=first)

    kernel.create_toolbox("research", cwd=second)
    reopened = Kernel(database, cwd=second)
    session_id = reopened.create_session()

    assert [item["name"] for item in reopened.active_toolboxes(session_id)] == [
        "research"
    ]
    research = next(
        item for item in reopened.list_toolboxes() if item["name"] == "research"
    )
    assert research["default"] is True


def test_missing_cwd_is_a_structured_toolbox_error(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db", cwd=tmp_path)

    with pytest.raises(ToolboxError) as error:
        kernel.create_toolbox("missing", cwd=tmp_path / "absent")

    assert error.value.code == "invalid_cwd"


def test_toolbox_union_collisions_versions_mutations_and_core_order(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db", cwd=tmp_path)
    kernel.create_toolbox("alpha")
    kernel.create_toolbox("beta")
    session_id = kernel.create_session()

    kernel.select_toolboxes(session_id, ["alpha"], mode="use")
    write(
        kernel,
        "same",
        "async def main(input, ctx):\n    return 'alpha'\n",
        session_id=session_id,
    )
    write(
        kernel,
        "search_tools",
        "async def main(input, ctx):\n    return {'origin': 'alpha'}\n",
        session_id=session_id,
        input_schema=CORE_SCHEMAS["search_tools"],
        base_version=1,
    )

    kernel.select_toolboxes(session_id, ["beta"], mode="use")
    write(
        kernel,
        "same",
        "async def main(input, ctx):\n    return 'beta-v1'\n",
        session_id=session_id,
    )
    write(
        kernel,
        "same",
        "async def main(input, ctx):\n    return 'beta-v2'\n",
        session_id=session_id,
        base_version=1,
    )

    kernel.select_toolboxes(session_id, ["alpha", "beta"], mode="use")
    assert call_tool(kernel, "same", {}, session_id=session_id) == "alpha"
    assert call_tool(kernel, "same", {}, session_id=session_id, version=1) == "alpha"
    assert kernel.tool_definitions(session_id=session_id)[1]["description"] == (
        "Test tool search_tools."
    )
    matches = [
        item for item in kernel.catalog(session_id=session_id) if item["name"] == "same"
    ]
    assert len(matches) == 1
    assert matches[0]["toolbox"] == "alpha"

    kernel.select_toolboxes(session_id, ["beta", "alpha"], mode="use")
    assert call_tool(kernel, "same", {}, session_id=session_id) == "beta-v2"
    assert call_tool(kernel, "same", {}, session_id=session_id, version=1) == "beta-v1"
    kernel.delete_tool("same", session_id=session_id)
    assert call_tool(kernel, "same", {}, session_id=session_id) == "alpha"
    assert kernel.rollback("same", version=1, toolbox="beta") == {
        "name": "same",
        "from_version": None,
        "to_version": 1,
    }
    assert call_tool(kernel, "same", {}, session_id=session_id) == "beta-v1"


def test_session_toolbox_selection_survives_reopen_and_is_isolated(
    tmp_path: Path,
) -> None:
    database = tmp_path / "toolbox.db"
    kernel = Kernel(database, cwd=tmp_path)
    kernel.create_toolbox("alpha")
    kernel.create_toolbox("beta")
    first = kernel.create_session()
    second = kernel.create_session()
    kernel.select_toolboxes(first, ["alpha", "beta"], mode="use")
    kernel.select_toolboxes(second, ["beta"], mode="use")

    reopened = Kernel(database, cwd=tmp_path)

    assert [item["name"] for item in reopened.active_toolboxes(first)] == [
        "alpha",
        "beta",
    ]
    assert [item["name"] for item in reopened.active_toolboxes(second)] == ["beta"]


def test_legacy_database_migrates_into_a_durable_namespace(tmp_path: Path) -> None:
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE tool_versions (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
            description TEXT NOT NULL, schema_json TEXT NOT NULL, source TEXT NOT NULL,
            created_at TEXT NOT NULL, UNIQUE(name, version)
        );
        CREATE TABLE bindings (
            name TEXT PRIMARY KEY,
            tool_id INTEGER NOT NULL REFERENCES tool_versions(id)
        );
        CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id), seq INTEGER NOT NULL,
            kind TEXT NOT NULL, call_id TEXT, parent_call_id TEXT, tool_name TEXT,
            tool_version INTEGER, payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(session_id, seq)
        );
        """
    )
    connection.execute(
        "INSERT INTO tool_versions VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "legacy_tool",
            1,
            "Legacy tool.",
            json.dumps({"type": "object"}),
            "def main(input, ctx):\n    return 'legacy'\n",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    connection.execute("INSERT INTO bindings VALUES ('legacy_tool', 1)")
    connection.execute(
        "INSERT INTO sessions VALUES ('old-session', '2026-01-01T00:00:00+00:00')"
    )
    connection.execute(
        """
        INSERT INTO events VALUES (
            1, 'old-session', 1, 'call_succeeded', 'call', NULL,
            'legacy_tool', 1, ?, '2026-01-01T00:00:01+00:00'
        )
        """,
        (json.dumps({"result": "legacy"}),),
    )
    connection.commit()
    connection.close()

    kernel = Kernel(database, cwd=tmp_path)

    assert [item["name"] for item in kernel.list_toolboxes()] == ["legacy"]
    assert [item["name"] for item in kernel.active_toolboxes("old-session")] == [
        "legacy"
    ]
    assert kernel.call("legacy_tool", {}, session_id="old-session") == "legacy"
    migrated = kernel.read_session("old-session", limit=100)["events"][0]
    assert migrated["toolbox_id"] == kernel.list_toolboxes()[0]["id"]
    with sqlite3.connect(database) as reopened:
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 7
    assert Kernel(database, cwd=tmp_path).call("legacy_tool", {}) == "legacy"


def test_cli_lists_and_creates_toolbox_namespaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "toolbox.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mechagnome",
            "--db",
            str(database),
            "toolboxes",
            "create",
            "research",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert cli.main() == 0
    assert '"name": "research"' in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["mechagnome", "--db", str(database), "toolboxes", "list"],
    )
    assert cli.main() == 0
    listed = json.loads(capsys.readouterr().out)
    assert "research" in {item["name"] for item in listed}

    monkeypatch.setattr(sys, "argv", ["mechagnome", "toolboxes", "--help"])
    with pytest.raises(SystemExit) as help_exit:
        cli.main()
    help_text = capsys.readouterr().out
    assert help_exit.value.code == 0
    assert "list registered toolboxes" in help_text
    assert "namespace" not in help_text


def test_cli_recovery_failure_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mechagnome",
            "--db",
            str(tmp_path / "toolbox.db"),
            "rollback",
            "missing",
            "1",
        ],
    )

    assert cli.main() == 1
    assert '"code": "unknown_tool"' in capsys.readouterr().out


class ScriptedModel:
    """A deterministic fake model used to prove the harness boundary."""

    def __init__(self) -> None:
        self.turn = 0
        self.tool_names: list[tuple[str, ...]] = []

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.tool_names.append(tuple(tool["name"] for tool in tools))
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                calls=(
                    ToolCall(
                        "write_tool",
                        {
                            "name": "hello",
                            "description": "Say hello.",
                            "input_schema": {"type": "object"},
                            "source": (
                                "async def main(input, ctx):\n"
                                "    return {'message': 'hello'}\n"
                            ),
                        },
                        "write-1",
                    ),
                )
            )
        if self.turn == 2:
            return ModelTurn(
                calls=(ToolCall("call_tool", {"name": "hello", "args": {}}, "call-1"),)
            )
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["content"] == {"message": "hello"}
        return ModelTurn(text="Built and reused hello.")


class ReprogrammingModel:
    """Replace a core implementation and observe its next-turn definition."""

    def __init__(self) -> None:
        self.search_descriptions: list[str] = []

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        search = next(tool for tool in tools if tool["name"] == "search_tools")
        self.search_descriptions.append(search["description"])
        if len(self.search_descriptions) == 1:
            return ModelTurn(
                calls=(
                    ToolCall(
                        "write_tool",
                        {
                            "name": "search_tools",
                            "description": "Live rewritten search.",
                            "input_schema": CORE_SCHEMAS["search_tools"],
                            "source": (
                                "async def main(input, ctx):\n"
                                "    return {'items': []}\n"
                            ),
                            "base_version": 1,
                        },
                        "rewrite-search",
                    ),
                )
            )
        return ModelTurn(text="Search changed.")


class ProviderUsingModel:
    """Exercise one authored tool and provide its nested completion."""

    def __init__(self) -> None:
        self.turn = 0
        self.provider_messages: list[Any] = []

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                calls=(ToolCall("call_tool", {"name": "ask", "args": {}}, "ask"),)
            )
        return ModelTurn(text=str(messages[-1]["content"]))

    def complete(self, messages: Any) -> str:
        self.provider_messages.append(messages)
        return "nested completion"

    def cancel_current(self) -> None:
        pass

    def reset_cancellation(self) -> None:
        pass


class OversizedBatchModel:
    """Request a batch wider than the harness permits, then repair it."""

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        observations = [message for message in messages if message["role"] == "tool"]
        if not observations:
            return ModelTurn(
                calls=tuple(
                    ToolCall("help", {"topic": "toc"}, f"help-{index}")
                    for index in range(3)
                )
            )
        assert len(observations) == 3
        assert {
            observation["content"]["error"]["code"] for observation in observations
        } == {"too_many_tool_calls"}
        return ModelTurn(text="I will use a smaller batch next time.")


class DetachedContinuationModel:
    """Detach a gated tool, do foreground work, then inspect it next prompt."""

    def __init__(self) -> None:
        self.turn = 0
        self.job_id: str | None = None

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                calls=(
                    ToolCall(
                        "call_tool",
                        {"name": "gated", "args": {}, "detach": True},
                        "detach",
                    ),
                )
            )
        if self.turn == 2:
            handle = messages[-1]["content"]
            assert handle["status"] == "running"
            self.job_id = handle["job_id"]
            return ModelTurn(calls=(ToolCall("help", {"topic": "toc"}, "help"),))
        if self.turn == 3:
            assert str(messages[-1]["content"]).startswith("# Mechagnome")
            return ModelTurn(text="Foreground work finished.")
        if self.turn == 4:
            assert self.job_id is not None
            return ModelTurn(
                calls=(ToolCall("call_tool", {"job_id": self.job_id}, "inspect"),)
            )
        inspected = messages[-1]["content"]
        assert inspected["status"] == "succeeded"
        assert "result" in inspected and inspected["result"] is None
        assert "finished" in inspected["output_tail"]
        return ModelTurn(text="Detached work finished.")


class BatchedDetachedModel:
    """Detach work beside another operation and retain its transient updates."""

    def __init__(self) -> None:
        self.job_id: str | None = None

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        observations = [message for message in messages if message["role"] == "tool"]
        if not observations:
            return ModelTurn(
                calls=(
                    ToolCall(
                        "call_tool",
                        {"name": "quick_detached", "args": {}, "detach": True},
                        "detach",
                    ),
                    ToolCall("help", {"topic": "toc"}, "help"),
                )
            )
        assert len(observations) == 2
        handle = observations[0]["content"]
        assert handle["status"] == "running"
        self.job_id = handle["job_id"]
        assert str(observations[1]["content"]).startswith("# Mechagnome")
        return ModelTurn(text="Batch completed.")


def test_parallel_batch_preserves_transient_detached_updates(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "quick_detached",
        "async def main(input, ctx):\n"
        "    print('detached output', flush=True)\n"
        "    return {'ok': True}\n",
    )
    runner = IsolatedToolRunner(kernel)
    harness = Harness(kernel, tool_runner=runner)
    model = BatchedDetachedModel()
    events: list[Any] = []
    conversation = harness.start(model)

    result = conversation.send("run both", on_event=events.append)

    assert result.answer == "Batch completed."
    assert model.job_id is not None
    wait_for_detached(runner, model.job_id, conversation.session_id, status="succeeded")
    detached_events = [event for event in events if event.kind.startswith("detached_")]
    assert {event.kind for event in detached_events} >= {
        "detached_started",
        "detached_finished",
    }
    assert all(event.seq is None for event in detached_events)
    harness.close()


def test_detached_call_continues_foreground_and_is_inspectable_later(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    started = tmp_path / "detached-started"
    release = tmp_path / "detached-release"
    write(
        kernel,
        "gated",
        "import time\n"
        "from pathlib import Path\n\n"
        "async def main(input, ctx):\n"
        "    print('started', flush=True)\n"
        f"    Path({str(started)!r}).write_text('yes')\n"
        f"    while not Path({str(release)!r}).exists():\n"
        "        time.sleep(0.01)\n"
        "    print('finished', flush=True)\n"
        "    return None\n",
    )
    runner = IsolatedToolRunner(kernel)
    harness = Harness(kernel, tool_runner=runner)
    model = DetachedContinuationModel()
    conversation = harness.start(model)

    first = conversation.send("start background work")

    assert first.answer == "Foreground work finished."
    assert model.job_id is not None
    assert started.exists()
    running = wait_for_detached(
        runner,
        model.job_id,
        conversation.session_id,
        status="running",
        output="started",
    )
    assert running == {
        "job_id": model.job_id,
        "status": "running",
        "output_tail": "started\n",
        "truncated": False,
    }
    assert [message["role"] for message in conversation.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    parent_events = kernel.read_session(conversation.session_id, limit=100)["events"]
    observations = [
        event for event in parent_events if event["kind"] == "tool_observation"
    ]
    assert len(observations) == 2
    assert observations[0]["payload"]["observation"] == {
        "job_id": model.job_id,
        "status": "running",
    }

    release.write_text("go")
    wait_for_detached(runner, model.job_id, conversation.session_id, status="succeeded")
    second = conversation.send("check the background work")

    assert second.answer == "Detached work finished."
    harness.close()


def test_detached_output_is_sanitized_bounded_and_providerless(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "noisy",
        "import os\n\n"
        "async def main(input, ctx):\n"
        "    print('before\\x1b[31mred\\x1b[0m')\n"
        "    print('x' * 300000)\n"
        "    print('after\\x1b[32mgreen\\x1b[0m')\n"
        "    os.write(1, b'bad-utf8-\\xff\\n')\n"
        "    try:\n"
        "        await ctx.model_provider.complete(\n"
        "            [{'role': 'user', 'content': 'x'}]\n"
        "        )\n"
        "    except Exception as error:\n"
        "        return {'provider_error': error.code}\n",
    )
    runner = IsolatedToolRunner(kernel)
    session_id = kernel.create_session(kind="conversation")

    handle = runner.start_detached("noisy", {}, session_id=session_id)
    completed = wait_for_detached(
        runner, handle["job_id"], session_id, status="succeeded"
    )

    assert completed["result"] == {"provider_error": "model_provider_unavailable"}
    assert completed["truncated"] is True
    assert len(completed["output_tail"].encode("utf-8")) <= 256 * 1024
    assert "\x1b" not in completed["output_tail"]
    assert "aftergreen" in completed["output_tail"]
    assert "bad-utf8-�" in completed["output_tail"]
    runner.close()


def test_detached_output_reader_does_not_wait_for_escaped_writer(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "escaped_writer",
        "import subprocess\n"
        "import sys\n\n"
        "async def main(input, ctx):\n"
        "    child = subprocess.Popen(\n"
        "        [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
        "        start_new_session=True,\n"
        "    )\n"
        "    print(f'escaped child {child.pid}', flush=True)\n"
        "    return child.pid\n",
    )
    runner = IsolatedToolRunner(kernel, timeout=2)
    session_id = kernel.create_session(kind="conversation")
    child_pid: int | None = None

    try:
        handle = runner.start_detached("escaped_writer", {}, session_id=session_id)
        completed = wait_for_detached(
            runner,
            handle["job_id"],
            session_id,
            status="succeeded",
            timeout=3,
        )
        child_pid = completed["result"]
        assert f"escaped child {child_pid}" in completed["output_tail"]
    finally:
        runner.close()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_detached_output_shutdown_is_bounded_for_continuous_escaped_writer(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "continuous_escaped_writer",
        "import subprocess\n"
        "import sys\n\n"
        "async def main(input, ctx):\n"
        "    code = (\n"
        "        'import os\\n'\n"
        "        'chunk = b\\\"x\\\" * 4096\\n'\n"
        "        'while True:\\n'\n"
        "        '    os.write(1, chunk)\\n'\n"
        "    )\n"
        "    child = subprocess.Popen(\n"
        "        [sys.executable, '-c', code], start_new_session=True\n"
        "    )\n"
        "    return child.pid\n",
    )
    runner = IsolatedToolRunner(kernel, timeout=2)
    session_id = kernel.create_session(kind="conversation")
    child_pid: int | None = None

    try:
        handle = runner.start_detached(
            "continuous_escaped_writer", {}, session_id=session_id
        )
        completed = wait_for_detached(
            runner,
            handle["job_id"],
            session_id,
            status="succeeded",
            timeout=3,
        )
        child_pid = completed["result"]
        assert completed["output_tail"]
    finally:
        runner.close()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_detached_cleanup_stops_owned_descendants_and_joins_job_thread(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "owned_descendant",
        "import subprocess\n"
        "import sys\n\n"
        "async def main(input, ctx):\n"
        "    child = subprocess.Popen(\n"
        "        [sys.executable, '-c', 'import time; time.sleep(30)']\n"
        "    )\n"
        "    return child.pid\n",
    )
    runner = IsolatedToolRunner(kernel, timeout=2)
    session_id = kernel.create_session(kind="conversation")

    handle = runner.start_detached("owned_descendant", {}, session_id=session_id)
    completed = wait_for_detached(
        runner, handle["job_id"], session_id, status="succeeded", timeout=3
    )
    child_pid = completed["result"]
    runner.close()

    assert runner._jobs[handle["job_id"]].thread is not None
    assert not runner._jobs[handle["job_id"]].thread.is_alive()
    deadline = time.monotonic() + 2
    while True:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            os.kill(child_pid, signal.SIGKILL)
            raise AssertionError("owned descendant survived worker-group cleanup")
        time.sleep(0.01)


def test_detached_job_limit_timeout_and_shutdown_are_structured(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    release = tmp_path / "release-all"
    write(
        kernel,
        "waiter",
        "import time\n"
        "from pathlib import Path\n\n"
        "async def main(input, ctx):\n"
        f"    while not Path({str(release)!r}).exists():\n"
        "        time.sleep(0.01)\n"
        "    return input.get('value')\n",
    )
    runner = IsolatedToolRunner(kernel, timeout=5)
    session_id = kernel.create_session(kind="conversation")
    handles = [
        runner.start_detached("waiter", {"value": index}, session_id=session_id)
        for index in range(4)
    ]

    with pytest.raises(ToolboxError) as limit_error:
        runner.start_detached("waiter", {}, session_id=session_id)
    assert limit_error.value.code == "detached_job_limit"

    runner.close()
    runner.close()
    for handle in handles:
        inspected = runner.inspect_detached(handle["job_id"], session_id=session_id)
        assert inspected["status"] == "failed"
        assert inspected["error"]["code"] == "detached_shutdown"
    with pytest.raises(ToolboxError) as closed_error:
        runner.start_detached("waiter", {}, session_id=session_id)
    assert closed_error.value.code == "detached_runner_closed"

    timeout_runner = IsolatedToolRunner(kernel, timeout=0.1)
    timeout_handle = timeout_runner.start_detached("waiter", {}, session_id=session_id)
    timed_out = wait_for_detached(
        timeout_runner,
        timeout_handle["job_id"],
        session_id,
        status="failed",
    )
    assert timed_out["error"]["code"] == "tool_timeout"
    timeout_runner.close()


def test_detached_job_limit_is_atomic_and_releases_completed_slots(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    release = tmp_path / "release-concurrent"
    write(
        kernel,
        "concurrent_waiter",
        "import time\n"
        "from pathlib import Path\n\n"
        "async def main(input, ctx):\n"
        f"    while not Path({str(release)!r}).exists():\n"
        "        time.sleep(0.01)\n"
        "    return input['value']\n",
    )
    runner = IsolatedToolRunner(kernel, timeout=5)
    session_id = kernel.create_session(kind="conversation")
    barrier = threading.Barrier(5)
    outcomes: list[dict[str, Any] | ToolboxError] = []
    outcomes_lock = threading.Lock()

    def start(index: int) -> None:
        barrier.wait()
        try:
            outcome: dict[str, Any] | ToolboxError = runner.start_detached(
                "concurrent_waiter", {"value": index}, session_id=session_id
            )
        except ToolboxError as error:
            outcome = error
        with outcomes_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=start, args=(index,)) for index in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    handles = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    errors = [outcome for outcome in outcomes if isinstance(outcome, ToolboxError)]
    assert len(handles) == 4
    assert [error.code for error in errors] == ["detached_job_limit"]

    release.write_text("go")
    for handle in handles:
        wait_for_detached(runner, handle["job_id"], session_id, status="succeeded")
    replacement = runner.start_detached(
        "concurrent_waiter", {"value": 5}, session_id=session_id
    )
    wait_for_detached(runner, replacement["job_id"], session_id, status="succeeded")
    runner.close()


def test_detached_job_retention_evicts_oldest_completed_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolation_module, "_MAX_RETAINED_DETACHED_JOBS", 2)
    kernel = kernel_at(tmp_path)
    write(
        kernel, "retained", "async def main(input, ctx):\n    return input['value']\n"
    )
    runner = IsolatedToolRunner(kernel)
    session_id = kernel.create_session(kind="conversation")
    handles: list[dict[str, Any]] = []

    for value in range(3):
        handle = runner.start_detached(
            "retained", {"value": value}, session_id=session_id
        )
        wait_for_detached(runner, handle["job_id"], session_id, status="succeeded")
        handles.append(handle)

    with pytest.raises(ToolboxError) as evicted:
        runner.inspect_detached(handles[0]["job_id"], session_id=session_id)
    assert evicted.value.code == "unknown_detached_job"
    assert (
        runner.inspect_detached(handles[1]["job_id"], session_id=session_id)["result"]
        == 1
    )
    assert (
        runner.inspect_detached(handles[2]["job_id"], session_id=session_id)["result"]
        == 2
    )
    runner.close()


def test_detached_result_size_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolation_module, "_MAX_DETACHED_RESULT_BYTES", 32)
    kernel = kernel_at(tmp_path)
    write(kernel, "large_result", "async def main(input, ctx):\n    return 'x' * 100\n")
    runner = IsolatedToolRunner(kernel)
    session_id = kernel.create_session(kind="conversation")

    handle = runner.start_detached("large_result", {}, session_id=session_id)
    completed = wait_for_detached(runner, handle["job_id"], session_id, status="failed")

    assert completed["error"] == {
        "code": "detached_result_too_large",
        "message": "detached result exceeds the retained byte limit",
        "details": {"limit_bytes": 32},
    }
    runner.close()


def test_detached_job_uses_custom_dispatcher_and_is_parent_scoped(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "call_tool",
        "async def main(input, ctx):\n"
        "    return {'custom_name': input['name'], 'args': input['args']}\n",
        input_schema=CORE_SCHEMAS["call_tool"],
        base_version=1,
    )
    runner = IsolatedToolRunner(kernel)
    owner = kernel.create_session(kind="conversation")
    foreign = kernel.create_session(kind="conversation")

    handle = runner.start_detached("not_a_real_tool", {"x": 1}, session_id=owner)
    completed = wait_for_detached(runner, handle["job_id"], owner, status="succeeded")

    assert completed["result"] == {"custom_name": "not_a_real_tool", "args": {"x": 1}}
    with pytest.raises(ToolboxError) as error:
        runner.inspect_detached(handle["job_id"], session_id=foreign)
    assert error.value.code == "unknown_detached_job"
    runner.close()


def test_call_tool_control_validation_and_detach_false_compatibility(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    write(kernel, "echo", "async def main(input, ctx):\n    return input\n")

    class ControlModel:
        def __init__(self) -> None:
            self.turn = 0

        def respond(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> ModelTurn:
            self.turn += 1
            if self.turn == 1:
                return ModelTurn(
                    calls=(
                        ToolCall(
                            "call_tool",
                            {
                                "name": "echo",
                                "args": {"ok": True},
                                "detach": False,
                            },
                            "sync",
                        ),
                    )
                )
            if self.turn == 2:
                assert messages[-1]["content"] == {"ok": True}
                return ModelTurn(
                    calls=(
                        ToolCall(
                            "call_tool",
                            {"job_id": "missing", "name": "mixed"},
                            "mixed",
                        ),
                    )
                )
            assert messages[-1]["content"]["error"]["code"] == (
                "invalid_call_tool_request"
            )
            if self.turn == 3:
                return ModelTurn(calls=(ToolCall("call_tool", {}, "empty"),))
            if self.turn == 4:
                return ModelTurn(
                    calls=(ToolCall("call_tool", {"detach": False}, "partial"),)
                )
            return ModelTurn(text="validated")

    assert (
        Harness(kernel).run(ControlModel(), "exercise controls").answer == "validated"
    )


class ParallelBatchModel:
    """Request two independent operations and inspect both observations."""

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        observations = [message for message in messages if message["role"] == "tool"]
        if not observations:
            return ModelTurn(
                calls=(
                    ToolCall("help", {}, "help-call"),
                    ToolCall("search_tools", {"query": "help"}, "search-call"),
                )
            )
        assert [message["content"] for message in observations] == [
            {"name": "help"},
            {"name": "search_tools"},
        ]
        return ModelTurn(text="Both completed.")


class RealParallelBatchModel:
    """Exercise concurrent isolated workers and their shared event stream."""

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        observations = [message for message in messages if message["role"] == "tool"]
        if not observations:
            return ModelTurn(
                calls=tuple(
                    ToolCall("help", {"topic": "toc"}, f"help-{index}")
                    for index in range(4)
                )
            )
        assert len(observations) == 4
        assert all(isinstance(message["content"], str) for message in observations)
        return ModelTurn(text="All completed.")


def test_harness_refreshes_core_descriptions_each_turn(tmp_path: Path) -> None:
    model = ReprogrammingModel()

    Harness(kernel_at(tmp_path)).run(model, "Rewrite search.")

    assert model.search_descriptions == [
        "Search or browse active tools by metadata and hierarchical namespace.",
        "Live rewritten search.",
    ]


def test_harness_routes_root_and_tool_traffic_through_one_provider(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "ask",
        "async def main(input, ctx):\n"
        "    return await ctx.model_provider.complete([\n"
        "        {'role': 'user', 'content': 'nested'},\n"
        "    ])\n",
    )
    model = ProviderUsingModel()

    result = Harness(kernel).run(
        model,
        "ask through the provider",
        model_provider=model,
    )

    assert result.answer == "nested completion"
    assert model.provider_messages == [[{"role": "user", "content": "nested"}]]

    implicit = Harness(kernel).run(
        ModelProvider(kernel, ProviderUsingModel()),
        "ask through the unified provider",
    )
    assert implicit.answer == "nested completion"


def test_raw_root_transport_does_not_implicitly_grant_tool_model_spend(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    write(
        kernel,
        "ask",
        "async def main(input, ctx):\n"
        "    return await ctx.model_provider.complete([\n"
        "        {'role': 'user', 'content': 'nested'},\n"
        "    ])\n",
    )

    result = Harness(kernel).run(ProviderUsingModel(), "try nested")

    assert "model_provider_unavailable" in result.answer


def test_harness_preserves_providerless_legacy_tool_runner_signature(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)

    class LegacyRunner:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call(
            self,
            name: str,
            args: dict[str, Any],
            *,
            session_id: str,
            on_event: Any = None,
            cancelled: Any = None,
        ) -> Any:
            self.calls.append(name)
            return {"legacy": True}

    runner = LegacyRunner()
    result = Harness(kernel, tool_runner=runner).run(
        ProviderUsingModel(),
        "use legacy runner",
    )

    assert result.answer == "{'legacy': True}"
    assert runner.calls == ["call_tool"]

    conversation = Harness(kernel, tool_runner=runner).start(
        ProviderUsingModel(),
        model_provider=ProviderUsingModel(),
    )
    assert conversation.model_session.provider is not None


def test_harness_processes_independent_tool_calls_concurrently(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)

    class BarrierRunner:
        def __init__(self) -> None:
            self.barrier = threading.Barrier(2)

        def call(
            self,
            name: str,
            args: dict[str, Any],
            *,
            session_id: str,
            on_event: Any = None,
            cancelled: Any = None,
        ) -> Any:
            self.barrier.wait(timeout=1)
            return {"name": name}

    result = Harness(kernel, tool_runner=BarrierRunner()).run(
        ParallelBatchModel(), "Run both."
    )

    assert result.answer == "Both completed."


def test_parallel_isolated_tools_emit_each_durable_event_once(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    events: list[Any] = []

    result = (
        Harness(kernel)
        .start(RealParallelBatchModel())
        .send("Run several.", on_event=events.append)
    )

    sequences = [event.seq for event in events if event.seq is not None]
    tool_events = [
        event
        for event in events
        if event.kind in {"call_started", "call_succeeded", "call_failed"}
    ]
    assert result.answer == "All completed."
    assert sequences == sorted(set(sequences))
    assert len([event for event in tool_events if event.kind == "call_started"]) == 4
    assert len([event for event in tool_events if event.kind == "call_succeeded"]) == 4
    assert not any(event.kind == "call_failed" for event in tool_events)


def test_harness_exposes_five_core_operations_plus_run_agent_and_saves_everything(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    model = ScriptedModel()

    result = Harness(kernel).run(model, "Create and use a greeting tool.")

    assert result.answer == "Built and reused hello."
    assert result.turns == 3
    assert model.tool_names == [
        MODEL_ACTION_NAMES,
        MODEL_ACTION_NAMES,
        MODEL_ACTION_NAMES,
    ]
    assert tuple(tool["name"] for tool in kernel.tool_definitions()) == CORE_NAMES
    kinds = [
        event["kind"]
        for event in kernel.read_session(result.session_id, limit=100)["events"]
    ]
    assert kinds[0] == "user"
    assert "binding_changed" in kinds
    assert kinds[-1] == "final"


def test_harness_does_not_limit_model_turns(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)

    class LongRunningModel:
        def __init__(self) -> None:
            self.turn = 0

        def respond(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> ModelTurn:
            self.turn += 1
            if self.turn <= 51:
                return ModelTurn(calls=(ToolCall("help", {}, f"help-{self.turn}"),))
            return ModelTurn(text="Finished after the former limit.")

    class ImmediateRunner:
        def call(
            self,
            name: str,
            args: dict[str, Any],
            *,
            session_id: str,
            on_event: Any = None,
            cancelled: Any = None,
        ) -> str:
            return "help"

    result = Harness(kernel, tool_runner=ImmediateRunner()).run(
        LongRunningModel(), "Keep working."
    )

    assert result.answer == "Finished after the former limit."
    assert result.turns == 52


def test_harness_rejects_oversized_batches_without_partial_execution(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)

    result = Harness(kernel, max_calls_per_turn=2).run(
        OversizedBatchModel(), "Read several help pages."
    )

    events = kernel.read_session(result.session_id, limit=100)["events"]
    assert not any(event["kind"] == "call_started" for event in events)
    observations = [event for event in events if event["kind"] == "tool_observation"]
    assert len(observations) == 3
    assert result.answer == "I will use a smaller batch next time."
