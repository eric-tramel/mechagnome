"""Behavioral tests for the metaprogrammable toolbox proof."""

from __future__ import annotations

import json
import sqlite3
import stat
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from mechagnome import Harness, Kernel, ModelTurn, ToolboxError, ToolCall
from mechagnome import __main__ as cli
from mechagnome.bootstrap import CORE_NAMES, CORE_SCHEMAS, HELP_SOURCE


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
    stale_source = "def main(input, ctx):\n    return 'stale default'\n"
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
    refreshed = reopened.read_tool_source("help", version=1)
    assert refreshed["description"] == (
        "Read progressive documentation for using and extending the toolbox."
    )
    assert refreshed["input_schema"] == CORE_SCHEMAS["help"]
    assert refreshed["source"] == HELP_SOURCE
    assert (
        reopened.read_tool_source("help", version=1, toolbox="secondary")["source"]
        == HELP_SOURCE
    )


def test_reopen_preserves_active_core_version_two_override(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    replacement_source = "def main(input, ctx):\n    return 'custom help'\n"
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
        "ctx.call_tool(name, args, version=None)",
        "ctx.caller_session_id",
        "ctx.sessions.current(after=0, limit=50)",
        "ctx.sessions.read(session_id, after=0, limit=50)",
        "ctx.sessions.list(limit=20, cursor=0)",
        "ctx.model_provider.complete([",
        "ctx.kernel.catalog(include_core=True)",
        "ctx.kernel.read_tool_source(name, version=None)",
        "ctx.kernel.write_tool(...)",
        "ctx.kernel.execute(name, args, version=None)",
    ):
        assert public_api in authoring

    composition = kernel.call("help", {"topic": "composition"})
    assert "Each nested invocation receives its own context object" in composition
    assert "base_version" in composition
    assert "eight completion attempts per top-level call tree" in composition

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


def test_write_immediately_search_read_and_call(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    source = "def main(input, ctx):\n    return {'echo': input['value']}\n"

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
    }
    assert call_tool(kernel, "echo", {"value": "hello"}) == {"echo": "hello"}
    search = kernel.call("search_tools", {"query": "echo"})
    assert search["items"][0]["name"] == "echo"
    read = kernel.call("read_tool_source", {"name": "echo"})
    assert read["source"] == source
    assert read["active"] is True


def test_versions_exact_calls_and_stale_base(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    version_one = "def main(input, ctx):\n    return 1\n"
    version_two = "def main(input, ctx):\n    return 2\n"

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
        "def main(input, ctx):\n    return 1\n",
        session_id=creator_one,
    )
    call_tool(kernel, "number", {}, session_id=caller)
    write(
        kernel,
        "number",
        "def main(input, ctx):\n    return 2\n",
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


def test_tools_compose_and_read_the_live_session(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    session_id = kernel.create_session()
    session_help = kernel.call("help", {"topic": "sessions"})
    assert "ctx.caller_session_id" in session_help
    write(kernel, "add", "def main(input, ctx):\n    return input['a'] + input['b']\n")
    write(
        kernel,
        "double",
        "def main(input, ctx):\n"
        "    return ctx.call_tool('add', {'a': input['x'], 'b': input['x']})\n",
    )
    write(
        kernel,
        "recall",
        "def main(input, ctx):\n"
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


def test_all_core_source_is_readable_and_schema_is_pinned(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)

    for name in CORE_NAMES:
        result = kernel.call("read_tool_source", {"name": name})
        assert result["source"].startswith('"""')
        assert result["input_schema"] == CORE_SCHEMAS[name]

    with pytest.raises(ToolboxError) as error:
        write(
            kernel,
            "search_tools",
            "def main(input, ctx):\n    return {}\n",
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
        "def main(input, ctx):\n"
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
        "def main(input, ctx):\n    return ctx.call_tool('missing', {})\n",
    )
    replacement_source = (
        "def main(input, ctx):\n"
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
        "def main(input, ctx):\n    return {'value': 42}\n",
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
        "def main(input, ctx):\n    return ctx.call_tool('forever', {})\n",
    )

    with pytest.raises(ToolboxError) as error:
        call_tool(kernel, "forever", {}, session_id=session_id)
    assert error.value.code == "max_depth"
    events = kernel.read_session(session_id, limit=100)["events"]
    assert any(event["kind"] == "call_failed" for event in events)


def test_invalid_source_never_moves_the_binding(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    write(kernel, "stable", "def main(input, ctx):\n    return 'old'\n")

    with pytest.raises(ToolboxError) as error:
        write(
            kernel,
            "stable",
            "def main(input, ctx)\n    return 'broken'\n",
            base_version=1,
        )
    assert error.value.code == "invalid_source"
    assert call_tool(kernel, "stable", {}) == "old"
    assert next(item for item in kernel.bindings() if item["name"] == "stable")[
        "versions"
    ] == [1]


def test_a_tool_can_write_and_call_a_new_tool_before_returning(tmp_path: Path) -> None:
    kernel = kernel_at(tmp_path)
    builder_source = """\
def main(input, ctx):
    written = ctx.call_tool("write_tool", {
        "name": "born_now",
        "description": "Created during another tool's invocation.",
        "input_schema": {"type": "object"},
        "source": "def main(input, ctx):\\n    return {'ready': True}\\n",
    })
    called = ctx.call_tool("born_now", {})
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
        source="def main(input, ctx):\n    return True\n",
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
        "def main(input, ctx):\n    return 'alpha'\n",
        session_id=session_id,
    )
    write(
        kernel,
        "search_tools",
        "def main(input, ctx):\n    return {'origin': 'alpha'}\n",
        session_id=session_id,
        input_schema=CORE_SCHEMAS["search_tools"],
        base_version=1,
    )

    kernel.select_toolboxes(session_id, ["beta"], mode="use")
    write(
        kernel,
        "same",
        "def main(input, ctx):\n    return 'beta-v1'\n",
        session_id=session_id,
    )
    write(
        kernel,
        "same",
        "def main(input, ctx):\n    return 'beta-v2'\n",
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
        assert reopened.execute("PRAGMA user_version").fetchone()[0] == 2
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
                                "def main(input, ctx):\n"
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
                                "def main(input, ctx):\n    return {'items': []}\n"
                            ),
                            "base_version": 1,
                        },
                        "rewrite-search",
                    ),
                )
            )
        return ModelTurn(text="Search changed.")


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


def test_harness_refreshes_core_descriptions_each_turn(tmp_path: Path) -> None:
    model = ReprogrammingModel()

    Harness(kernel_at(tmp_path)).run(model, "Rewrite search.")

    assert model.search_descriptions == [
        "Search active tools by name, description, schema, and source.",
        "Live rewritten search.",
    ]


def test_harness_exposes_only_five_operations_and_saves_everything(
    tmp_path: Path,
) -> None:
    kernel = kernel_at(tmp_path)
    model = ScriptedModel()

    result = Harness(kernel).run(model, "Create and use a greeting tool.")

    assert result.answer == "Built and reused hello."
    assert result.turns == 3
    assert model.tool_names == [CORE_NAMES, CORE_NAMES, CORE_NAMES]
    kinds = [
        event["kind"]
        for event in kernel.read_session(result.session_id, limit=100)["events"]
    ]
    assert kinds[0] == "user"
    assert "binding_changed" in kinds
    assert kinds[-1] == "final"


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
