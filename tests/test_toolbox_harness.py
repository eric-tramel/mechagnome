"""Behavioral tests for the metaprogrammable toolbox proof."""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from mechagnome import Harness, Kernel, ModelTurn, ToolboxError, ToolCall
from mechagnome import __main__ as cli
from mechagnome.bootstrap import CORE_NAMES, CORE_SCHEMAS


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
        "        'session_id': ctx.sessions.id,\n"
        "        'saw_start': any(e['kind'] == 'call_started' and "
        "e['tool_name'] == 'recall' for e in events),\n"
        "    }\n",
    )

    assert call_tool(kernel, "double", {"x": 9}, session_id=session_id) == 18
    recall = call_tool(kernel, "recall", {}, session_id=session_id)
    assert recall == {"session_id": session_id, "saw_start": True}

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
