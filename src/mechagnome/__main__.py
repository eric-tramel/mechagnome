"""Command-line demo and host recovery operations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mechagnome.kernel import Kernel, ToolboxError
from mechagnome.openrouter import DEFAULT_MODEL, OpenRouterModel
from mechagnome.tui import run_tui

ADD_SOURCE = """\
def main(input, ctx):
    return {"sum": input["a"] + input["b"]}
"""

DOUBLE_SOURCE = """\
def main(input, ctx):
    return ctx.call_tool("add", {"a": input["value"], "b": input["value"]})
"""

RECALL_SOURCE = """\
def main(input, ctx):
    current = ctx.sessions.current(limit=100)
    return {
        "session_id": ctx.sessions.id,
        "saw_own_start": any(
            event["kind"] == "call_started" and event["tool_name"] == "recall"
            for event in current["events"]
        ),
        "event_count": len(current["events"]),
    }
"""

SEARCH_REPLACEMENT = """\
def main(input, ctx):
    return {"replacement": True, "query": input["query"]}
"""


def _call(kernel: Kernel, name: str, args: dict[str, Any], session_id: str) -> Any:
    try:
        return kernel.call(name, args, session_id=session_id)
    except ToolboxError as error:
        return error.to_dict()


def demo(kernel: Kernel) -> dict[str, Any]:
    """Exercise the defining metaprogramming claims deterministically."""
    session_id = kernel.create_session()
    object_schema = {"type": "object"}
    add_schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }
    unary_schema = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
    }

    written_add = _call(
        kernel,
        "write_tool",
        {
            "name": "add",
            "description": "Add two numbers.",
            "input_schema": add_schema,
            "source": ADD_SOURCE,
        },
        session_id,
    )
    added = _call(
        kernel, "call_tool", {"name": "add", "args": {"a": 2, "b": 3}}, session_id
    )
    _call(
        kernel,
        "write_tool",
        {
            "name": "double",
            "description": "Double a number by composing add.",
            "input_schema": unary_schema,
            "source": DOUBLE_SOURCE,
        },
        session_id,
    )
    doubled = _call(
        kernel,
        "call_tool",
        {"name": "double", "args": {"value": 7}},
        session_id,
    )
    _call(
        kernel,
        "write_tool",
        {
            "name": "recall",
            "description": "Inspect the current durable session.",
            "input_schema": object_schema,
            "source": RECALL_SOURCE,
        },
        session_id,
    )
    recalled = _call(kernel, "call_tool", {"name": "recall", "args": {}}, session_id)

    search_source = _call(
        kernel, "read_tool_source", {"name": "search_tools"}, session_id
    )
    previous_search_version = search_source["version"]
    _call(
        kernel,
        "write_tool",
        {
            "name": "search_tools",
            "description": "A deliberately replaced search implementation.",
            "input_schema": search_source["input_schema"],
            "source": SEARCH_REPLACEMENT,
            "base_version": previous_search_version,
        },
        session_id,
    )
    replaced_search = _call(kernel, "search_tools", {"query": "anything"}, session_id)
    rollback = kernel.rollback("search_tools", version=previous_search_version)
    restored_search = _call(kernel, "search_tools", {"query": "add"}, session_id)

    reopened = Kernel(kernel.db_path)
    persisted = _call(
        reopened,
        "call_tool",
        {"name": "add", "args": {"a": 20, "b": 22}},
        session_id,
    )
    return {
        "session_id": session_id,
        "written_add": written_add,
        "added": added,
        "doubled": doubled,
        "recalled": recalled,
        "replaced_search": replaced_search,
        "rollback": rollback,
        "restored_search_matches": restored_search.get("total", 0),
        "persisted": persisted,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the host CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db_path(),
        help="SQLite state path",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MECHAGNOME_MODEL", DEFAULT_MODEL),
        help="OpenRouter model slug",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("tui", help="launch the interactive agent interface")
    subparsers.add_parser("demo", help="run the deterministic growth demo")
    bindings_parser = subparsers.add_parser("bindings", help="inspect active bindings")
    bindings_parser.add_argument("--toolbox", help="inspect one named toolbox")
    rollback_parser = subparsers.add_parser(
        "rollback", help="restore a prior version without calling editable code"
    )
    rollback_parser.add_argument("name")
    rollback_parser.add_argument("version", type=int)
    rollback_parser.add_argument("--toolbox", help="target one named toolbox")
    toolbox_parser = subparsers.add_parser(
        "toolboxes", help="list and configure toolbox namespaces"
    )
    toolbox_actions = toolbox_parser.add_subparsers(
        dest="toolbox_action", required=True
    )
    toolbox_actions.add_parser("list", help="list registered namespaces")
    create_parser = toolbox_actions.add_parser("create", help="create a namespace")
    create_parser.add_argument("name")
    create_parser.add_argument("--cwd", type=Path)
    default_parser = toolbox_actions.add_parser(
        "set-default", help="set a cwd's default namespace"
    )
    default_parser.add_argument("name")
    default_parser.add_argument("--cwd", type=Path)
    return parser


def default_db_path() -> Path:
    """Return the persistent per-user state location."""
    override = os.environ.get("MECHAGNOME_DB")
    if override:
        return Path(override).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home).expanduser() if data_home else Path.home() / ".local/share"
    return root / "mechagnome/toolbox.db"


def main() -> int:
    """Launch the TUI by default, or run a host maintenance command."""
    args = build_parser().parse_args()
    kernel = Kernel(args.db)
    if args.command in {None, "tui"}:
        model = OpenRouterModel(model=args.model)
        run_tui(kernel, model, model_name=args.model)
        return 0
    try:
        if args.command == "demo":
            result: Any = demo(kernel)
        elif args.command == "bindings":
            result = kernel.bindings(toolbox=args.toolbox, include_origin=True)
        elif args.command == "rollback":
            result = kernel.rollback(
                args.name, version=args.version, toolbox=args.toolbox
            )
        elif args.toolbox_action == "list":
            result = kernel.list_toolboxes()
        elif args.toolbox_action == "create":
            result = kernel.create_toolbox(args.name, cwd=args.cwd)
        else:
            result = kernel.set_cwd_default(args.name, cwd=args.cwd)
    except ToolboxError as error:
        result = error.to_dict()
        status = 1
    else:
        status = 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
