"""Pinned schemas and editable bootstrap tool implementations for Mechagnome."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any


@dataclass(frozen=True)
class BootstrapTool:
    """One initial implementation installed into a distinguished core slot."""

    name: str
    description: str
    input_schema: dict[str, Any]
    source: str


HELP_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
    },
    "additionalProperties": False,
}

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "include_core": {"type": "boolean"},
        "cursor": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
    },
    "required": ["query"],
    "additionalProperties": False,
}

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "version": {"type": "integer", "minimum": 1},
    },
    "required": ["name"],
    "additionalProperties": False,
}

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "input_schema": {"type": "object"},
        "source": {"type": "string"},
        "base_version": {"type": "integer", "minimum": 1},
    },
    "required": ["name", "description", "input_schema", "source"],
    "additionalProperties": False,
}

CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "args": {"type": "object"},
        "version": {"type": "integer", "minimum": 1},
    },
    "required": ["name", "args"],
    "additionalProperties": False,
}


HELP_SOURCE = dedent(
    '''\
    """Progressive documentation for agents entering an empty toolbox."""

    from importlib.resources import files

    DOCUMENTS = {
        "toc": "index.md",
        "quickstart": "getting-started/quickstart.md",
        "authoring": "tools/authoring.md",
        "composition": "tools/composition.md",
        "sessions": "runtime/sessions.md",
        "versioning": "runtime/versioning.md",
        "core": "runtime/core.md",
    }

    def main(input, ctx):
        topic = input.get("topic") or "toc"
        if topic not in DOCUMENTS:
            return {
                "error": f"unknown help topic: {topic}",
                "topics": list(DOCUMENTS),
            }
        document = files("mechagnome").joinpath(
            "assets", "help", *DOCUMENTS[topic].split("/")
        )
        return document.read_text(encoding="utf-8")
    '''
)

SEARCH_SOURCE = dedent(
    '''\
    """Simple replaceable lexical search over active tool metadata."""

    def main(input, ctx):
        query = input["query"].strip().lower()
        include_core = input.get("include_core", True)
        cursor = max(0, input.get("cursor", 0))
        limit = min(50, max(1, input.get("limit", 10)))
        terms = query.split()
        ranked = []
        for tool in ctx.kernel.catalog(include_core=include_core):
            haystack = " ".join([
                tool["name"], tool["description"],
                str(tool["input_schema"]), tool["source"],
            ]).lower()
            if terms and not all(term in haystack for term in terms):
                continue
            score = sum(haystack.count(term) for term in terms)
            if query and tool["name"].lower() == query:
                score += 100
            ranked.append((score, tool))
        ranked.sort(key=lambda item: (-item[0], item[1]["name"]))
        items = [tool for _, tool in ranked]
        next_cursor = cursor + limit if cursor + limit < len(items) else None
        return {
            "items": items[cursor:cursor + limit],
            "next_cursor": next_cursor,
            "total": len(items),
        }
    '''
)

READ_SOURCE = dedent(
    '''\
    """Read the exact stored source and metadata for a tool version."""

    def main(input, ctx):
        return ctx.kernel.read_tool_source(
            input["name"], version=input.get("version")
        )
    '''
)

WRITE_SOURCE = dedent(
    '''\
    """Compile, store, and activate an immutable tool version."""

    def main(input, ctx):
        return ctx.kernel.write_tool(
            name=input["name"],
            description=input["description"],
            input_schema=input["input_schema"],
            source=input["source"],
            base_version=input.get("base_version"),
        )
    '''
)

CALL_SOURCE = dedent(
    '''\
    """Resolve and execute an active or exact tool version."""

    def main(input, ctx):
        return ctx.kernel.execute(
            input["name"], input["args"], version=input.get("version")
        )
    '''
)


BOOTSTRAP_TOOLS = (
    BootstrapTool(
        "help",
        "Read progressive documentation for using and extending the toolbox.",
        HELP_SCHEMA,
        HELP_SOURCE,
    ),
    BootstrapTool(
        "search_tools",
        "Search active tools by name, description, schema, and source.",
        SEARCH_SCHEMA,
        SEARCH_SOURCE,
    ),
    BootstrapTool(
        "read_tool_source",
        "Read the source and metadata of an active or historical tool version.",
        READ_SCHEMA,
        READ_SOURCE,
    ),
    BootstrapTool(
        "write_tool",
        "Create and activate an immutable tool version.",
        WRITE_SCHEMA,
        WRITE_SOURCE,
    ),
    BootstrapTool(
        "call_tool",
        "Invoke an active or exact tool version with a JSON object.",
        CALL_SCHEMA,
        CALL_SOURCE,
    ),
)

CORE_NAMES = tuple(tool.name for tool in BOOTSTRAP_TOOLS)
CORE_SCHEMAS = {tool.name: tool.input_schema for tool in BOOTSTRAP_TOOLS}
