"""Pinned schemas and editable bootstrap tool implementations for Mechagnome."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any

NAMESPACE_PATH_MAX = 255
NAMESPACE_PATH_PATTERN = (
    r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}"
    r"(?:/[A-Za-z0-9_][A-Za-z0-9_.-]{0,63})*$"
)


@dataclass(frozen=True)
class BootstrapTool:
    """One code-shipped default for a distinguished core slot."""

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
        "namespace": {
            "type": "string",
            "maxLength": NAMESPACE_PATH_MAX,
            "pattern": NAMESPACE_PATH_PATTERN,
        },
        "cursor": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
    },
    "required": ["query"],
    "additionalProperties": False,
}

LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "namespace": {
            "type": "string",
            "maxLength": NAMESPACE_PATH_MAX,
            "pattern": NAMESPACE_PATH_PATTERN,
        },
        "cursor": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
    },
    "additionalProperties": False,
}

VIEW_SCHEMA = {
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
        "source": {
            "type": "string",
            "description": (
                "Python source defining async def main(input, ctx); await async "
                "context operations such as ctx.call_tool."
            ),
        },
        "base_version": {"type": "integer", "minimum": 1},
        "namespaces": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "string",
                "maxLength": NAMESPACE_PATH_MAX,
                "pattern": NAMESPACE_PATH_PATTERN,
            },
        },
    },
    "required": ["name"],
    "oneOf": [
        {"required": ["description", "input_schema", "source"]},
        {
            "required": ["namespaces"],
            "not": {
                "anyOf": [
                    {"required": ["description"]},
                    {"required": ["input_schema"]},
                    {"required": ["source"]},
                ]
            },
        },
    ],
    "additionalProperties": False,
}

CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "args": {"type": "object"},
        "version": {"type": "integer", "minimum": 1},
        "detach": {"type": "boolean"},
        "job_id": {"type": "string"},
    },
    "additionalProperties": False,
}

DELETE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "version": {"type": "integer", "minimum": 1},
    },
    "required": ["name"],
    "additionalProperties": False,
}

RUN_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
        "prompt": {"type": "string"},
        "mode": {"type": "string", "enum": ["continue", "spawn", "fork"]},
        "detach": {"type": "boolean"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "job_id": {"type": "string"},
    },
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
        "namespaces": "runtime/namespaces.md",
        "toolboxes": "runtime/toolboxes.md",
        "versioning": "runtime/versioning.md",
        "core": "runtime/core.md",
    }

    async def main(input, ctx):
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
    """BM25 search over the active tool catalog."""

    import json
    import math
    import re
    from collections import Counter


    FIELD_WEIGHTS = (
        ("name", 8.0),
        ("namespaces", 6.0),
        ("description", 4.0),
        ("input_schema", 2.0),
        ("source", 1.0),
    )


    def _tokens(value):
        if not isinstance(value, str):
            value = json.dumps(value, sort_keys=True)
        value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\\1 \\2", value)
        value = re.sub(r"([a-z0-9])([A-Z])", r"\\1 \\2", value)
        return re.findall(r"[^\\W_]+", value.lower())


    def _field_scores(tools, query_tokens, field):
        corpus = [_tokens(tool[field]) for tool in tools]
        document_count = len(corpus)
        if not document_count or not any(corpus):
            return [0.0] * len(tools)

        frequencies = [Counter(document) for document in corpus]
        average_length = sum(map(len, corpus)) / document_count
        document_frequencies = Counter()
        for frequency in frequencies:
            document_frequencies.update(frequency.keys())

        k1 = 1.5
        b = 0.75
        scores = [0.0] * document_count
        for term in set(query_tokens):
            matching_documents = document_frequencies[term]
            if not matching_documents:
                continue
            inverse_document_frequency = math.log(
                1 + (
                    document_count - matching_documents + 0.5
                ) / (matching_documents + 0.5)
            )
            for index, frequency in enumerate(frequencies):
                term_frequency = frequency[term]
                if not term_frequency:
                    continue
                length_ratio = len(corpus[index]) / average_length
                saturation = term_frequency + k1 * (1 - b + b * length_ratio)
                scores[index] += inverse_document_frequency * (
                    term_frequency * (k1 + 1) / saturation
                )
        return scores

    async def main(input, ctx):
        query = input["query"].strip()
        include_core = input.get("include_core", True)
        cursor = max(0, input.get("cursor", 0))
        limit = min(50, max(1, input.get("limit", 10)))
        tools = ctx.kernel.catalog(
            include_core=include_core,
            namespace=input.get("namespace"),
        )
        query_tokens = _tokens(query)

        if not query_tokens:
            items = sorted(tools, key=lambda tool: tool["name"])
            next_cursor = cursor + limit if cursor + limit < len(items) else None
            return {
                "items": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "namespaces": tool["namespaces"],
                    }
                    for tool in items[cursor:cursor + limit]
                ],
                "next_cursor": next_cursor,
                "total": len(items),
            }

        scores = [0.0] * len(tools)
        for field, weight in FIELD_WEIGHTS:
            field_scores = _field_scores(tools, query_tokens, field)
            for index, score in enumerate(field_scores):
                scores[index] += weight * float(score)
        lowered_query = query.lower()
        ranked = []
        for score, tool in zip(scores, tools):
            name = tool["name"].lower()
            if score > 0:
                ranked.append((name != lowered_query, -score, tool["name"], tool))
        ranked.sort(key=lambda item: item[:3])
        items = [tool for *_, tool in ranked]
        next_cursor = cursor + limit if cursor + limit < len(items) else None
        return {
            "items": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "namespaces": tool["namespaces"],
                }
                for tool in items[cursor:cursor + limit]
            ],
            "next_cursor": next_cursor,
            "total": len(items),
        }
    '''
)

LIST_TOOLS_SOURCE = dedent(
    '''\
    """Page through active tools, optionally within one namespace subtree."""

    async def main(input, ctx):
        cursor = max(0, input.get("cursor", 0))
        limit = min(50, max(1, input.get("limit", 10)))
        tools = sorted(
            ctx.kernel.list_tools(namespace=input.get("namespace")),
            key=lambda tool: tool["name"],
        )
        next_cursor = cursor + limit if cursor + limit < len(tools) else None
        return {
            "items": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "namespaces": tool["namespaces"],
                }
                for tool in tools[cursor:cursor + limit]
            ],
            "next_cursor": next_cursor,
            "total": len(tools),
        }
    '''
)

LIST_TOOL_NAMESPACES_SOURCE = dedent(
    '''\
    """Page through hierarchical namespaces and their recursive tool counts."""

    async def main(input, ctx):
        cursor = max(0, input.get("cursor", 0))
        limit = min(50, max(1, input.get("limit", 10)))
        namespaces = ctx.kernel.list_tool_namespaces(
            namespace=input.get("namespace")
        )
        next_cursor = (
            cursor + limit if cursor + limit < len(namespaces) else None
        )
        return {
            "items": namespaces[cursor:cursor + limit],
            "next_cursor": next_cursor,
            "total": len(namespaces),
        }
    '''
)

VIEW_SOURCE = dedent(
    '''\
    """View the exact stored source and metadata for a tool version."""

    async def main(input, ctx):
        return ctx.kernel.view_tool(
            input["name"], version=input.get("version")
        )
    '''
)

WRITE_SOURCE = dedent(
    '''\
    """Create tool versions or replace hierarchical namespace assignments."""

    async def main(input, ctx):
        return ctx.kernel.write_tool(
            name=input["name"],
            description=input.get("description"),
            input_schema=input.get("input_schema"),
            source=input.get("source"),
            base_version=input.get("base_version"),
            namespaces=input.get("namespaces"),
        )
    '''
)

CALL_SOURCE = dedent(
    '''\
    """Resolve and execute an active or exact tool version."""

    async def main(input, ctx):
        return await ctx.kernel.execute(
            input["name"], input["args"], version=input.get("version")
        )
    '''
)

DELETE_SOURCE = dedent(
    '''\
    """Remove a tool binding or specific version while retaining lineage history."""

    async def main(input, ctx):
        return ctx.kernel.delete_tool(
            name=input["name"],
            version=input.get("version"),
        )
    '''
)

RUN_AGENT_SOURCE = dedent(
    '''\
    """Prompt a durable conversation through the generic session capability."""

    from mechagnome import ToolboxError

    async def main(input, ctx):
        keys = set(input)
        if "job_id" in input:
            if keys != {"job_id"}:
                raise ToolboxError(
                    "invalid_session_prompt",
                    "session prompt inspection requires only a non-empty job_id",
                )
            return await ctx.sessions.inspect(input["job_id"])
        if "prompt" not in input or not keys <= {
            "session_id", "prompt", "mode", "detach", "title", "description"
        }:
            raise ToolboxError(
                "invalid_session_prompt",
                "session prompting requires a prompt and optional session_id, "
                "mode, detach, title, and description",
            )
        session = ctx.sessions.get(input.get("session_id"))
        metadata = {
            name: input[name]
            for name in ("title", "description")
            if name in input
        }
        outcome = await session.prompt(
            input["prompt"],
            mode=input.get("mode", "spawn"),
            detach=input.get("detach", False),
            metadata=metadata or None,
        )
        return outcome if input.get("detach", False) else outcome["result"]
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
        "list_tools",
        "List active tools, optionally within a hierarchical namespace.",
        LIST_SCHEMA,
        LIST_TOOLS_SOURCE,
    ),
    BootstrapTool(
        "list_tool_namespaces",
        "List hierarchical tool namespaces and recursive tool counts.",
        LIST_SCHEMA,
        LIST_TOOL_NAMESPACES_SOURCE,
    ),
    BootstrapTool(
        "search_tools",
        "Search or browse active tools by metadata and hierarchical namespace.",
        SEARCH_SCHEMA,
        SEARCH_SOURCE,
    ),
    BootstrapTool(
        "view_tool",
        "View source, metadata, and schema for a tool version.",
        VIEW_SCHEMA,
        VIEW_SOURCE,
    ),
    BootstrapTool(
        "write_tool",
        "Create tool versions or replace hierarchical namespace assignments.",
        WRITE_SCHEMA,
        WRITE_SOURCE,
    ),
    BootstrapTool(
        "call_tool",
        "Invoke a tool, detach it for later inspection, or inspect a detached job.",
        CALL_SCHEMA,
        CALL_SOURCE,
    ),
    BootstrapTool(
        "delete_tool",
        "Delete a tool from the active toolbox, retaining its version history.",
        DELETE_SCHEMA,
        DELETE_SOURCE,
    ),
    BootstrapTool(
        "run_agent",
        "Continue, spawn, or fork a session; set its title and description; or "
        "inspect a detached prompt job.",
        RUN_AGENT_SCHEMA,
        RUN_AGENT_SOURCE,
    ),
)

CORE_NAMES = tuple(tool.name for tool in BOOTSTRAP_TOOLS)
CORE_SCHEMAS = {tool.name: tool.input_schema for tool in BOOTSTRAP_TOOLS}
