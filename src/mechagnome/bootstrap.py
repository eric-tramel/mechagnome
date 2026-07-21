"""Pinned schemas and editable bootstrap tool implementations for Mechagnome."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any


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
        "query": {
            "type": "string",
            "description": "Search query; omit when only recording feedback.",
        },
        "include_core": {"type": "boolean"},
        "cursor": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        "feedback": {
            "type": "object",
            "description": (
                "Create or replace this session's feedback for a tool version."
            ),
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Tool receiving feedback.",
                },
                "version": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Exact version; defaults to the active version.",
                },
                "rating": {
                    "type": "integer",
                    "enum": [-1, 0, 1],
                    "description": "-1 downvote, 0 comment-only, or 1 upvote.",
                },
                "comment": {
                    "type": "string",
                    "maxLength": 4000,
                    "description": "Optional qualitative feedback.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
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
        "namespaces": "runtime/namespaces.md",
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
    """BM25 search over the active tool catalog."""

    import json
    import math
    import re
    from collections import Counter


    FIELD_WEIGHTS = (
        ("name", 8.0),
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

    def _feedback_multiplier(tool):
        feedback = tool.get("feedback") or {}
        upvotes = int(feedback.get("upvotes", 0))
        downvotes = int(feedback.get("downvotes", 0))
        ratings = upvotes + downvotes
        if not ratings:
            return 1.0
        # Bayesian smoothing keeps one vote useful without letting sparse
        # feedback completely overwhelm lexical relevance.
        quality = (upvotes - downvotes) / (ratings + 2)
        return max(0.5, min(1.5, 1.0 + 0.5 * quality))


    def main(input, ctx):
        feedback_result = None
        feedback = input.get("feedback")
        if feedback is not None:
            feedback_result = ctx.kernel.submit_tool_feedback(
                feedback["name"],
                rating=feedback.get("rating", 0),
                comment=feedback.get("comment"),
                version=feedback.get("version"),
            )
        if "query" not in input:
            if feedback_result is None:
                return {"error": "query or feedback is required"}
            return {"feedback": feedback_result}

        query = input["query"].strip()
        include_core = input.get("include_core", True)
        cursor = max(0, input.get("cursor", 0))
        limit = min(50, max(1, input.get("limit", 10)))
        tools = ctx.kernel.catalog(include_core=include_core)
        query_tokens = _tokens(query)

        if not query_tokens:
            items = sorted(tools, key=lambda tool: tool["name"])
            next_cursor = cursor + limit if cursor + limit < len(items) else None
            result = {
                "items": items[cursor:cursor + limit],
                "next_cursor": next_cursor,
                "total": len(items),
            }
            if feedback_result is not None:
                result["feedback"] = feedback_result
            return result

        scores = [0.0] * len(tools)
        for field, weight in FIELD_WEIGHTS:
            field_scores = _field_scores(tools, query_tokens, field)
            for index, score in enumerate(field_scores):
                scores[index] += weight * float(score)
        for index, tool in enumerate(tools):
            scores[index] *= _feedback_multiplier(tool)

        lowered_query = query.lower()
        ranked = []
        for score, tool in zip(scores, tools):
            name = tool["name"].lower()
            if score > 0:
                ranked.append((name != lowered_query, -score, tool["name"], tool))
        ranked.sort(key=lambda item: item[:3])
        items = [tool for *_, tool in ranked]
        next_cursor = cursor + limit if cursor + limit < len(items) else None
        result = {
            "items": items[cursor:cursor + limit],
            "next_cursor": next_cursor,
            "total": len(items),
        }
        if feedback_result is not None:
            result["feedback"] = feedback_result
        return result
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
        "Search active tools and record version-specific ratings and feedback.",
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
