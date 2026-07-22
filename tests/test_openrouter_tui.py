"""Tests for the OpenRouter adapter, persistent chat, and default TUI."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
import time
from io import StringIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console
from rich.markup import render as render_markup
from rich.syntax import Syntax
from rich.text import Text
from textual.widgets import (
    Button,
    Input,
    OptionList,
    RichLog,
    Select,
    Static,
    TabbedContent,
)

from mechagnome import (
    Harness,
    Kernel,
    ModelProvider,
    ModelStreamEvent,
    ModelTurn,
    RunCancelled,
    ToolboxError,
    ToolCall,
)
from mechagnome import __main__ as cli
from mechagnome import openrouter as openrouter_module
from mechagnome.harness import AgentEvent
from mechagnome.isolation import IsolatedToolRunner
from mechagnome.openrouter import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OpenRouterError,
    OpenRouterModel,
    OpenRouterModelOption,
)
from mechagnome.tui import (
    ChatFeed,
    DeleteToolScreen,
    ModelSelectionScreen,
    NamespaceNameScreen,
    ReasoningEffortScreen,
    ToolboxApp,
    ToolEvent,
    ToolManagerScreen,
    _format_duration,
)


def chat_text(app: ToolboxApp, chat: ChatFeed | None = None) -> str:
    """Render mounted chat entries as plain text for headless assertions."""
    output = StringIO()
    console = Console(
        file=output,
        width=120,
        color_system=None,
        force_terminal=False,
    )
    for entry in (chat or app.chat).children:
        if isinstance(entry, ToolEvent):
            console.print(
                f"{ToolEvent.SYMBOLS[entry.kind]} {entry.tool_name}\n{entry.detail}"
            )
        elif isinstance(entry, Static):
            console.print(entry.content)
    return output.getvalue().rstrip()


def tool_title_text(event: ToolEvent) -> str:
    """Render a tool event's public Rich-markup title as plain text."""
    return render_markup(event.title).plain


class FinalModel:
    """Deterministic multi-prompt model for conversation and TUI tests."""

    def __init__(self) -> None:
        self.message_snapshots: list[list[dict[str, Any]]] = []

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.message_snapshots.append([dict(message) for message in messages])
        return ModelTurn(text=f"answer {len(self.message_snapshots)}")


class StaticProvider:
    """Simple cooperative provider for isolated-runner tests."""

    def __init__(self, result: str = "provided") -> None:
        self.result = result
        self.calls = 0

    def complete(self, messages: Any) -> str:
        self.calls += 1
        return self.result

    def cancel_current(self) -> None:
        pass

    def reset_cancellation(self) -> None:
        pass


def sse_response(
    *payloads: dict[str, Any],
    finish_reason: str = "stop",
    post_terminal: tuple[dict[str, Any], ...] = (),
    done: bool = True,
) -> httpx.Response:
    """Build a deterministic OpenRouter-style event stream."""
    content = "".join(f"data: {json.dumps(payload)}\n\n" for payload in payloads)
    content += (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {"delta": {}, "finish_reason": finish_reason},
                ]
            }
        )
        + "\n\n"
    )
    content += "".join(f"data: {json.dumps(payload)}\n\n" for payload in post_terminal)
    if done:
        content += "data: [DONE]\n\n"
    return httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream"},
        content=content,
    )


def test_conversation_keeps_model_context_and_one_durable_session(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    model = FinalModel()
    conversation = Harness(kernel).start(model)

    first = conversation.send("first")
    second = conversation.send("second")

    assert first.session_id == second.session_id == conversation.session_id
    assert model.message_snapshots[0] == [{"role": "user", "content": "first"}]
    assert model.message_snapshots[1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer 1", "tool_calls": []},
        {"role": "user", "content": "second"},
    ]
    kinds = [
        event["kind"]
        for event in kernel.read_session(conversation.session_id, limit=100)["events"]
    ]
    assert kinds == ["user", "model", "final", "user", "model", "final"]


def test_conversation_rehydrates_an_existing_saved_session(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    first_model = FinalModel()
    first = Harness(kernel).start(first_model)
    first.send("first")

    second_model = FinalModel()
    resumed = Harness(kernel).start(second_model, session_id=first.session_id)
    resumed.send("second")

    assert second_model.message_snapshots[0] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer 1", "tool_calls": []},
        {"role": "user", "content": "second"},
    ]


def test_openrouter_adapter_uses_glm_defaults_and_translates_tool_calls(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["title"] = request.headers["X-OpenRouter-Title"]
        captured["body"] = json.loads(request.content)
        return sse_response(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tool-1",
                                    "function": {
                                        "name": "help",
                                        "arguments": '{"topic":',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"quickstart"}'},
                                }
                            ]
                        }
                    }
                ]
            },
            finish_reason="tool_calls",
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenRouterModel(api_key="test-key", client=client)

    turn = model.respond(
        [{"role": "user", "content": "How do I begin?"}],
        kernel.tool_definitions(),
    )

    assert model.model == DEFAULT_MODEL
    assert captured["url"] == f"{DEFAULT_BASE_URL}/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["title"] == "mechagnome"
    assert captured["body"]["model"] == "z-ai/glm-5.2"
    assert captured["body"]["parallel_tool_calls"] is True
    assert captured["body"]["stream"] is True
    assert captured["body"]["messages"][0]["role"] == "system"
    tools = {
        tool["function"]["name"]: tool["function"] for tool in captured["body"]["tools"]
    }
    assert list(tools) == [
        "help",
        "search_tools",
        "view_tool",
        "write_tool",
        "call_tool",
    ]
    write_schema = tools["write_tool"]["parameters"]["properties"]["input_schema"]
    call_schema = tools["call_tool"]["parameters"]["properties"]["args"]
    assert write_schema["type"] == "string"
    assert "JSON-encoded JSON Schema object" in write_schema["description"]
    assert call_schema["type"] == "string"
    assert "JSON-encoded object" in call_schema["description"]
    assert turn.calls[0].name == "help"
    assert turn.calls[0].args == {"topic": "quickstart"}


def test_openrouter_completion_is_text_only_and_omits_agent_surface() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Nested."},
                    }
                ]
            },
        )

    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Answer."},
    ]

    assert model.complete(messages) == "Nested."
    assert captured["authorization"] == "Bearer test-key"
    assert captured["body"] == {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": 2048,
        "stream": False,
    }


def test_openrouter_completion_rejects_tool_calls_and_invalid_content() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {"content": None, "tool_calls": [{}]},
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": []}}]
                },
            ),
        ]
    )
    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: next(responses))
        ),
    )

    for _ in range(2):
        with pytest.raises(OpenRouterError, match="invalid completion"):
            model.complete([{"role": "user", "content": "hello"}])


def test_openrouter_completion_reuses_borrowed_client_after_cancellation() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "reused"},
                        }
                    ]
                },
            )
        )
    )
    model = OpenRouterModel(api_key="test-key", client=client)

    model.cancel_current()
    with pytest.raises(OpenRouterError) as cancelled:
        model.complete([{"role": "user", "content": "first"}])
    assert cancelled.value.code == "openrouter_cancelled"

    model.reset_cancellation()
    assert model.complete([{"role": "user", "content": "second"}]) == "reused"
    assert client.is_closed is False


def test_openrouter_adapter_lists_tool_models_and_reasoning_support() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.params["supported_parameters"] == "tools"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "example/reasoner",
                        "name": "Example Reasoner",
                        "architecture": {
                            "input_modalities": ["text", "image", "audio"]
                        },
                        "context_length": 65536,
                        "supported_parameters": ["tools", "reasoning"],
                        "reasoning": {
                            "supported_efforts": ["high", "low", "none"],
                            "mandatory": False,
                        },
                    },
                    {
                        "id": "example/standard",
                        "name": "Example Standard",
                        "supported_parameters": ["tools"],
                    },
                    {
                        "id": "example/mandatory-reasoner",
                        "name": "Mandatory Reasoner",
                        "supported_parameters": ["tools", "reasoning"],
                        "reasoning": {
                            "supported_efforts": None,
                            "mandatory": True,
                        },
                    },
                    {
                        "id": "example/no-tools",
                        "name": "No Tools",
                        "supported_parameters": ["reasoning"],
                    },
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenRouterModel(api_key="test-key", client=client)

    options = model.available_models()

    assert [
        (
            option.id,
            option.name,
            option.input_modalities,
            option.reasoning_efforts,
            option.reasoning_mandatory,
            option.context_length,
        )
        for option in options
    ] == [
        (
            "example/reasoner",
            "Example Reasoner",
            ("text", "image", "audio"),
            ("high", "low", "none"),
            False,
            65536,
        ),
        ("example/standard", "Example Standard", (), (), False, None),
        (
            "example/mandatory-reasoner",
            "Mandatory Reasoner",
            (),
            ("minimal", "low", "medium", "high", "xhigh", "max"),
            True,
            None,
        ),
    ]


@pytest.mark.parametrize(
    "context_length",
    [None, 0, -1, True, 1.5, "65536"],
)
def test_openrouter_catalog_ignores_invalid_context_lengths(
    context_length: Any,
) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "example/model",
                            "name": "Example",
                            "context_length": context_length,
                            "supported_parameters": ["tools"],
                        }
                    ]
                },
            )
        )
    )

    [option] = OpenRouterModel(api_key="test-key", client=client).available_models()

    assert option.context_length is None


def test_openrouter_adapter_sends_configured_reasoning_effort(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return sse_response({"choices": [{"delta": {"content": "Ready."}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenRouterModel(
        api_key="test-key",
        client=client,
    )
    model.reasoning_effort = "high"

    model.respond(
        [{"role": "user", "content": "hello"}],
        Kernel(tmp_path / "toolbox.db").tool_definitions(),
    )

    assert captured["body"]["reasoning"] == {"effort": "high"}


def test_openrouter_adapter_serializes_prior_tool_results(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assistant = body["messages"][-2]
        observation = body["messages"][-1]
        assert assistant["tool_calls"][0]["function"] == {
            "name": "help",
            "arguments": '{"topic": "quickstart"}',
        }
        assert observation == {
            "role": "tool",
            "tool_call_id": "tool-1",
            "name": "help",
            "content": '{"topic": "quickstart"}',
        }
        return sse_response(
            {"choices": [{"delta": {"content": "Rea"}}]},
            {"choices": [{"delta": {"content": "dy."}}]},
        )

    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    messages = [
        {"role": "user", "content": "begin"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tool-1", "name": "help", "args": {"topic": "quickstart"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tool-1",
            "name": "help",
            "content": {"topic": "quickstart"},
        },
    ]

    assert model.respond(messages, kernel.tool_definitions()).text == "Ready."


def test_openrouter_preserves_reasoning_across_tool_continuation(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    detail = {
        "type": "reasoning.encrypted",
        "data": "opaque-provider-payload",
        "id": "reasoning-1",
        "format": "anthropic-claude-v1",
        "index": 0,
    }
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return sse_response(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning": "I should inspect the help topic.",
                                "reasoning_details": [detail],
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "reason-call",
                                        "function": {
                                            "name": "help",
                                            "arguments": '{"topic":"quickstart"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
                finish_reason="tool_calls",
            )
        assistant = body["messages"][-2]
        assert "reasoning" not in assistant
        assert assistant["reasoning_details"] == [detail]
        assert body["reasoning"] == {"effort": "high"}
        return sse_response({"choices": [{"delta": {"content": "Ready with help."}}]})

    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    model.reasoning_effort = "high"

    result = Harness(kernel).start(model).send("How do I begin?")

    assert result.answer == "Ready with help."
    first_model_event = next(
        event
        for event in kernel.read_session(result.session_id, limit=100)["events"]
        if event["kind"] == "model"
    )
    assert first_model_event["payload"]["reasoning_details"] == [detail]


def test_openrouter_compacts_plaintext_reasoning_details(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return sse_response(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning": "Inspect",
                                "reasoning_details": [
                                    {
                                        "type": "reasoning.text",
                                        "text": "Inspect",
                                        "format": "unknown",
                                        "index": 0,
                                    }
                                ],
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning": " help.",
                                "reasoning_details": [
                                    {
                                        "type": "reasoning.text",
                                        "text": " help.",
                                        "format": "unknown",
                                        "index": 0,
                                    }
                                ],
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "reason-call",
                                        "function": {
                                            "name": "help",
                                            "arguments": '{"topic":"quickstart"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
                finish_reason="tool_calls",
            )
        assistant = body["messages"][-2]
        assert assistant["reasoning"] == "Inspect help."
        assert "reasoning_details" not in assistant
        return sse_response({"choices": [{"delta": {"content": "Ready."}}]})

    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = Harness(kernel).start(model).send("How do I begin?")

    assert result.answer == "Ready."
    first_model_event = next(
        event
        for event in kernel.read_session(result.session_id, limit=100)["events"]
        if event["kind"] == "model"
    )
    assert first_model_event["payload"]["reasoning"] == "Inspect help."
    assert "reasoning_details" not in first_model_event["payload"]

    legacy_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [],
        "reasoning": "Inspect help.",
        "reasoning_details": [
            {
                "type": "reasoning.text",
                "text": "Inspect help.",
                "format": "unknown",
                "index": 0,
            }
        ],
    }
    assert OpenRouterModel._wire_messages([legacy_message]) == [
        {
            "role": "assistant",
            "content": None,
            "reasoning": "Inspect help.",
        }
    ]

    legacy_message["reasoning"] = "A different summary."
    assert OpenRouterModel._wire_messages([legacy_message]) == [
        {
            "role": "assistant",
            "content": None,
            "reasoning_details": legacy_message["reasoning_details"],
        }
    ]


def test_openrouter_adapter_round_trips_objects_and_keeps_parallel_calls(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prior_call = body["messages"][-1]["tool_calls"][0]["function"]
        prior_args = json.loads(prior_call["arguments"])
        assert json.loads(prior_args["input_schema"]) == input_schema
        return sse_response(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tool-2",
                                    "function": {
                                        "name": "call_tool",
                                        "arguments": json.dumps(
                                            {
                                                "name": "search",
                                                "args": json.dumps({"query": "gnomes"}),
                                            }
                                        ),
                                    },
                                },
                                {
                                    "index": 1,
                                    "id": "tool-3",
                                    "function": {
                                        "name": "help",
                                        "arguments": '{"topic":"composition"}',
                                    },
                                },
                            ]
                        }
                    }
                ]
            },
            finish_reason="tool_calls",
        )

    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool-1",
                    "name": "write_tool",
                    "args": {
                        "name": "search",
                        "description": "Search for gnomes.",
                        "input_schema": input_schema,
                        "source": "def main(input, ctx):\n    return input\n",
                    },
                }
            ],
        }
    ]

    turn = model.respond(messages, kernel.tool_definitions())

    assert turn.calls == (
        ToolCall(
            name="call_tool",
            args={"name": "search", "args": {"query": "gnomes"}},
            id="tool-2",
        ),
        ToolCall(
            name="help",
            args={"topic": "composition"},
            id="tool-3",
        ),
    )


def test_openrouter_adapter_preserves_malformed_json_for_tool_repair() -> None:
    call = OpenRouterModel._tool_call(
        {
            "id": "tool-1",
            "function": {
                "name": "write_tool",
                "arguments": '{"input_schema":""}',
            },
        }
    )

    assert call.args == {"input_schema": ""}


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"delta": []}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": "not-an-object"}],
                    }
                }
            ]
        },
    ],
)
def test_openrouter_adapter_normalizes_wrong_json_shapes(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: sse_response(payload))
    )
    model = OpenRouterModel(api_key="test-key", client=client)

    with pytest.raises(OpenRouterError):
        model.respond([{"role": "user", "content": "hello"}], kernel.tool_definitions())


def test_openrouter_adapter_normalizes_non_object_http_error(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(400, json=["bad request"])
        )
    )
    model = OpenRouterModel(api_key="test-key", client=client)

    with pytest.raises(OpenRouterError, match="HTTP 400"):
        model.respond([{"role": "user", "content": "hello"}], kernel.tool_definitions())


def test_conversation_records_openrouter_transport_failure(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    secret = "sentinel-provider-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(secret, request=request)

    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    conversation = Harness(kernel).start(model)

    with pytest.raises(OpenRouterError, match=secret):
        conversation.send("hello")

    events = kernel.read_session(conversation.session_id, limit=100)["events"]
    assert [event["kind"] for event in events] == ["user", "model_failed"]
    assert events[-1]["payload"] == {
        "code": "openrouter_transport",
        "message": "OpenRouter transport failed",
        "details": {},
    }
    assert secret not in str(events[-1]["payload"])


@pytest.mark.parametrize(
    ("response", "code", "message"),
    [
        (
            httpx.Response(
                400,
                json={"error": {"message": "sentinel-provider-secret"}},
            ),
            "openrouter_http",
            "OpenRouter returned an HTTP error",
        ),
        (
            httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=(
                    'data: {"error":{"message":"sentinel-provider-secret"}}\n\n'
                    "data: [DONE]\n\n"
                ),
            ),
            "openrouter_stream",
            "OpenRouter stream failed",
        ),
    ],
)
def test_conversation_sanitizes_provider_controlled_openrouter_errors(
    tmp_path: Path,
    response: httpx.Response,
    code: str,
    message: str,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    secret = "sentinel-provider-secret"
    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(lambda request: response)),
    )
    conversation = Harness(kernel).start(model)

    with pytest.raises(OpenRouterError, match=secret):
        conversation.send("hello")

    events = kernel.read_session(conversation.session_id, limit=100)["events"]
    assert events[-1]["payload"] == {
        "code": code,
        "message": message,
        "details": {},
    }
    assert secret not in str(events[-1]["payload"])


def test_completion_session_records_safe_openrouter_failure(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    secret = "sentinel-provider-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(secret, request=request)

    completion_model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    provider = ModelProvider(
        kernel,
        FinalModel(),
        completion_transport=completion_model,
    )
    session = provider.start_session()

    with pytest.raises(ToolboxError) as failure:
        session.completion_provider().complete([{"role": "user", "content": "hello"}])

    assert failure.value.code == "model_provider_failed"
    children = [
        child
        for child in kernel.list_sessions(limit=100)["sessions"]
        if child["parent_session_id"] == session.session_id
    ]
    assert len(children) == 1
    events = kernel.read_session(children[0]["id"], limit=100)["events"]
    assert events[-1]["payload"] == {
        "code": "openrouter_transport",
        "message": "OpenRouter transport failed",
        "details": {},
    }
    assert secret not in str(events[-1]["payload"])


def test_completion_session_sanitizes_unknown_failure(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    secret = "sentinel-provider-secret"

    class UnsafeCompletion:
        def complete(self, messages: Any) -> str:
            raise RuntimeError(secret)

        def cancel_current(self) -> None:
            pass

        def reset_cancellation(self) -> None:
            pass

    provider = ModelProvider(
        kernel,
        FinalModel(),
        completion_transport=UnsafeCompletion(),
    )
    session = provider.start_session()

    with pytest.raises(ToolboxError) as failure:
        session.completion_provider().complete([{"role": "user", "content": "hello"}])

    assert failure.value.code == "model_provider_failed"
    children = [
        child
        for child in kernel.list_sessions(limit=100)["sessions"]
        if child["parent_session_id"] == session.session_id
    ]
    events = kernel.read_session(children[0]["id"], limit=100)["events"]
    assert events[-1]["payload"] == {
        "code": "model_provider_failed",
        "message": "model provider request failed",
        "details": {},
    }
    assert secret not in str(events[-1]["payload"])


def test_conversation_still_sanitizes_unknown_model_failures(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    secret = "sentinel-provider-secret"

    class UnsafeModel:
        def respond(self, messages: Any, tools: Any) -> ModelTurn:
            raise ToolboxError("unsafe_provider_error", secret)

    conversation = Harness(kernel).start(UnsafeModel())

    with pytest.raises(ToolboxError, match=secret):
        conversation.send("hello")

    events = kernel.read_session(conversation.session_id, limit=100)["events"]
    assert events[-1]["payload"] == {
        "code": "model_provider_failed",
        "message": "model provider request failed",
        "details": {},
    }


def test_openrouter_adapter_rejects_truncated_stream(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: sse_response(
                {"choices": [{"delta": {"content": "partial"}}]},
                done=False,
            )
        )
    )
    model = OpenRouterModel(api_key="test-key", client=client)

    with pytest.raises(OpenRouterError, match="before \\[DONE\\]"):
        model.respond([{"role": "user", "content": "hello"}], kernel.tool_definitions())


def test_openrouter_adapter_rejects_non_success_finish_reason(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: sse_response(
                {"choices": [{"delta": {"content": "partial"}}]},
                finish_reason="length",
            )
        )
    )
    model = OpenRouterModel(api_key="test-key", client=client)

    with pytest.raises(OpenRouterError, match="length"):
        model.respond([{"role": "user", "content": "hello"}], kernel.tool_definitions())


def test_openrouter_adapter_allows_usage_metadata_after_finish(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: sse_response(
                {"choices": [{"delta": {"content": "Ready."}}]},
                post_terminal=(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                        },
                    },
                ),
            )
        )
    )
    model = OpenRouterModel(api_key="test-key", client=client)

    turn = model.respond(
        [{"role": "user", "content": "hello"}], kernel.tool_definitions()
    )

    assert turn.text == "Ready."
    assert turn.total_tokens == 12

    conversation = Harness(kernel).start(model)
    conversation.send("hello")
    model_event = next(
        event
        for event in kernel.read_session(conversation.session_id, limit=100)["events"]
        if event["kind"] == "model"
    )
    assert model_event["payload"]["total_tokens"] == 12
    assert "total_tokens" not in conversation.messages[-1]


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"total_tokens": 0},
        {"total_tokens": -1},
        {"total_tokens": True},
        {"total_tokens": 1.5},
        {"total_tokens": "12"},
        "invalid",
    ],
)
def test_openrouter_adapter_ignores_invalid_usage_metadata(
    tmp_path: Path, usage: Any
) -> None:
    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: sse_response(
                    {"choices": [{"delta": {"content": "Ready."}}]},
                    post_terminal=({"choices": [], "usage": usage},),
                )
            )
        ),
    )

    turn = model.respond(
        [{"role": "user", "content": "hello"}],
        Kernel(tmp_path / "toolbox.db").tool_definitions(),
    )

    assert turn.text == "Ready."
    assert turn.total_tokens is None


def test_openrouter_adapter_rejects_model_data_after_finish(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: sse_response(
                {"choices": [{"delta": {"content": "Ready."}}]},
                post_terminal=(
                    {
                        "choices": [
                            {"delta": {"content": "late"}, "finish_reason": None}
                        ]
                    },
                ),
            )
        )
    )
    model = OpenRouterModel(api_key="test-key", client=client)

    with pytest.raises(OpenRouterError, match="model data after"):
        model.respond([{"role": "user", "content": "hello"}], kernel.tool_definitions())


@pytest.mark.parametrize("repeated_reason", [None, "stop"])
def test_openrouter_adapter_allows_empty_terminal_metadata(
    tmp_path: Path, repeated_reason: str | None
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: sse_response(
                {"choices": [{"delta": {"content": "Ready."}}]},
                post_terminal=(
                    {"choices": [{"delta": {}, "finish_reason": repeated_reason}]},
                ),
            )
        )
    )
    model = OpenRouterModel(api_key="test-key", client=client)

    assert (
        model.respond(
            [{"role": "user", "content": "hello"}], kernel.tool_definitions()
        ).text
        == "Ready."
    )


@pytest.mark.parametrize(
    ("late_delta", "message"),
    [
        (
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "late",
                        "function": {"name": "help", "arguments": "{}"},
                    }
                ]
            },
            "model data after",
        ),
        ({"tool_calls": {}}, "invalid tool-call deltas"),
    ],
)
def test_openrouter_adapter_rejects_late_tool_call_shapes(
    tmp_path: Path, late_delta: dict[str, Any], message: str
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: sse_response(
                {"choices": [{"delta": {"content": "Ready."}}]},
                post_terminal=(
                    {"choices": [{"delta": late_delta, "finish_reason": None}]},
                ),
            )
        )
    )
    model = OpenRouterModel(api_key="test-key", client=client)

    with pytest.raises(OpenRouterError, match=message):
        model.respond([{"role": "user", "content": "hello"}], kernel.tool_definitions())


def test_openrouter_adapter_rejects_changed_terminal_reason(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: sse_response(
                {"choices": [{"delta": {"content": "Ready."}}]},
                post_terminal=(
                    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
                ),
            )
        )
    )
    model = OpenRouterModel(api_key="test-key", client=client)

    with pytest.raises(OpenRouterError, match="changed the terminal"):
        model.respond([{"role": "user", "content": "hello"}], kernel.tool_definitions())


def test_openrouter_adapter_bounds_stream_size(
    tmp_path: Path, monkeypatch: Any
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    monkeypatch.setattr(openrouter_module, "MAX_STREAM_BYTES", 10)
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: sse_response(
                {"choices": [{"delta": {"content": "too large"}}]},
            )
        )
    )
    model = OpenRouterModel(api_key="test-key", client=client)

    with pytest.raises(OpenRouterError, match="size limit"):
        model.respond([{"role": "user", "content": "hello"}], kernel.tool_definitions())


class PausingSSEStream(httpx.SyncByteStream):
    """Hold a provider response open until another thread closes it."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def __iter__(self) -> Any:
        yield b'data: {"choices":[{"delta":{"content":"Partial"}}]}\n\n'
        self.started.set()
        self.release.wait(timeout=3)
        if not self.closed:
            yield (
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            )

    def close(self) -> None:
        self.closed = True
        self.release.set()


def test_openrouter_latches_cancellation_until_stream_registration(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    requests = 0

    def response(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return sse_response({"choices": [{"delta": {"content": "too late"}}]})

    client = httpx.Client(transport=httpx.MockTransport(response))
    model = OpenRouterModel(api_key="test-key", client=client)

    model.cancel_current()

    with pytest.raises(OpenRouterError):
        model.respond(
            [{"role": "user", "content": "do not start"}],
            kernel.tool_definitions(),
        )
    assert requests == 0


def test_conversation_cancellation_closes_active_openrouter_stream(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    stream = PausingSSEStream()
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                stream=stream,
            )
        )
    )
    model = OpenRouterModel(api_key="test-key", client=client)
    conversation = Harness(kernel).start(model)
    failures: list[BaseException] = []

    def run() -> None:
        try:
            conversation.send("stream forever")
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert stream.started.wait(timeout=1)
    assert conversation.cancel() is True
    thread.join(timeout=2)

    assert thread.is_alive() is False
    assert stream.closed is True
    assert len(failures) == 1
    assert isinstance(failures[0], RunCancelled)
    events = kernel.read_session(conversation.session_id, limit=100)["events"]
    assert [event["kind"] for event in events] == ["user", "cancelled"]


def test_openrouter_cancellation_is_isolated_between_root_sessions(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    streams = {
        "first": PausingSSEStream(),
        "second": PausingSSEStream(),
    }

    def response(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=streams[prompt],
        )

    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(response)),
    )
    provider = ModelProvider(kernel, model)
    conversations = {
        name: Harness(kernel).start(provider) for name in ("first", "second")
    }
    results: dict[str, str] = {}
    failures: dict[str, BaseException] = {}

    def run(name: str) -> None:
        try:
            results[name] = conversations[name].send(name).answer
        except BaseException as error:
            failures[name] = error

    threads = {
        name: threading.Thread(target=run, args=(name,)) for name in conversations
    }
    for thread in threads.values():
        thread.start()
    assert streams["first"].started.wait(timeout=1)
    assert streams["second"].started.wait(timeout=1)

    assert conversations["first"].cancel() is True
    threads["first"].join(timeout=2)
    assert threads["first"].is_alive() is False
    assert streams["first"].closed is True
    assert streams["second"].closed is False
    assert threads["second"].is_alive() is True

    streams["second"].release.set()
    threads["second"].join(timeout=2)

    assert threads["second"].is_alive() is False
    assert isinstance(failures["first"], RunCancelled)
    assert "second" not in failures
    assert results["second"] == "Partial"


def test_openrouter_pre_registration_cancellation_is_root_local() -> None:
    requests: list[str] = []

    def response(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        requests.append(prompt)
        return sse_response({"choices": [{"delta": {"content": prompt}}]})

    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(response)),
    )
    first = model.for_session("first-root")
    second = model.for_session("second-root")

    first.cancel_current()

    assert second.respond([{"role": "user", "content": "second"}], []).text == (
        "second"
    )
    with pytest.raises(OpenRouterError) as cancelled:
        first.respond([{"role": "user", "content": "first"}], [])
    assert cancelled.value.code == "openrouter_cancelled"
    assert requests == ["second"]

    first.reset_cancellation()


def test_closed_conversation_rejects_a_late_rollout(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    model = FinalModel()
    conversation = Harness(kernel).start(model)

    conversation.close()

    with pytest.raises(RunCancelled):
        conversation.send("too late")
    assert model.message_snapshots == []
    assert kernel.read_session(conversation.session_id, limit=100)["events"] == []


class PausingResetModel(FinalModel):
    """Expose rollout teardown so cancellation can race it deterministically."""

    def __init__(self) -> None:
        super().__init__()
        self.reset_started = threading.Event()
        self.release_reset = threading.Event()
        self.cancel_calls = 0

    def reset_cancellation(self) -> None:
        self.reset_started.set()
        self.release_reset.wait(timeout=2)

    def cancel_current(self) -> None:
        self.cancel_calls += 1


def test_conversation_cleanup_cannot_latch_cancellation_for_next_send(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    model = PausingResetModel()
    conversation = Harness(kernel).start(model)
    failures: list[BaseException] = []
    cancel_results: list[bool] = []
    cancel_started = threading.Event()

    def send() -> None:
        try:
            conversation.send("first")
        except BaseException as error:
            failures.append(error)

    send_thread = threading.Thread(target=send)
    send_thread.start()
    assert model.reset_started.wait(timeout=1)

    def cancel() -> None:
        cancel_started.set()
        cancel_results.append(conversation.cancel())

    cancel_thread = threading.Thread(target=cancel)
    cancel_thread.start()
    assert cancel_started.wait(timeout=1)
    cancel_thread.join(timeout=0.05)
    assert cancel_thread.is_alive() is True
    model.release_reset.set()
    send_thread.join(timeout=2)
    cancel_thread.join(timeout=2)

    assert failures == []
    assert cancel_results == [False]
    assert model.cancel_calls == 0
    assert conversation.send("second").answer == "answer 2"


def test_authored_tools_do_not_inherit_provider_credentials(
    tmp_path: Path, monkeypatch: Any
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    kernel.call(
        "write_tool",
        {
            "name": "read_environment",
            "description": "Report whether provider credentials are inherited.",
            "input_schema": {"type": "object"},
            "source": (
                "import os\n\n"
                "def main(input, ctx):\n"
                "    return {'key': os.environ.get('OPENROUTER_API_KEY')}\n"
            ),
        },
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel-provider-key")
    runner = IsolatedToolRunner(kernel)
    session_id = kernel.create_session()

    result = runner.call(
        "call_tool",
        {"name": "read_environment", "args": {}},
        session_id=session_id,
    )

    assert result == {"key": None}


def test_authored_tools_inherit_git_and_ssh_environment(
    tmp_path: Path, monkeypatch: Any
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    names = (
        "GIT_ASKPASS",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
        "HOME",
        "SSH_ASKPASS",
        "SSH_AUTH_SOCK",
        "XDG_CONFIG_HOME",
        "GIT_TOKEN",
        "SSH_PRIVATE_KEY",
        "UNRELATED_SECRET",
    )
    kernel.call(
        "write_tool",
        {
            "name": "read_git_environment",
            "description": "Report Git and SSH environment variables.",
            "input_schema": {"type": "object"},
            "source": (
                "import os\n\n"
                "def main(input, ctx):\n"
                f"    names = {names!r}\n"
                "    return {name: os.environ.get(name) for name in names}\n"
            ),
        },
    )
    expected = {
        "GIT_ASKPASS": "/tmp/git-askpass",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "credential.helper",
        "GIT_CONFIG_VALUE_0": "test-helper",
        "HOME": "/tmp/git-home",
        "SSH_ASKPASS": "/tmp/ssh-askpass",
        "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
        "XDG_CONFIG_HOME": "/tmp/git-config",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("GIT_TOKEN", "sentinel-git-token")
    monkeypatch.setenv("SSH_PRIVATE_KEY", "sentinel-private-key")
    monkeypatch.setenv("UNRELATED_SECRET", "sentinel-secret")

    result = IsolatedToolRunner(kernel).call(
        "call_tool",
        {"name": "read_git_environment", "args": {}},
        session_id=kernel.create_session(),
    )

    assert result == {
        **expected,
        "GIT_TOKEN": None,
        "SSH_PRIVATE_KEY": None,
        "UNRELATED_SECRET": None,
    }


def test_isolated_model_provider_is_predictably_unavailable(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    kernel.call(
        "write_tool",
        {
            "name": "ask",
            "description": "Request a completion.",
            "input_schema": {"type": "object"},
            "source": (
                "def main(input, ctx):\n"
                "    return ctx.model_provider.complete([\n"
                "        {'role': 'user', 'content': 'hello'},\n"
                "    ])\n"
            ),
        },
    )

    with pytest.raises(ToolboxError) as error:
        IsolatedToolRunner(kernel).call(
            "call_tool",
            {"name": "ask", "args": {}},
            session_id=kernel.create_session(),
        )

    assert error.value.code == "model_provider_unavailable"


def test_isolated_tool_uses_host_authenticated_model_provider(
    tmp_path: Path, monkeypatch: Any
) -> None:
    secret = "sentinel-provider-key"
    host_pid = os.getpid()
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["pid"] = os.getpid()
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "brokered"},
                    }
                ]
            },
        )

    model = OpenRouterModel(
        api_key=secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    kernel = Kernel(tmp_path / "toolbox.db")
    kernel.call(
        "write_tool",
        {
            "name": "brokered_completion",
            "description": "Use the host model provider.",
            "input_schema": {"type": "object"},
            "source": (
                "import os\n"
                "import sys\n"
                "from pathlib import Path\n\n"
                "def main(input, ctx):\n"
                "    request_text = Path(sys.argv[1]).read_text()\n"
                "    return {\n"
                "        'text': ctx.model_provider.complete([\n"
                "            {'role': 'user', 'content': 'nested'},\n"
                "        ]),\n"
                "        'worker_pid': os.getpid(),\n"
                "        'environment_key': os.environ.get('OPENROUTER_API_KEY'),\n"
                "        'worker_request': request_text,\n"
                "    }\n"
            ),
        },
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    session_id = kernel.create_session()

    result = IsolatedToolRunner(kernel).call_with_model_provider(
        "call_tool",
        {"name": "brokered_completion", "args": {}},
        session_id=session_id,
        model_provider=model,
    )

    assert result["text"] == "brokered"
    assert result["worker_pid"] != host_pid
    assert result["environment_key"] is None
    request = json.loads(result["worker_request"])
    assert set(request) == {
        "db_path",
        "max_depth",
        "max_calls",
        "name",
        "args",
        "session_id",
        "toolbox_ids",
        "cwd",
        "model_provider_fd",
    }
    assert isinstance(request["model_provider_fd"], int)
    assert secret not in result["worker_request"]
    assert captured["pid"] == host_pid
    assert captured["authorization"] == f"Bearer {secret}"
    assert captured["body"] == {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "nested"}],
        "max_tokens": 2048,
        "stream": False,
    }
    events = kernel.read_session(session_id, limit=100)["events"]
    assert secret not in json.dumps(events)
    children = [
        item
        for item in kernel.list_sessions(limit=100)["sessions"]
        if item["parent_session_id"] == session_id
    ]
    assert len(children) == 1
    assert children[0]["kind"] == "completion"
    assert children[0]["origin_call_id"] is not None
    child_events = kernel.read_session(children[0]["id"], limit=100)["events"]
    assert [event["kind"] for event in child_events] == [
        "model_input",
        "model",
        "final",
    ]
    assert secret not in json.dumps(child_events)
    for database_file in tmp_path.glob("toolbox.db*"):
        assert secret.encode() not in database_file.read_bytes()


def test_isolated_provider_budget_counts_invalid_attempts(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    kernel.call(
        "write_tool",
        {
            "name": "budget",
            "description": "Exercise the model provider budget.",
            "input_schema": {"type": "object"},
            "source": (
                "def main(input, ctx):\n"
                "    codes = []\n"
                "    try:\n"
                "        ctx.model_provider.complete([])\n"
                "    except Exception as error:\n"
                "        codes.append(error.code)\n"
                "    for _ in range(7):\n"
                "        ctx.model_provider.complete([\n"
                "            {'role': 'user', 'content': 'valid'},\n"
                "        ])\n"
                "    try:\n"
                "        ctx.model_provider.complete([\n"
                "            {'role': 'user', 'content': 'exhausted'},\n"
                "        ])\n"
                "    except Exception as error:\n"
                "        codes.append(error.code)\n"
                "    return codes\n"
            ),
        },
    )
    provider = StaticProvider()

    result = IsolatedToolRunner(kernel).call_with_model_provider(
        "call_tool",
        {"name": "budget", "args": {}},
        session_id=kernel.create_session(),
        model_provider=provider,
    )

    assert result == ["invalid_model_request", "model_provider_limit"]
    assert provider.calls == 7


def test_tool_timeout_cancels_and_resets_cooperative_model_provider(
    tmp_path: Path,
) -> None:
    class BlockingProvider:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.exited = threading.Event()
            self.cancel_calls = 0
            self.reset_calls = 0

        def complete(self, messages: Any) -> str:
            self.started.set()
            self.release.wait(timeout=2)
            self.exited.set()
            raise RuntimeError("cancelled upstream")

        def cancel_current(self) -> None:
            self.cancel_calls += 1
            self.release.set()

        def reset_cancellation(self) -> None:
            self.reset_calls += 1

    kernel = Kernel(tmp_path / "toolbox.db")
    kernel.call(
        "write_tool",
        {
            "name": "blocked_completion",
            "description": "Wait for a model completion.",
            "input_schema": {"type": "object"},
            "source": (
                "def main(input, ctx):\n"
                "    return ctx.model_provider.complete([\n"
                "        {'role': 'user', 'content': 'wait'},\n"
                "    ])\n"
            ),
        },
    )
    provider = BlockingProvider()
    runner = IsolatedToolRunner(kernel, timeout=0.2)

    with pytest.raises(ToolboxError) as error:
        runner.call_with_model_provider(
            "call_tool",
            {"name": "blocked_completion", "args": {}},
            session_id=kernel.create_session(),
            model_provider=provider,
        )

    assert error.value.code == "tool_timeout"
    assert provider.started.is_set()
    assert provider.exited.wait(timeout=1)
    assert provider.cancel_calls == 1
    assert provider.reset_calls == 1


def test_runner_cancels_broker_request_left_by_worker_daemon_thread(
    tmp_path: Path,
) -> None:
    class BlockingProvider:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.cancel_calls = 0
            self.reset_calls = 0

        def complete(self, messages: Any) -> str:
            self.started.set()
            self.release.wait(timeout=2)
            return "released"

        def cancel_current(self) -> None:
            self.cancel_calls += 1
            self.release.set()

        def reset_cancellation(self) -> None:
            self.reset_calls += 1

    kernel = Kernel(tmp_path / "toolbox.db")
    kernel.call(
        "write_tool",
        {
            "name": "background_completion",
            "description": "Leave a provider request active while returning.",
            "input_schema": {"type": "object"},
            "source": (
                "import threading\n"
                "import time\n\n"
                "def main(input, ctx):\n"
                "    started = threading.Event()\n"
                "    def request():\n"
                "        started.set()\n"
                "        ctx.model_provider.complete([\n"
                "            {'role': 'user', 'content': 'background'},\n"
                "        ])\n"
                "    threading.Thread(target=request, daemon=True).start()\n"
                "    started.wait(timeout=1)\n"
                "    time.sleep(0.1)\n"
                "    return 'worker returned'\n"
            ),
        },
    )
    provider = BlockingProvider()

    result = IsolatedToolRunner(kernel).call_with_model_provider(
        "call_tool",
        {"name": "background_completion", "args": {}},
        session_id=kernel.create_session(),
        model_provider=provider,
    )

    assert result == "worker returned"
    assert provider.started.is_set()
    assert provider.cancel_calls == 1
    assert provider.reset_calls == 1


def test_isolated_runner_rejects_provider_without_cancellation(
    tmp_path: Path,
) -> None:
    class NonCancellableProvider:
        def complete(self, messages: Any) -> str:
            return "unreachable"

    kernel = Kernel(tmp_path / "toolbox.db")

    with pytest.raises(ToolboxError) as error:
        IsolatedToolRunner(kernel).call_with_model_provider(
            "help",
            {"topic": "quickstart"},
            session_id=kernel.create_session(),
            model_provider=NonCancellableProvider(),  # type: ignore[arg-type]
        )

    assert error.value.code == "invalid_model_provider"


def test_isolated_runner_rejects_wrapped_provider_without_cancellation(
    tmp_path: Path,
) -> None:
    class NonCancellableTransport:
        def respond(self, messages: Any, tools: Any) -> ModelTurn:
            return ModelTurn(text="unused")

        def complete(self, messages: Any) -> str:
            return "unreachable"

    kernel = Kernel(tmp_path / "toolbox.db")
    provider = ModelProvider(kernel, NonCancellableTransport())
    session = provider.start_session()

    with pytest.raises(ToolboxError) as error:
        IsolatedToolRunner(kernel).call_with_model_provider(
            "help",
            {"topic": "quickstart"},
            session_id=session.session_id,
            model_provider=session.completion_provider(),
        )

    assert error.value.code == "invalid_model_provider"


def test_isolated_worker_uses_persisted_session_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shadow = workspace / "mechagnome"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("")
    shadow_marker = tmp_path / "shadow-worker-ran"
    (shadow / "tool_worker.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(shadow_marker)!r}).write_text('shadowed')\n"
    )
    kernel = Kernel(tmp_path / "toolbox.db", cwd=workspace)
    kernel.write_tool(
        name="where",
        description="Return the process working directory.",
        input_schema={"type": "object"},
        source="import os\n\ndef main(input, ctx):\n    return os.getcwd()\n",
    )
    session_id = kernel.create_session()

    result = IsolatedToolRunner(kernel).call_with_model_provider(
        "call_tool",
        {"name": "where", "args": {}},
        session_id=session_id,
        model_provider=StaticProvider(),
    )

    assert result == str(workspace.resolve())
    assert shadow_marker.exists() is False


def test_inflight_worker_keeps_its_toolbox_selection_snapshot(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db", cwd=tmp_path)
    kernel.create_toolbox("alpha")
    kernel.create_toolbox("beta")
    session_id = kernel.create_session()
    kernel.select_toolboxes(session_id, ["alpha"], mode="use")
    kernel.write_tool(
        name="same",
        description="Return alpha.",
        input_schema={"type": "object"},
        source="def main(input, ctx):\n    return 'alpha'\n",
        session_id=session_id,
    )
    kernel.write_tool(
        name="wait_then_call",
        description="Wait for the host, then resolve a nested call.",
        input_schema={"type": "object"},
        source=(
            "import time\n"
            "from pathlib import Path\n\n"
            "def main(input, ctx):\n"
            "    Path(input['ready']).write_text('ready')\n"
            "    while not Path(input['release']).exists():\n"
            "        time.sleep(0.01)\n"
            "    return ctx.call_tool('same', {})\n"
        ),
        session_id=session_id,
    )
    kernel.select_toolboxes(session_id, ["beta"], mode="use")
    kernel.write_tool(
        name="same",
        description="Return beta.",
        input_schema={"type": "object"},
        source="def main(input, ctx):\n    return 'beta'\n",
        session_id=session_id,
    )
    kernel.select_toolboxes(session_id, ["alpha", "beta"], mode="use")
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    outcomes: list[Any] = []

    def run() -> None:
        outcomes.append(
            IsolatedToolRunner(kernel).call_with_model_provider(
                "call_tool",
                {
                    "name": "wait_then_call",
                    "args": {"ready": str(ready), "release": str(release)},
                },
                session_id=session_id,
                model_provider=StaticProvider(),
            )
        )

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 2
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    kernel.select_toolboxes(session_id, ["beta", "alpha"], mode="use")
    release.write_text("go")
    thread.join(timeout=2)

    assert outcomes == ["alpha"]
    assert kernel.call("same", {}, session_id=session_id) == "beta"


def test_bare_command_launches_tui_with_openrouter_defaults(
    tmp_path: Path, monkeypatch: Any
) -> None:
    database = tmp_path / "toolbox.db"
    launched: dict[str, Any] = {}

    def fake_run_tui(
        kernel: Kernel, model: OpenRouterModel, *, model_name: str
    ) -> None:
        launched.update(kernel=kernel, model=model, model_name=model_name)

    monkeypatch.setenv("MECHAGNOME_DB", str(database))
    monkeypatch.setattr(sys, "argv", ["mechagnome"])
    monkeypatch.setattr(cli, "run_tui", fake_run_tui)

    assert cli.main() == 0
    assert launched["kernel"].db_path == database
    assert launched["model"].base_url == DEFAULT_BASE_URL
    assert launched["model_name"] == DEFAULT_MODEL


def test_tui_clicks_to_change_model_and_reasoning_effort(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "example/standard",
                        "name": "Example Standard",
                        "supported_parameters": ["tools"],
                    },
                    {
                        "id": "example/reasoner",
                        "name": "Example Reasoner",
                        "supported_parameters": ["tools", "reasoning"],
                        "reasoning": {
                            "supported_efforts": ["high", "low"],
                            "mandatory": False,
                        },
                    },
                ]
            },
        )

    model = OpenRouterModel(
        model="example/standard",
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    app = ToolboxApp(
        kernel=Kernel(tmp_path / "toolbox.db"), model=model, model_name=model.model
    )

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            reasoning = app.query_one("#reasoning-selector", Button)
            assert reasoning.display is False

            await pilot.click("#model-selector")
            await pilot.pause()
            assert isinstance(app.screen, ModelSelectionScreen)
            app.screen.query_one("#model-picker", Select).value = "example/reasoner"
            await pilot.pause()
            await pilot.click("#confirm-model")
            await pilot.pause()

            assert model.model == "example/reasoner"
            assert reasoning.display is True
            assert "example/reasoner" in str(
                app.query_one("#model-selector", Button).label
            )

            await pilot.click("#reasoning-selector")
            await pilot.pause()
            assert isinstance(app.screen, ReasoningEffortScreen)
            app.screen.query_one("#reasoning-picker", Select).value = "high"
            await pilot.click("#confirm-reasoning")
            await pilot.pause()

            assert model.reasoning_effort == "high"
            assert "high" in str(reasoning.label)

            await pilot.click("#model-selector")
            await pilot.pause()
            assert isinstance(app.screen, ModelSelectionScreen)
            app.screen.query_one("#model-name", Input).value = "custom/unknown"
            await pilot.click("#confirm-model")
            await pilot.pause()

            assert model.model == "custom/unknown"
            assert model.reasoning_effort is None
            assert reasoning.display is False

    asyncio.run(exercise())


def test_model_selector_sorts_and_filters_catalog_options(tmp_path: Path) -> None:
    options = [
        OpenRouterModelOption(
            id="zeta/first", name="Zeta", input_modalities=("text", "audio")
        ),
        OpenRouterModelOption(
            id="alpha/second", name="Alpha Two", input_modalities=("image",)
        ),
        OpenRouterModelOption(
            id="alpha/first",
            name="Alpha One",
            input_modalities=("AUDIO", "IMAGE", "text"),
        ),
    ]
    app = ToolboxApp(
        kernel=Kernel(tmp_path / "toolbox.db"),
        model=FinalModel(),
        model_name="current/model",
    )

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(ModelSelectionScreen("current/model", options))
            await pilot.pause()
            picker = app.screen.query_one("#model-picker", Select)
            option_list = picker.query_one(OptionList)

            assert [
                str(option.prompt)
                for option in option_list.options
                if "  ·  " in str(option.prompt)
            ] == [
                "🖼️ 🎧  Alpha One  ·  alpha/first",
                "🖼️  Alpha Two  ·  alpha/second",
                "🎧  Zeta  ·  zeta/first",
            ]

            app.screen.query_one("#model-name", Input).value = "TWO"
            await pilot.pause()
            assert [
                str(option.prompt)
                for option in option_list.options
                if "  ·  " in str(option.prompt)
            ] == ["🖼️  Alpha Two  ·  alpha/second"]

            app.screen.query_one("#model-name", Input).value = "alpha/"
            await pilot.pause()
            assert [
                str(option.prompt)
                for option in option_list.options
                if "  ·  " in str(option.prompt)
            ] == [
                "🖼️ 🎧  Alpha One  ·  alpha/first",
                "🖼️  Alpha Two  ·  alpha/second",
            ]

    asyncio.run(exercise())


def test_tui_hides_reasoning_when_model_catalog_fails(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    model = OpenRouterModel(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    app = ToolboxApp(Kernel(tmp_path / "toolbox.db"), model, model_name=model.model)

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await app.workers.wait_for_complete()
            assert app.query_one("#reasoning-selector", Button).display is False
            await pilot.click("#model-selector")
            await pilot.pause()
            assert isinstance(app.screen, ModelSelectionScreen)
            assert app.screen.query_one("#model-name", Input).value == DEFAULT_MODEL

    asyncio.run(exercise())


def test_tui_shows_session_local_context_remaining(tmp_path: Path) -> None:
    app = ToolboxApp(
        Kernel(tmp_path / "toolbox.db"), FinalModel(), model_name="test/model"
    )

    async def exercise() -> None:
        async with app.run_test(size=(92, 32)) as pilot:
            context = app.query_one("#status-context", Static)
            separator = app.query_one("#context-separator", Static)
            first = app.active_session

            assert context.display is False
            assert separator.display is False

            app._display_event(
                first,
                AgentEvent(
                    "model",
                    {"text": "", "calls": [], "total_tokens": 1},
                    1,
                ),
            )
            assert context.display is False

            app.model_options = [
                OpenRouterModelOption(
                    id="test/model", name="Test Model", context_length=100
                )
            ]
            app._refresh_active_status()
            await pilot.pause()
            assert str(context.render()) == "context: 99% left"
            assert context.display is True
            assert separator.display is True
            assert context.region.right <= app.size.width

            app._display_event(
                first,
                AgentEvent(
                    "model",
                    {"text": "", "calls": [], "total_tokens": 100},
                    2,
                ),
            )
            assert str(context.render()) == "context: 0% left"
            app._display_event(
                first,
                AgentEvent(
                    "model",
                    {"text": "", "calls": [], "total_tokens": 120},
                    3,
                ),
            )
            assert str(context.render()) == "context: 0% left"

            app._display_event(
                first,
                AgentEvent(
                    "model",
                    {"text": "", "calls": [], "total_tokens": 25},
                    4,
                ),
            )
            assert str(context.render()) == "context: 75% left"

            await app.action_new_session()
            await pilot.pause()
            second = app.active_session
            assert second is not first
            assert context.display is False
            assert separator.display is False

            app._display_event(
                second,
                AgentEvent(
                    "model",
                    {"text": "", "calls": [], "total_tokens": 60},
                    1,
                ),
            )
            assert str(context.render()) == "context: 40% left"

            await pilot.press("tab")
            await pilot.pause()
            assert app.active_session is first
            assert str(context.render()) == "context: 75% left"

            app.model_name = "other/model"
            app._refresh_active_status()
            assert context.display is False
            assert separator.display is False
            app.model_name = "test/model"
            app._refresh_active_status()
            assert str(context.render()) == "context: 75% left"

            await pilot.press("tab")
            await pilot.pause()
            assert app.active_session is second
            await app.action_clear_session()
            assert context.display is False
            assert separator.display is False

            await pilot.press("tab")
            await pilot.pause()
            assert app.active_session is first
            assert str(context.render()) == "context: 75% left"

            app._display_event(
                first,
                AgentEvent("model", {"text": "", "calls": []}, 5),
            )
            assert context.display is False
            assert separator.display is False

    asyncio.run(exercise())


def test_tui_sends_a_prompt_without_blocking_and_saves_the_session(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    model = FinalModel()
    app = ToolboxApp(kernel, model, model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "hello from the tui"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert prompt.disabled is False
            assert model.message_snapshots[0][-1] == {
                "role": "user",
                "content": "hello from the tui",
            }

    asyncio.run(exercise())
    events = kernel.read_session(app.conversation.session_id, limit=100)["events"]
    assert [event["kind"] for event in events] == ["user", "model", "final"]


def test_tui_arrow_keys_scrub_session_prompt_history(tmp_path: Path) -> None:
    app = ToolboxApp(
        Kernel(tmp_path / "toolbox.db"), FinalModel(), model_name="test/model"
    )

    async def submit(pilot: Any, value: str) -> None:
        prompt = app.query_one("#prompt", Input)
        prompt.value = value
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "unsent draft"
            await pilot.press("up")
            assert prompt.value == "unsent draft"

            await submit(pilot, "first prompt")
            await submit(pilot, "second prompt")

            prompt.value = "new draft"
            await pilot.press("up")
            assert prompt.value == "second prompt"
            assert prompt.cursor_position == len("second prompt")
            await pilot.press("up")
            assert prompt.value == "first prompt"
            await pilot.press("up")
            assert prompt.value == "first prompt"

            await pilot.press("down")
            assert prompt.value == "second prompt"
            await pilot.press("down")
            assert prompt.value == "new draft"
            await pilot.press("down")
            assert prompt.value == "new draft"

            first = app.active_session
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert app.active_session is not first
            assert prompt.value == ""
            await pilot.press("up")
            assert prompt.value == ""

            await pilot.press("tab")
            await pilot.pause()
            assert app.active_session is first
            assert prompt.value == "new draft"
            await pilot.press("up")
            assert prompt.value == "second prompt"

    asyncio.run(exercise())


def test_tui_preserves_model_provider_when_starting_a_new_session(
    tmp_path: Path,
) -> None:
    provider = StaticProvider()
    app = ToolboxApp(
        Kernel(tmp_path / "toolbox.db"),
        FinalModel(),
        model_name="test/model",
        model_provider=provider,
    )
    original = app.active_session
    gateway = app.model_provider
    assert original.conversation.model_session.provider is gateway

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "first draft"
            await pilot.press("ctrl+n")
            await pilot.pause()
            second = app.active_session
            assert len(app.session_tabs) == 2
            assert second is not original
            assert second.conversation.session_id != original.conversation.session_id
            assert second.conversation.model_session.provider is gateway
            assert original in app.session_tabs

            prompt.value = "second draft"
            await pilot.press("tab")
            await pilot.pause()
            assert app.active_session is original
            assert prompt.value == "first draft"

            await pilot.press("tab")
            await pilot.pause()
            assert app.active_session is second
            assert prompt.value == "second draft"

            tabs = app.query_one("#session-tabs", TabbedContent)
            await pilot.click(tabs.get_tab(original.pane_id))
            await pilot.pause()
            assert app.active_session is original

    asyncio.run(exercise())


def test_clear_resets_one_tab_and_end_closes_it(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db", cwd=tmp_path)
    app = ToolboxApp(kernel, FinalModel(), model_name="test/model")

    async def submit(pilot: Any, value: str) -> None:
        prompt = app.query_one("#prompt", Input)
        prompt.value = value
        prompt.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            first = app.active_session
            await submit(pilot, "keep the first tab")
            prompt = app.query_one("#prompt", Input)
            prompt.value = "keep this draft too"
            await pilot.press("ctrl+n")
            await pilot.pause()
            second = app.active_session
            await submit(pilot, "clear this tab")

            old_conversation = second.conversation
            old_session_id = old_conversation.session_id
            pane_id = second.pane_id
            label = second.label
            await submit(pilot, "/clear")

            assert app.active_session is second
            assert second.pane_id == pane_id
            assert second.label == label
            assert second.conversation.session_id != old_session_id
            assert second.conversation.messages == []
            with pytest.raises(RunCancelled):
                old_conversation.send("closed")
            assert "clear this tab" not in chat_text(app, second.chat)
            assert kernel.read_session(old_session_id, limit=100)["events"]
            assert len(app.session_tabs) == 2
            assert first.conversation.messages
            assert not second.forwarded_targets
            assert not second.active_tool_events

            replacement = second.conversation
            await submit(pilot, "/end")
            with pytest.raises(RunCancelled):
                replacement.send("closed")
            assert app.active_session is first
            assert app.session_tabs == [first]
            assert first.draft == "keep this draft too"
            assert prompt.value == "keep this draft too"

            await pilot.press("ctrl+n")
            await pilot.pause()
            third = app.active_session
            assert third.pane_id == "session-3"
            assert third.label == "Session 3"

    asyncio.run(exercise())


def test_end_on_final_session_exits_and_closes_conversation(tmp_path: Path) -> None:
    app = ToolboxApp(
        Kernel(tmp_path / "toolbox.db"), FinalModel(), model_name="test/model"
    )
    conversation = app.conversation

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)):
            await app.action_end_session()

    asyncio.run(exercise())
    with pytest.raises(RunCancelled):
        conversation.send("closed")


class PausingStreamingModel:
    """Pause after one delta so the live TUI state can be inspected."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Any:
        yield ModelStreamEvent(text_delta="Partial")
        self.started.set()
        self.release.wait(timeout=3)
        yield ModelStreamEvent(text_delta=" response")
        yield ModelStreamEvent(turn=ModelTurn(text="Partial response"))

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        raise AssertionError("streaming interface should be preferred")

    def cancel_current(self) -> None:
        self.cancelled.set()
        self.release.set()


class ConcurrentPausingModel:
    """Pause independent root sessions so overlap and cancellation are observable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started: dict[str, threading.Event] = {}
        self.releases: dict[str, threading.Event] = {}
        self.cancelled: set[str] = set()

    def for_session(self, root_session_id: str) -> BoundConcurrentPausingModel:
        with self._lock:
            self.started.setdefault(root_session_id, threading.Event())
            self.releases.setdefault(root_session_id, threading.Event())
        return BoundConcurrentPausingModel(self, root_session_id)

    def wait_until_started(self, root_session_id: str) -> bool:
        return self.started[root_session_id].wait(timeout=1)

    def release(self, root_session_id: str) -> None:
        self.releases[root_session_id].set()


class BoundConcurrentPausingModel:
    def __init__(self, model: ConcurrentPausingModel, root_session_id: str) -> None:
        self.model = model
        self.root_session_id = root_session_id

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Any:
        prompt = str(messages[-1]["content"])
        self.model.started[self.root_session_id].set()
        yield ModelStreamEvent(text_delta=f"Partial {prompt}")
        self.model.releases[self.root_session_id].wait(timeout=3)
        text = f"Finished {prompt}"
        yield ModelStreamEvent(turn=ModelTurn(text=text))

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        raise AssertionError("streaming interface should be preferred")

    def cancel_current(self) -> None:
        self.model.cancelled.add(self.root_session_id)
        self.model.release(self.root_session_id)

    def reset_cancellation(self) -> None:
        pass


def test_tui_renders_model_text_before_stream_completion(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    model = PausingStreamingModel()
    app = ToolboxApp(kernel, model, model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "stream please"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if model.started.is_set() and app.streamed_text:
                    break
            assert app.streamed_text == "Partial"
            stream = app.query_one(".streaming-response", Static)
            assert stream.parent is app.chat
            assert "Partial" in chat_text(app)
            model.release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.streamed_text == ""
            assert not app.query(".streaming-response")
            chat = chat_text(app)
            assert "Partial response" in chat

    try:
        asyncio.run(exercise())
    finally:
        model.release.set()

    kinds = [
        event["kind"]
        for event in kernel.read_session(app.conversation.session_id, limit=100)[
            "events"
        ]
    ]
    assert kinds == ["user", "model", "final"]


def test_tabs_can_run_concurrently_without_misrouting_output(
    tmp_path: Path,
) -> None:
    model = ConcurrentPausingModel()
    app = ToolboxApp(Kernel(tmp_path / "toolbox.db"), model, model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            first = app.active_session
            first_id = first.conversation.session_id
            prompt = app.query_one("#prompt", Input)
            prompt.value = "first tab"
            await pilot.press("enter")
            assert model.wait_until_started(first_id)

            await pilot.press("ctrl+n")
            await pilot.pause()
            second = app.active_session
            second_id = second.conversation.session_id
            assert second is not first
            assert prompt.disabled is False

            prompt.value = "second tab"
            await pilot.press("enter")
            assert model.wait_until_started(second_id)
            for _ in range(30):
                await pilot.pause()
                if first.streamed_text and second.streamed_text:
                    break

            assert first.running is True
            assert second.running is True
            assert app.busy is True
            assert "Partial first tab" in chat_text(app, first.chat)
            assert "Partial second tab" in chat_text(app, second.chat)

            model.release(second_id)
            for _ in range(30):
                await pilot.pause()
                if not second.running:
                    break
            assert first.running is True
            assert second.running is False
            assert prompt.disabled is False
            assert "Finished second tab" in chat_text(app, second.chat)
            assert "Finished second tab" not in chat_text(app, first.chat)

            model.release(first_id)
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert first.running is False
            assert app.busy is False
            assert prompt.disabled is False
            assert "Finished first tab" in chat_text(app, first.chat)
            assert "Finished first tab" not in chat_text(app, second.chat)

    asyncio.run(exercise())


def test_active_tab_cancellation_and_idle_teardown_are_session_local(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    model = ConcurrentPausingModel()
    app = ToolboxApp(kernel, model, model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            first = app.active_session
            first_id = first.conversation.session_id
            prompt = app.query_one("#prompt", Input)
            prompt.value = "keep running"
            await pilot.press("enter")
            assert model.wait_until_started(first_id)

            await pilot.press("ctrl+n")
            await pilot.pause()
            second = app.active_session
            second_id = second.conversation.session_id
            prompt.value = "cancel me"
            await pilot.press("enter")
            assert model.wait_until_started(second_id)

            await pilot.press("escape")
            for _ in range(30):
                await pilot.pause()
                if not second.running:
                    break
            assert second_id in model.cancelled
            assert first_id not in model.cancelled
            assert first.running is True
            assert second.running is False
            assert "stopped" in chat_text(app, second.chat)

            old_second = second.conversation
            await app.action_clear_session()
            assert second.conversation is not old_second
            await app.action_end_session()
            assert second not in app.session_tabs
            assert app.active_session is first

            await app.action_clear_session()
            await app.action_end_session()
            assert first in app.session_tabs
            assert first.running is True

            model.release(first_id)
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert first.running is False
            assert first_id not in model.cancelled

    asyncio.run(exercise())


def test_tool_manager_opens_during_streaming_rollout(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    model = PausingStreamingModel()
    app = ToolboxApp(kernel, model, model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "stream while inspecting tools"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if model.started.is_set() and app.streamed_text:
                    break

            assert model.started.is_set()
            assert app.streamed_text == "Partial"
            assert app.busy is True
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert isinstance(app.screen, ToolManagerScreen)
            assert app.busy is True

            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, ToolManagerScreen)
            assert app.busy is False
            assert model.cancelled.is_set()

    try:
        asyncio.run(exercise())
    finally:
        model.release.set()


def test_ctrl_t_toggles_tool_manager(tmp_path: Path) -> None:
    app = ToolboxApp(
        Kernel(tmp_path / "toolbox.db"), FinalModel(), model_name="test/model"
    )

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+n")
            await pilot.pause()
            active = app.active_session
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert isinstance(app.screen, ToolManagerScreen)

            await pilot.press("tab")
            await pilot.pause()
            assert app.active_session is active

            await pilot.press("ctrl+t")
            await pilot.pause()
            assert not isinstance(app.screen, ToolManagerScreen)

    asyncio.run(exercise())


def test_escape_stops_streaming_rollout_and_records_cancellation(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    model = PausingStreamingModel()
    app = ToolboxApp(kernel, model, model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "stop this stream"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if model.started.is_set() and app.streamed_text:
                    break
            assert app.streamed_text == "Partial"
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.busy is False
            assert prompt.disabled is False
            chat = chat_text(app)
            assert "stopped" in chat

    try:
        asyncio.run(exercise())
    finally:
        model.release.set()

    events = kernel.read_session(app.conversation.session_id, limit=100)["events"]
    assert [event["kind"] for event in events] == ["user", "cancelled"]
    assert app.conversation.messages == []
    resumed = Harness(kernel).start(
        FinalModel(), session_id=app.conversation.session_id
    )
    assert resumed.messages == []


def test_ctrl_q_stops_active_rollout_before_exiting(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    model = PausingStreamingModel()
    app = ToolboxApp(kernel, model, model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "quit during this stream"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if model.started.is_set():
                    break
            assert model.started.is_set()
            await pilot.press("ctrl+q")

    try:
        asyncio.run(exercise())
        assert model.cancelled.is_set()
    finally:
        model.release.set()

    events = kernel.read_session(app.conversation.session_id, limit=100)["events"]
    assert [event["kind"] for event in events] == ["user", "cancelled"]


def test_unmount_stops_active_rollout_before_programmatic_exit(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    model = PausingStreamingModel()
    app = ToolboxApp(kernel, model, model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "exit during this stream"
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause()
                if model.started.is_set():
                    break
            assert model.started.is_set()
            app.exit()

    try:
        asyncio.run(exercise())
        assert model.cancelled.is_set()
    finally:
        model.release.set()

    events = kernel.read_session(app.conversation.session_id, limit=100)["events"]
    assert [event["kind"] for event in events] == ["user", "cancelled"]


class ToolThenFinalModel:
    """Call a deliberately slow tool unless the rollout is stopped."""

    def __init__(self) -> None:
        self.turn = 0

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                calls=(ToolCall("call_tool", {"name": "slow", "args": {}}, "slow-1"),)
            )
        return ModelTurn(text="Tool completed.")


def test_escape_terminates_active_tool_subprocess(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    pid_path = tmp_path / "slow-tool-pids"
    child_ready_path = tmp_path / "slow-tool-child-ready"
    child_code = (
        "import signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(child_ready_path)!r}).write_text('ready')\n"
        "time.sleep(30)\n"
    )
    kernel.call(
        "write_tool",
        {
            "name": "slow",
            "description": "Sleep long enough to exercise cancellation.",
            "input_schema": {"type": "object"},
            "source": (
                "import os\n"
                "from pathlib import Path\n"
                "import subprocess\n"
                "import sys\n"
                "import time\n\n"
                "def main(input, ctx):\n"
                f"    child_code = {child_code!r}\n"
                "    child = subprocess.Popen(\n"
                "        [sys.executable, '-c', child_code]\n"
                "    )\n"
                f"    while not Path({str(child_ready_path)!r}).exists():\n"
                "        time.sleep(0.01)\n"
                f"    Path({str(pid_path)!r}).write_text(\n"
                "        f'{os.getpid()} {child.pid}', encoding='utf-8'\n"
                "    )\n"
                "    time.sleep(30)\n"
                "    return {'finished': True}\n"
            ),
        },
    )
    model = ToolThenFinalModel()
    app = ToolboxApp(kernel, model, model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "run the slow tool"
            await pilot.press("enter")
            for _ in range(200):
                await pilot.pause()
                events = kernel.read_session(app.conversation.session_id, limit=100)[
                    "events"
                ]
                if (
                    any(
                        event["kind"] == "call_started" and event["tool_name"] == "slow"
                        for event in events
                    )
                    and pid_path.exists()
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("slow tool never started")
            for _ in range(30):
                await pilot.pause()
                if any(
                    event.kind == "call" and event.tool_name == "slow"
                    for event in app.query(ToolEvent)
                ):
                    break
            slow_call = next(
                event
                for event in app.query(ToolEvent)
                if event.kind == "call" and event.tool_name == "slow"
            )
            await asyncio.sleep(0.1)
            assert slow_call.processing is True
            assert slow_call._spinner_timer is not None
            assert (
                tool_title_text(slow_call).split(" ", 1)[0] in ToolEvent.SPINNER_FRAMES
            )
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert slow_call.processing is False
            assert slow_call._spinner_timer is None
            assert tool_title_text(slow_call) == "→ slow"
            assert app.busy is False
            assert prompt.disabled is False

    worker_pid = child_pid = None
    try:
        asyncio.run(exercise())
        worker_pid, child_pid = map(int, pid_path.read_text().split())
        for _ in range(100):
            if not _process_exists(worker_pid) and not _process_exists(child_pid):
                break
            time.sleep(0.01)
        assert _process_exists(worker_pid) is False
        assert _process_exists(child_pid) is False
    finally:
        for pid in (worker_pid, child_pid):
            if pid is not None and _process_exists(pid):
                os.kill(pid, signal.SIGKILL)

    events = kernel.read_session(app.conversation.session_id, limit=100)["events"]
    assert events[-1]["kind"] == "cancelled"
    assert model.turn == 1
    assert app.conversation.messages == []


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class ToolWritingModel:
    """Write and call one user tool for headless TUI activity coverage."""

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
                        "write_tool",
                        {
                            "name": "hello",
                            "description": "Return a greeting.",
                            "input_schema": {"type": "object"},
                            "source": (
                                "def main(input, ctx):\n    return {'hello': 'world'}\n"
                            ),
                        },
                        "write-hello",
                    ),
                )
            )
        if self.turn == 2:
            return ModelTurn(
                calls=(
                    ToolCall(
                        "call_tool",
                        {"name": "hello", "args": {"greeting": "hello world"}},
                        "use",
                    ),
                )
            )
        return ModelTurn(text="Built and used hello.")


def test_tui_shows_tool_activity_refreshes_sidebar_and_runs_commands(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    app = ToolboxApp(kernel, ToolWritingModel(), model_name="test/model")

    async def submit(pilot: Any, value: str) -> None:
        prompt = app.query_one("#prompt", Input)
        prompt.value = value
        prompt.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(pilot, "build a hello tool")
            chat = chat_text(app)
            tools = "\n".join(
                line.text for line in app.query_one("#tools", RichLog).lines
            )
            assert "write_tool" in chat
            assert "call_tool" not in chat
            assert "hello" in chat
            activity = [(event.kind, event.tool_name) for event in app.query(ToolEvent)]
            assert activity.count(("response", "hello")) == 1
            assert len(activity) == 2
            assert all(kind == "response" for kind, _ in activity)
            assert all(event.collapsed for event in app.query(ToolEvent))
            hello = next(
                event
                for event in app.query(ToolEvent)
                if event.kind == "response" and event.tool_name == "hello"
            )
            events = kernel.read_session(app.conversation.session_id, limit=100)[
                "events"
            ]
            outer = next(
                event
                for event in events
                if event["kind"] == "call_succeeded"
                and event["tool_name"] == "call_tool"
            )
            assert tool_title_text(hello) == (
                '✓ hello [greeting="hello world"] · completed in '
                f"{_format_duration(outer['payload']['duration_ms'])}"
            )
            assert hello.processing is False
            assert hello._spinner_timer is None
            assert hello.has_class("tool-response")
            assert hello.has_class("tool-call") is False
            assert '"greeting": "hello world"' in hello.detail
            assert '"hello": "world"' in hello.detail
            response_title = hello.query_one("CollapsibleTitle", Static)
            assert hello.styles.margin.bottom == 0
            assert response_title.styles.text_style.italic is True
            assert hello.detail_widget.styles.border.top[0] == "round"
            await pilot.click(hello)
            assert hello.collapsed is False
            assert "hello  v1" in tools
            assert "1 user" in str(app.query_one("#model-info", Static).render())

            await submit(pilot, "/tools")
            assert isinstance(app.screen, ToolManagerScreen)
            assert app.screen.selected_name == "hello"
            await pilot.press("escape")
            await pilot.pause()
            await submit(pilot, "/sessions")
            old_session = app.conversation.session_id
            inspection = chat_text(app)
            assert "saved sessions" in inspection
            assert old_session[:12] in inspection
            await submit(pilot, "/new")
            assert app.conversation.session_id != old_session
            assert len(kernel.list_sessions(limit=10)["sessions"]) == 2

    asyncio.run(exercise())


def test_tool_argument_summary_is_single_line_and_truncated() -> None:
    summary = ToolboxApp._argument_summary(
        {"query": "first line\nsecond line", "limit": 50},
        limit=32,
    )

    assert summary.startswith(' [query="first line\\nsecond')
    assert summary.endswith("…]")
    assert len(summary) == 32
    assert "\n" not in summary

    unsafe = ToolboxApp._argument_summary(
        {"que\n\x1bry": "value", "\x9b": "x"},
    )
    assert unsafe == ' [que\\n\\u001bry="value", \\u009b="x"]'
    assert all(character.isprintable() for character in unsafe)


class SingleToolModel:
    """Call one authored tool, then stop."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.turn = 0

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                calls=(
                    ToolCall("call_tool", {"name": self.tool_name, "args": {}}, "use"),
                )
            )
        return ModelTurn(text="Observed the result.")


def replace_call_tool(kernel: Kernel, source: str) -> None:
    current = kernel.view_tool("call_tool")
    kernel.write_tool(
        name="call_tool",
        description="Test dispatcher replacement.",
        input_schema=current["input_schema"],
        source=source,
        base_version=current["version"],
    )


def test_tui_collapses_forwarded_tool_failure_without_duplicates(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    kernel.write_tool(
        name="boom",
        description="Raise a test error.",
        input_schema={"type": "object"},
        source="def main(input, ctx):\n    raise RuntimeError('broken')\n",
    )
    app = ToolboxApp(kernel, SingleToolModel("boom"), model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "run boom"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            activity = [(event.kind, event.tool_name) for event in app.query(ToolEvent)]
            assert activity == [("error", "boom")]
            assert "call_tool" not in chat_text(app)
            failed = next(iter(app.query(ToolEvent)))
            events = kernel.read_session(app.conversation.session_id, limit=100)[
                "events"
            ]
            outer = next(
                event
                for event in events
                if event["kind"] == "call_failed" and event["tool_name"] == "call_tool"
            )
            assert tool_title_text(failed).endswith(
                f"completed in {_format_duration(outer['payload']['duration_ms'])}"
            )
            assert "duration_ms" not in failed.detail

    asyncio.run(exercise())


@pytest.mark.parametrize(
    (
        "target",
        "target_source",
        "dispatcher_source",
        "expected_kind",
        "expected_name",
        "expected_fragment",
    ),
    [
        (
            "boom",
            "def main(input, ctx):\n    raise RuntimeError('child broke')\n",
            (
                "def main(input, ctx):\n"
                "    try:\n"
                "        return ctx.kernel.execute(input['name'], input['args'])\n"
                "    except Exception:\n"
                "        return {'recovered': True}\n"
            ),
            "response",
            "boom",
            '"recovered": true',
        ),
        (
            "okay",
            "def main(input, ctx):\n    return {'child': 'success'}\n",
            (
                "def main(input, ctx):\n"
                "    ctx.kernel.execute(input['name'], input['args'])\n"
                "    raise RuntimeError('outer broke')\n"
            ),
            "error",
            "okay",
            "outer broke",
        ),
        (
            "unused",
            "def main(input, ctx):\n    return 'not called'\n",
            "def main(input, ctx):\n    return {'intercepted': input['name']}\n",
            "response",
            "call_tool",
            '"intercepted": "unused"',
        ),
    ],
)
def test_tui_uses_authoritative_dispatcher_outcome(
    tmp_path: Path,
    target: str,
    target_source: str,
    dispatcher_source: str,
    expected_kind: str,
    expected_name: str,
    expected_fragment: str,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    kernel.write_tool(
        name=target,
        description="Exercise dispatcher behavior.",
        input_schema={"type": "object"},
        source=target_source,
    )
    replace_call_tool(kernel, dispatcher_source)
    app = ToolboxApp(kernel, SingleToolModel(target), model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = f"run {target}"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            activity = list(app.query(ToolEvent))
            assert [(event.kind, event.tool_name) for event in activity] == [
                (expected_kind, expected_name),
            ]
            assert expected_fragment in activity[0].detail

    asyncio.run(exercise())


def test_tui_creates_and_composes_toolbox_namespaces(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db", cwd=tmp_path)
    kernel.create_toolbox("alpha")
    kernel.create_toolbox("beta")
    setup = kernel.create_session()
    kernel.select_toolboxes(setup, ["alpha"], mode="use")
    kernel.write_tool(
        name="identity",
        description="Return alpha.",
        input_schema={"type": "object"},
        source="def main(input, ctx):\n    return 'alpha'\n",
        session_id=setup,
    )
    kernel.select_toolboxes(setup, ["beta"], mode="use")
    kernel.write_tool(
        name="identity",
        description="Return beta.",
        input_schema={"type": "object"},
        source="def main(input, ctx):\n    return 'beta'\n",
        session_id=setup,
    )
    app = ToolboxApp(kernel, FinalModel(), model_name="test/model")

    async def submit(pilot: Any, value: str) -> None:
        prompt = app.query_one("#prompt", Input)
        prompt.value = value
        await pilot.press("enter")
        await pilot.pause()

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await submit(pilot, "/toolbox use beta alpha")
            selected = kernel.active_toolboxes(app.conversation.session_id)
            assert [item["name"] for item in selected] == ["beta", "alpha"]
            assert (
                kernel.call("identity", {}, session_id=app.conversation.session_id)
                == "beta"
            )
            sidebar = "\n".join(
                line.text for line in app.query_one("#tools", RichLog).lines
            )
            assert "identity  v1  [beta]" in sidebar
            assert "beta + alpha" in str(app.query_one("#model-info", Static).render())
            await submit(pilot, "/toolbox list")
            chat = chat_text(app)
            assert "toolbox namespaces" in chat

    asyncio.run(exercise())


def test_tool_manager_switches_blanks_and_renames_namespaces(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db", cwd=tmp_path)
    kernel.create_toolbox("alpha", cwd=tmp_path)
    kernel.create_toolbox("beta")
    app = ToolboxApp(kernel, FinalModel(), model_name="test/model")
    session_id = app.conversation.session_id
    kernel.write_tool(
        name="alpha_tool",
        description="Only in alpha.",
        input_schema={"type": "object"},
        source="def main(input, ctx):\n    return 'alpha'\n",
        session_id=session_id,
    )
    kernel.select_toolboxes(session_id, ["beta"], mode="use")
    kernel.write_tool(
        name="beta_tool",
        description="Only in beta.",
        input_schema={"type": "object"},
        source="def main(input, ctx):\n    return 'beta'\n",
        session_id=session_id,
    )
    kernel.select_toolboxes(session_id, ["alpha"], mode="use")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert isinstance(app.screen, ToolManagerScreen)
            manager = app.screen
            assert manager.selected_namespace == "alpha"
            assert manager.selected_name == "alpha_tool"

            manager.query_one("#namespace-picker", Select).value = "beta"
            await pilot.pause()
            assert [item["name"] for item in kernel.active_toolboxes(session_id)] == [
                "beta"
            ]
            assert manager.selected_namespace == "beta"
            assert manager.selected_name == "beta_tool"

            await pilot.click("#blank-namespace")
            await pilot.pause()
            assert isinstance(app.screen, NamespaceNameScreen)
            app.screen.query_one("#namespace-name", Input).value = "gamma"
            await pilot.click("#confirm-namespace")
            await pilot.pause()
            assert isinstance(app.screen, ToolManagerScreen)
            manager = app.screen
            assert manager.selected_namespace == "gamma"
            assert [item["name"] for item in kernel.active_toolboxes(session_id)] == [
                "gamma"
            ]
            assert "alpha_tool" not in {item["name"] for item in manager.inventory}
            assert "beta_tool" not in {item["name"] for item in manager.inventory}

            gamma_id = kernel.active_toolboxes(session_id)[0]["id"]
            await pilot.click("#save-as-namespace")
            await pilot.pause()
            assert isinstance(app.screen, NamespaceNameScreen)
            app.screen.query_one("#namespace-name", Input).value = "renamed"
            await pilot.click("#confirm-namespace")
            await pilot.pause()
            assert isinstance(app.screen, ToolManagerScreen)
            manager = app.screen
            assert manager.selected_namespace == "renamed"
            active = kernel.active_toolboxes(session_id)
            assert [(item["id"], item["name"]) for item in active] == [
                (gamma_id, "renamed")
            ]
            assert "gamma" not in {item["name"] for item in kernel.list_toolboxes()}

    asyncio.run(exercise())


def test_tui_toolbox_is_roomy_themed_and_sorted_by_recent_usage(
    tmp_path: Path,
) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    kernel.call("help", {"topic": "quickstart"})
    kernel.call("search_tools", {"query": "missing"})
    app = ToolboxApp(kernel, FinalModel(), model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(80, 30)):
            sidebar = app.query_one("#sidebar")
            tools = app.query_one("#tools", RichLog)
            rendered = "\n".join(line.text for line in tools.lines)

            assert sidebar.outer_size.width == 42
            assert tools.styles.background.a == 0
            assert tools.min_width == 1
            assert tools.virtual_size.width <= tools.size.width
            assert rendered.index("search_tools  v1") < rendered.index("help  v1")
            assert rendered.index("help  v1") < rendered.index("call_tool  v1")

    asyncio.run(exercise())


def test_tool_manager_navigates_source_diff_stats_and_deletes(tmp_path: Path) -> None:
    kernel = Kernel(tmp_path / "toolbox.db")
    creator = kernel.create_session()
    kernel.write_tool(
        name="number",
        description="Return the first number without parsing [bold]markup[/bold].",
        input_schema={"type": "object"},
        source="def main(input, ctx):\n    return 1",
        session_id=creator,
    )
    kernel.write_tool(
        name="number",
        description="A literal closing tag: [/bold]",
        input_schema={"type": "object"},
        source="def main(input, ctx):\n    return 2",
        base_version=1,
        session_id=creator,
    )
    kernel.call("number", {}, session_id=creator)
    app = ToolboxApp(kernel, FinalModel(), model_name="test/model")

    async def exercise() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert isinstance(app.screen, ToolManagerScreen)
            manager = app.screen
            assert manager.selected_name == "number"
            assert manager.selected_version == 2
            assert "feedback-tab" not in {pane.id for pane in manager.query("TabPane")}
            source = manager.query_one("#tool-source", Static).content
            assert isinstance(source, Syntax)
            assert "return 2" in source.code
            diff = manager.query_one("#tool-diff", Static).content
            assert isinstance(diff, Syntax)
            assert "number@v1" in diff.code
            assert "+    return 2" in diff.code
            assert "-    return 1\n+    return 2" in diff.code
            assert manager.history["call_count"] == 1
            assert manager.history["versions"][0]["created_session_id"] == creator
            assert manager.history["versions"][0]["timed_call_count"] == 1
            assert manager.history["versions"][0]["average_duration_ms"] >= 0
            summary = manager.query_one("#tool-summary", Static).content
            assert isinstance(summary, Text)
            assert "[/bold]" in summary.plain
            usage = manager.query_one("#tool-usage", Static).content
            rendered_usage = StringIO()
            Console(file=rendered_usage, force_terminal=False).print(usage)
            assert "Version ID" in rendered_usage.getvalue()
            assert "Timed calls" in rendered_usage.getvalue()
            assert "Average duration" in rendered_usage.getvalue()

            manager.query_one("#version-picker", Select).value = 1
            await pilot.pause()
            assert manager.selected_version == 1
            source = manager.query_one("#tool-source", Static).content
            assert isinstance(source, Syntax)
            assert "return 1" in source.code

            await pilot.click("#delete-tool")
            await pilot.pause()
            assert isinstance(app.screen, DeleteToolScreen)
            await pilot.click("#confirm-delete")
            await pilot.pause()
            assert isinstance(app.screen, ToolManagerScreen)
            manager = app.screen
            assert manager.history["active_version"] is None
            assert manager.query_one("#delete-tool", Button).disabled is True
            assert "number" not in {binding["name"] for binding in kernel.bindings()}
            assert kernel.tool_history("number")["versions"][0]["version"] == 2

    asyncio.run(exercise())
