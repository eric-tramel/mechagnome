"""OpenRouter adapter for the provider-neutral toolbox harness."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import httpx

from mechagnome.model_provider import (
    ModelStreamEvent,
    ModelTransportError,
    ModelTurn,
    ToolCall,
)

DEFAULT_MODEL = "z-ai/glm-5.2"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"
MAX_STREAM_BYTES = 4_000_000
MAX_COMPLETION_TOKENS = 2048
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_DEFAULT_SESSION = object()
_PUBLIC_ERROR_MESSAGES = {
    "missing_api_key": "OpenRouter API key is missing",
    "openrouter_cancelled": "OpenRouter request was cancelled",
    "openrouter_finish_reason": "OpenRouter returned an unexpected finish reason",
    "openrouter_http": "OpenRouter returned an HTTP error",
    "openrouter_models": "OpenRouter returned an invalid model catalog",
    "openrouter_response": "OpenRouter returned an invalid response",
    "openrouter_response_too_large": "OpenRouter response exceeded the size limit",
    "openrouter_stream": "OpenRouter stream failed",
    "openrouter_timeout": "OpenRouter request timed out",
    "openrouter_tool_call": "OpenRouter returned an invalid tool call",
    "openrouter_transport": "OpenRouter transport failed",
    "openrouter_truncated": "OpenRouter stream ended unexpectedly",
}

# Open-ended nested objects are not represented consistently by every model or
# provider behind a Responses-compatible tool-calling endpoint. Keep the kernel's
# object-based ABI and use explicit JSON strings only at this transport boundary.
JSON_OBJECT_ARGUMENTS = {
    "write_tool": {
        "input_schema": (
            "A JSON-encoded JSON Schema object for the authored tool's input. "
            'For example: {"type":"object","properties":{"query":'
            '{"type":"string"}},"required":["query"],'
            '"additionalProperties":false}'
        ),
    },
    "call_tool": {
        "args": (
            "A JSON-encoded object containing the target tool's arguments. "
            'Use "{}" when the tool takes no input.'
        ),
    },
}

DEFAULT_SYSTEM_PROMPT = """\
You are the agent inside Mechagnome, a persistent metaprogrammable toolbox.
You can only directly call help, list_tools, list_tool_namespaces, search_tools,
view_tool, write_tool, and call_tool, plus the host-owned run_agent action. Every
agent you run receives this same action surface. The session's selected toolbox
stack begins with no domain-specific user tools. Tools can belong to multiple
hierarchical discovery namespaces; list or search those namespaces before
creating duplicates; build small reusable Python tools when they improve the
task; call and repair them immediately; and reuse them across later requests.
Request independent operations together in modest batches. Keep an operation
in a later turn when it depends on the output of an earlier operation.
For a long-running independent call_tool invocation, set detach=true, continue
other work with the returned job_id, and inspect it later with call_tool.
Run delegated work directly with run_agent. Set detach=true when it should
continue independently, and inspect the returned job_id with run_agent later.
Tools may call other tools through ctx.call_tool, read or annotate current and
historical sessions through ctx.sessions, and use the ordinary Linux/Python environment.
Source passed to write_tool must define async def main(input, ctx). Await
ctx.call_tool and async ctx.model_provider operations; the call_tool core slot
must also await ctx.kernel.execute.
Use help when you need the tool ABI or examples. Core operation source is also
readable and editable, but keep changes deliberate. When the user's task is
complete, return a concise final answer instead of making another tool call.
"""


class OpenRouterError(ModelTransportError):
    """A structured provider or response-shape failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(
            code,
            message,
            public_message=_PUBLIC_ERROR_MESSAGES.get(code),
            **details,
        )


@dataclass(frozen=True)
class OpenRouterModelOption:
    """One tool-capable model exposed by OpenRouter's model catalog."""

    id: str
    name: str
    input_modalities: tuple[str, ...] = ()
    reasoning_efforts: tuple[str, ...] = ()
    reasoning_mandatory: bool = False
    context_length: int | None = None


def _positive_int(value: Any) -> int | None:
    """Return provider metadata only when it is a positive plain integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


@dataclass
class _ActiveSession:
    """Cancelable OpenRouter resources active in one root session."""

    clients: dict[int, httpx.Client] = field(default_factory=dict)
    responses: dict[int, httpx.Response] = field(default_factory=dict)
    cancel_requested: bool = False


class OpenRouterModel:
    """Streaming adapter for OpenRouter's stateless Responses API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout: float = 180.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort: str | None = None
        self.base_url = DEFAULT_BASE_URL
        self.api_key_env = DEFAULT_KEY_ENV
        self.api_key = api_key or os.environ.get(DEFAULT_KEY_ENV)
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.client = client
        self._active_lock = Lock()
        self._active_sessions: dict[object, _ActiveSession] = {}

    @property
    def ready(self) -> bool:
        """Whether the configured API-key environment is available."""
        return bool(self.api_key)

    def cancel_current(self) -> None:
        """Close the active streaming response so a blocked read wakes promptly."""
        self._cancel_session(_DEFAULT_SESSION)

    def reset_cancellation(self) -> None:
        """Discard the direct-call cancellation latch after a rollout ends."""
        self._reset_session(_DEFAULT_SESSION)

    def for_session(self, session_id: str) -> _SessionOpenRouterModel:
        """Return a view isolated to one conversation cancellation domain."""
        return _SessionOpenRouterModel(self, session_id)

    def _cancel_session(self, session_key: object) -> None:
        with self._active_lock:
            state = self._active_sessions.setdefault(session_key, _ActiveSession())
            state.cancel_requested = True
            responses = list(state.responses.values())
            clients = list(state.clients.values())
        for response in responses:
            with suppress(Exception):
                response.close()
        for client in clients:
            with suppress(Exception):
                client.close()

    def _reset_session(self, session_key: object) -> None:
        with self._active_lock:
            state = self._active_sessions.get(session_key)
            if state is None:
                return
            state.cancel_requested = False
            if not state.clients and not state.responses:
                self._active_sessions.pop(session_key, None)

    def available_models(self) -> list[OpenRouterModelOption]:
        """Return text models that can use Mechagnome's tool-calling surface."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            if self.client is None:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.get(
                        f"{self.base_url}/models",
                        headers=headers,
                        params={"supported_parameters": "tools"},
                    )
            else:
                response = self.client.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    params={"supported_parameters": "tools"},
                )
        except httpx.HTTPError as error:
            raise OpenRouterError(
                "openrouter_transport", f"OpenRouter model catalog failed: {error}"
            ) from error
        if response.status_code >= 400:
            raise self._http_error(response)
        try:
            payload = response.json()
        except ValueError as error:
            raise OpenRouterError(
                "openrouter_models", "OpenRouter returned an invalid model catalog"
            ) from error
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, list):
            raise OpenRouterError(
                "openrouter_models", "OpenRouter returned an invalid model catalog"
            )
        options: list[OpenRouterModelOption] = []
        for item in data:
            if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
                continue
            parameters = item.get("supported_parameters")
            supported = parameters if isinstance(parameters, list) else []
            if "tools" not in supported:
                continue
            architecture = item.get("architecture")
            input_modalities: tuple[str, ...] = ()
            if isinstance(architecture, Mapping):
                raw_modalities = architecture.get("input_modalities")
                if isinstance(raw_modalities, list):
                    input_modalities = tuple(
                        modality
                        for modality in raw_modalities
                        if isinstance(modality, str)
                    )
            reasoning = item.get("reasoning")
            efforts: tuple[str, ...] = ()
            mandatory = False
            if isinstance(reasoning, Mapping):
                mandatory = reasoning.get("mandatory") is True
                if "supported_efforts" in reasoning:
                    raw_efforts = reasoning["supported_efforts"]
                    if raw_efforts is None:
                        efforts = REASONING_EFFORTS
                    elif isinstance(raw_efforts, list):
                        efforts = tuple(
                            effort for effort in raw_efforts if isinstance(effort, str)
                        )
                if mandatory:
                    efforts = tuple(effort for effort in efforts if effort != "none")
            options.append(
                OpenRouterModelOption(
                    id=item["id"],
                    name=(
                        item["name"]
                        if isinstance(item.get("name"), str)
                        else item["id"]
                    ),
                    input_modalities=input_modalities,
                    reasoning_efforts=efforts,
                    reasoning_mandatory=mandatory,
                    context_length=_positive_int(item.get("context_length")),
                )
            )
        return options

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        """Consume a streaming OpenRouter response into one model turn."""
        return self._respond(messages, tools, _DEFAULT_SESSION)

    def _respond(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_key: object,
    ) -> ModelTurn:
        completed: ModelTurn | None = None
        for event in self._stream(messages, tools, session_key):
            if event.turn is not None:
                completed = event.turn
        if completed is None:
            raise OpenRouterError(
                "openrouter_response", "OpenRouter stream ended without a completion"
            )
        return completed

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return a text-only completion without the outer agent prompt or tools."""
        return self._complete(messages, _DEFAULT_SESSION)

    def _complete(
        self, messages: Sequence[Mapping[str, str]], session_key: object
    ) -> str:
        if not self.api_key:
            raise OpenRouterError(
                "missing_api_key",
                f"set {self.api_key_env} before sending a message",
                env=self.api_key_env,
            )
        body: dict[str, Any] = {
            "model": self.model,
            "input": self._wire_input(messages),
            "max_output_tokens": MAX_COMPLETION_TOKENS,
            "store": False,
            "stream": False,
        }
        try:
            if self.client is None:
                with (
                    httpx.Client(timeout=self.timeout) as client,
                    self._active(
                        session_key,
                        client=client,
                        close_client_on_cancel=True,
                    ),
                ):
                    return self._completion_response(client, body, session_key)
            with self._active(session_key, client=self.client):
                return self._completion_response(self.client, body, session_key)
        except (httpx.HTTPError, httpx.StreamError) as error:
            raise OpenRouterError(
                "openrouter_transport", f"OpenRouter request failed: {error}"
            ) from error

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Iterator[ModelStreamEvent]:
        """Yield OpenRouter text deltas and one assembled final turn."""
        yield from self._stream(messages, tools, _DEFAULT_SESSION)

    def _stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        session_key: object,
    ) -> Iterator[ModelStreamEvent]:
        if not self.api_key:
            raise OpenRouterError(
                "missing_api_key",
                f"set {self.api_key_env} before sending a message",
                env=self.api_key_env,
            )
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": self._wire_input(messages),
            "tools": [self._wire_tool(tool) for tool in tools],
            "parallel_tool_calls": True,
            "max_output_tokens": 8192,
            "store": False,
            "stream": True,
        }
        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}
        try:
            if self.client is None:
                with (
                    httpx.Client(timeout=self.timeout) as client,
                    self._active(
                        session_key,
                        client=client,
                        close_client_on_cancel=True,
                    ),
                ):
                    yield from self._stream_response(client, body, session_key)
            else:
                with self._active(session_key, client=self.client):
                    yield from self._stream_response(self.client, body, session_key)
        except (httpx.HTTPError, httpx.StreamError) as error:
            raise OpenRouterError(
                "openrouter_transport", f"OpenRouter request failed: {error}"
            ) from error

    def _stream_response(
        self, client: httpx.Client, body: dict[str, Any], session_key: object
    ) -> Iterator[ModelStreamEvent]:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        output_items: dict[int, dict[str, Any]] = {}
        total_tokens: int | None = None
        terminal_status: str | None = None
        terminal_finish_reason: str | None = None
        incomplete_reason: str | None = None
        saw_done = False
        stream_bytes = 0
        line_buffer = bytearray()
        deadline = time.monotonic() + self.timeout
        with (
            client.stream(
                "POST",
                f"{self.base_url}/responses",
                headers={
                    **self._headers(),
                    "Accept": "text/event-stream",
                },
                json=body,
            ) as response,
            self._active(session_key, response=response),
        ):
            if response.is_error:
                response.read()
                raise self._http_error(response)
            for chunk in response.iter_bytes():
                if time.monotonic() > deadline:
                    raise OpenRouterError(
                        "openrouter_timeout",
                        f"OpenRouter stream exceeded {self.timeout:g} seconds",
                    )
                stream_bytes += len(chunk)
                if stream_bytes > MAX_STREAM_BYTES:
                    raise OpenRouterError(
                        "openrouter_response_too_large",
                        "OpenRouter stream exceeded the response size limit",
                        limit=MAX_STREAM_BYTES,
                    )
                line_buffer.extend(chunk)
                while (newline := line_buffer.find(b"\n")) >= 0:
                    raw_line = bytes(line_buffer[:newline]).rstrip(b"\r")
                    del line_buffer[: newline + 1]
                    try:
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise OpenRouterError(
                            "openrouter_response",
                            "OpenRouter returned invalid UTF-8 stream data",
                        ) from error
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        payload = json.loads(data)
                        if not isinstance(payload, Mapping):
                            raise TypeError("stream event is not an object")
                        if payload.get("error") is not None:
                            raise self._stream_error(payload["error"])
                        event_type = payload.get("type")
                        if not isinstance(event_type, str):
                            raise TypeError("stream event type is invalid")
                        if terminal_status is not None and event_type not in {
                            "response.completed",
                            "response.done",
                        }:
                            raise OpenRouterError(
                                "openrouter_response",
                                "OpenRouter sent model data after stream completion",
                            )
                    except OpenRouterError:
                        raise
                    except (KeyError, TypeError, ValueError) as error:
                        raise OpenRouterError(
                            "openrouter_response",
                            "OpenRouter returned an invalid stream event",
                        ) from error
                    if event_type in {
                        "response.content_part.delta",
                        "response.output_text.delta",
                    }:
                        delta = payload.get("delta")
                        if not isinstance(delta, str):
                            raise OpenRouterError(
                                "openrouter_response",
                                "model returned invalid text delta",
                            )
                        if delta:
                            text_parts.append(delta)
                            yield ModelStreamEvent(text_delta=delta)
                    elif event_type == "response.reasoning.delta":
                        delta = payload.get("delta")
                        if not isinstance(delta, str):
                            raise OpenRouterError(
                                "openrouter_response",
                                "model returned invalid reasoning text",
                            )
                        reasoning_parts.append(delta)
                    elif event_type == "response.output_item.done":
                        output_index = payload.get("output_index")
                        item = payload.get("item")
                        if not isinstance(output_index, int) or not isinstance(
                            item, Mapping
                        ):
                            raise OpenRouterError(
                                "openrouter_response",
                                "OpenRouter returned an invalid output item",
                            )
                        item = dict(item)
                        if item.get("type") not in {
                            "function_call",
                            "reasoning",
                            "message",
                        }:
                            raise OpenRouterError(
                                "openrouter_response",
                                "OpenRouter returned an unsupported output item",
                            )
                        output_items[output_index] = item
                    elif event_type in {
                        "response.completed",
                        "response.done",
                        "response.failed",
                    }:
                        response_payload = payload.get("response")
                        if not isinstance(response_payload, Mapping):
                            raise OpenRouterError(
                                "openrouter_response",
                                "OpenRouter returned an invalid terminal response",
                            )
                        status = response_payload.get("status")
                        if not isinstance(status, str):
                            raise OpenRouterError(
                                "openrouter_response",
                                "OpenRouter returned an invalid response status",
                            )
                        if terminal_status is not None and status != terminal_status:
                            raise OpenRouterError(
                                "openrouter_response",
                                "OpenRouter changed the terminal response status",
                            )
                        terminal_status = status
                        error = response_payload.get("error")
                        if error is not None:
                            raise self._stream_error(error)
                        finish_reason = response_payload.get("finish_reason")
                        if finish_reason is not None and not isinstance(
                            finish_reason, str
                        ):
                            raise OpenRouterError(
                                "openrouter_response",
                                "OpenRouter returned an invalid finish reason",
                            )
                        terminal_finish_reason = finish_reason
                        details = response_payload.get("incomplete_details")
                        if isinstance(details, Mapping) and isinstance(
                            details.get("reason"), str
                        ):
                            incomplete_reason = details["reason"]
                        usage = response_payload.get("usage")
                        if isinstance(usage, Mapping):
                            reported_total = _positive_int(usage.get("total_tokens"))
                            if reported_total is not None:
                                total_tokens = reported_total
                if saw_done:
                    break

        if not saw_done:
            raise OpenRouterError(
                "openrouter_truncated", "OpenRouter stream ended before [DONE]"
            )
        if terminal_status != "completed" or terminal_finish_reason not in {
            None,
            "stop",
            "tool_calls",
        }:
            raise OpenRouterError(
                "openrouter_finish_reason",
                (
                    f"OpenRouter response ended with {terminal_status!r}"
                    + (
                        f": {incomplete_reason or terminal_finish_reason}"
                        if incomplete_reason or terminal_finish_reason
                        else ""
                    )
                ),
                finish_reason=(
                    incomplete_reason or terminal_finish_reason or terminal_status
                ),
            )
        ordered_items = tuple(item for _, item in sorted(output_items.items()))
        calls = tuple(
            self._tool_call(item)
            for item in ordered_items
            if item.get("type") == "function_call"
        )
        reasoning = "".join(reasoning_parts) or None
        message_items = [
            item for item in ordered_items if item.get("type") == "message"
        ]
        if not text_parts and message_items:
            try:
                completed_text = "".join(
                    self._response_message_text(item) for item in message_items
                )
            except TypeError as error:
                raise OpenRouterError(
                    "openrouter_response",
                    "OpenRouter returned an invalid response message",
                ) from error
            if completed_text:
                text_parts.append(completed_text)
        preserved_details = tuple(
            item for item in ordered_items if item.get("type") == "reasoning"
        )
        yield ModelStreamEvent(
            turn=ModelTurn(
                text="".join(text_parts) or None,
                calls=calls,
                reasoning=reasoning,
                reasoning_details=preserved_details,
                response_items=ordered_items,
                total_tokens=total_tokens,
            )
        )

    def _completion_response(
        self, client: httpx.Client, body: dict[str, Any], session_key: object
    ) -> str:
        deadline = time.monotonic() + self.timeout
        response_bytes = bytearray()
        with (
            client.stream(
                "POST",
                f"{self.base_url}/responses",
                headers={**self._headers(), "Accept": "application/json"},
                json=body,
            ) as response,
            self._active(session_key, response=response),
        ):
            if response.is_error:
                response.read()
                raise self._http_error(response)
            for chunk in response.iter_bytes():
                if time.monotonic() > deadline:
                    raise OpenRouterError(
                        "openrouter_timeout",
                        f"OpenRouter request exceeded {self.timeout:g} seconds",
                    )
                response_bytes.extend(chunk)
                if len(response_bytes) > MAX_STREAM_BYTES:
                    raise OpenRouterError(
                        "openrouter_response_too_large",
                        "OpenRouter response exceeded the response size limit",
                        limit=MAX_STREAM_BYTES,
                    )
        try:
            payload = json.loads(response_bytes)
            if not isinstance(payload, Mapping):
                raise TypeError("response is not an object")
            if payload.get("error") is not None:
                raise self._stream_error(payload["error"])
            if payload.get("status") != "completed" or payload.get(
                "finish_reason"
            ) not in {None, "stop"}:
                raise TypeError("completion did not complete normally")
            output = payload.get("output")
            if not isinstance(output, list):
                raise TypeError("response output is invalid")
            messages = []
            for item in output:
                if not isinstance(item, Mapping):
                    raise TypeError("response output item is invalid")
                if item.get("type") == "reasoning":
                    continue
                if item.get("type") != "message":
                    raise TypeError("completion returned non-text output")
                messages.append(item)
            if len(messages) != 1:
                raise TypeError("completion response message is invalid")
            content = self._response_message_text(messages[0])
        except (TypeError, ValueError) as error:
            raise OpenRouterError(
                "openrouter_response",
                "OpenRouter returned an invalid completion",
            ) from error
        return content

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "mechagnome",
        }

    @contextmanager
    def _active(
        self,
        session_key: object,
        *,
        client: httpx.Client | None = None,
        response: httpx.Response | None = None,
        close_client_on_cancel: bool = False,
    ) -> Iterator[None]:
        with self._active_lock:
            state = self._active_sessions.setdefault(session_key, _ActiveSession())
            if client is not None and close_client_on_cancel:
                state.clients[id(client)] = client
            if response is not None:
                state.responses[id(response)] = response
            cancel_requested = state.cancel_requested
        try:
            if cancel_requested:
                if response is not None:
                    with suppress(Exception):
                        response.close()
                if client is not None and close_client_on_cancel:
                    with suppress(Exception):
                        client.close()
                raise OpenRouterError(
                    "openrouter_cancelled", "OpenRouter request was cancelled"
                )
            yield
        finally:
            with self._active_lock:
                state = self._active_sessions.get(session_key)
                if state is not None:
                    if client is not None and state.clients.get(id(client)) is client:
                        state.clients.pop(id(client), None)
                    if (
                        response is not None
                        and state.responses.get(id(response)) is response
                    ):
                        state.responses.pop(id(response), None)
                    if (
                        not state.cancel_requested
                        and not state.clients
                        and not state.responses
                    ):
                        self._active_sessions.pop(session_key, None)

    @staticmethod
    def _stream_error(error: Any) -> OpenRouterError:
        message = error.get("message") if isinstance(error, Mapping) else None
        return OpenRouterError(
            "openrouter_stream",
            str(message or "OpenRouter returned a stream error"),
        )

    @staticmethod
    def _wire_tool(tool: dict[str, Any]) -> dict[str, Any]:
        parameters = tool["input_schema"]
        json_arguments = JSON_OBJECT_ARGUMENTS.get(tool["name"])
        if json_arguments:
            parameters = dict(parameters)
            properties = dict(parameters.get("properties", {}))
            for name, description in json_arguments.items():
                properties[name] = {
                    "type": "string",
                    "description": description,
                }
            parameters["properties"] = properties
        return {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": parameters,
        }

    @staticmethod
    def _to_wire_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
        wired = dict(args)
        for argument in JSON_OBJECT_ARGUMENTS.get(name, {}):
            value = wired.get(argument)
            if isinstance(value, dict):
                wired[argument] = json.dumps(value, separators=(",", ":"))
        return wired

    @staticmethod
    def _from_wire_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
        decoded = dict(args)
        for argument in JSON_OBJECT_ARGUMENTS.get(name, {}):
            value = decoded.get(argument)
            if not isinstance(value, str):
                continue
            try:
                decoded[argument] = json.loads(value)
            except json.JSONDecodeError:
                # Preserve malformed model output so the core operation can
                # return a normal repairable tool observation.
                pass
        return decoded

    @staticmethod
    def _wire_input(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Translate complete local history into stateless Responses input items."""
        wired: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            role = message["role"]
            if role == "assistant":
                response_items = message.get("response_items")
                if (
                    isinstance(response_items, Sequence)
                    and not isinstance(response_items, (str, bytes))
                    and response_items
                ):
                    wired.extend(
                        dict(item)
                        for item in response_items
                        if isinstance(item, Mapping) and item.get("type") != "reasoning"
                    )
                    continue
                calls = message.get("tool_calls") or ()
                wired.extend(
                    {
                        "type": "function_call",
                        "id": OpenRouterModel._item_id("fc", index, call),
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": json.dumps(
                            OpenRouterModel._to_wire_args(call["name"], call["args"])
                        ),
                    }
                    for call in calls
                )
                content = message.get("content")
                if isinstance(content, str) and content:
                    wired.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "id": OpenRouterModel._item_id("msg", index, message),
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": content,
                                    "annotations": [],
                                }
                            ],
                        }
                    )
            elif role == "tool":
                content = message.get("content")
                wired.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": (
                            content if isinstance(content, str) else json.dumps(content)
                        ),
                    }
                )
            else:
                content = message.get("content", "")
                wired.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    content
                                    if isinstance(content, str)
                                    else json.dumps(content)
                                ),
                            }
                        ],
                    }
                )
        return wired

    @staticmethod
    def _item_id(prefix: str, index: int, item: Mapping[str, Any]) -> str:
        seed = json.dumps([index, item], sort_keys=True, default=str)
        return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"

    @staticmethod
    def _tool_call(item: Any) -> ToolCall:
        try:
            if not isinstance(item, Mapping):
                raise TypeError("tool call is not an object")
            if item.get("type") not in {None, "function_call"}:
                raise TypeError("output item is not a function call")
            arguments = item.get("arguments") or "{}"
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(args, dict):
                raise TypeError("tool arguments are not an object")
            name = item["name"]
            return ToolCall(
                name=name,
                args=OpenRouterModel._from_wire_args(name, args),
                id=item.get("call_id") or item.get("id") or uuid.uuid4().hex,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise OpenRouterError(
                "openrouter_tool_call", "model returned an invalid tool call"
            ) from error

    @staticmethod
    def _response_message_text(item: Mapping[str, Any]) -> str:
        if item.get("type") != "message" or item.get("role") != "assistant":
            raise TypeError("response item is not an assistant message")
        content = item.get("content")
        if not isinstance(content, list):
            raise TypeError("response message content is invalid")
        parts: list[str] = []
        for part in content:
            if (
                not isinstance(part, Mapping)
                or part.get("type") != "output_text"
                or not isinstance(part.get("text"), str)
            ):
                raise TypeError("response message content is not text")
            parts.append(part["text"])
        return "".join(parts)

    @staticmethod
    def _http_error(response: httpx.Response) -> OpenRouterError:
        try:
            payload = response.json()
            provider_error = (
                payload.get("error", payload) if isinstance(payload, Mapping) else None
            )
            message = (
                provider_error.get("message")
                if isinstance(provider_error, dict)
                else None
            )
        except ValueError:
            message = None
        return OpenRouterError(
            "openrouter_http",
            message or f"OpenRouter returned HTTP {response.status_code}",
            status=response.status_code,
        )


class _SessionOpenRouterModel:
    """OpenRouter transport view bound to one conversation cancellation domain."""

    def __init__(self, model: OpenRouterModel, session_key: str) -> None:
        self._model = model
        self._session_key = session_key

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        return self._model._respond(messages, tools, self._session_key)

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Iterator[ModelStreamEvent]:
        yield from self._model._stream(messages, tools, self._session_key)

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        return self._model._complete(messages, self._session_key)

    def cancel_current(self) -> None:
        self._model._cancel_session(self._session_key)

    def reset_cancellation(self) -> None:
        self._model._reset_session(self._session_key)
