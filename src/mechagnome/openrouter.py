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

from mechagnome.kernel import ToolboxError
from mechagnome.model_provider import ModelStreamEvent, ModelTurn, ToolCall

DEFAULT_MODEL = "z-ai/glm-5.2"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"
MAX_STREAM_BYTES = 4_000_000
MAX_COMPLETION_TOKENS = 2048
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_DEFAULT_SESSION = object()

# Open-ended nested objects are not represented consistently by every model or
# provider behind an OpenAI-compatible tool-calling endpoint. Keep the kernel's
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
You can only directly call help, search_tools, view_tool, write_tool, and
call_tool. The session's selected toolbox namespaces begin with no domain-specific
user tools. Search before
creating duplicates; build small reusable Python tools when they improve the
task; call and repair them immediately; and reuse them across later requests.
Request independent operations together in modest batches. Keep an operation
in a later turn when it depends on the output of an earlier operation.
Tools may call other tools through ctx.call_tool, read current or historical
sessions through ctx.sessions, and use the ordinary Linux/Python environment.
Use help when you need the tool ABI or examples. Core operation source is also
readable and editable, but keep changes deliberate. When the user's task is
complete, return a concise final answer instead of making another tool call.
"""


class OpenRouterError(ToolboxError):
    """A structured provider or response-shape failure."""


@dataclass(frozen=True)
class OpenRouterModelOption:
    """One tool-capable model exposed by OpenRouter's model catalog."""

    id: str
    name: str
    input_modalities: tuple[str, ...] = ()
    reasoning_efforts: tuple[str, ...] = ()
    reasoning_mandatory: bool = False


@dataclass
class _ActiveSession:
    """Cancelable OpenRouter resources active in one root session."""

    clients: dict[int, httpx.Client] = field(default_factory=dict)
    responses: dict[int, httpx.Response] = field(default_factory=dict)
    cancel_requested: bool = False


class OpenRouterModel:
    """Streaming OpenRouter Chat Completions model adapter."""

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

    def for_session(self, root_session_id: str) -> _SessionOpenRouterModel:
        """Return a view whose cancellation is isolated to one durable root."""
        return _SessionOpenRouterModel(self, root_session_id)

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
            "messages": [dict(message) for message in messages],
            "max_tokens": MAX_COMPLETION_TOKENS,
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
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *self._wire_messages(messages),
            ],
            "tools": [self._wire_tool(tool) for tool in tools],
            "parallel_tool_calls": True,
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
        reasoning_details: list[dict[str, Any]] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        saw_done = False
        stream_bytes = 0
        line_buffer = bytearray()
        deadline = time.monotonic() + self.timeout
        with (
            client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
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
                        choices = payload.get("choices")
                        if choices == []:
                            continue
                        if not isinstance(choices, list) or len(choices) != 1:
                            raise TypeError("stream choices are invalid")
                        choice = choices[0]
                        if not isinstance(choice, Mapping):
                            raise TypeError("stream choice is not an object")
                        delta = choice["delta"]
                        if not isinstance(delta, Mapping):
                            raise TypeError("stream delta is not an object")
                        reason = choice.get("finish_reason")
                        if reason is not None and not isinstance(reason, str):
                            raise TypeError("finish reason is not a string")
                    except OpenRouterError:
                        raise
                    except (KeyError, IndexError, TypeError, ValueError) as error:
                        raise OpenRouterError(
                            "openrouter_response",
                            "OpenRouter returned an invalid stream event",
                        ) from error
                    text = self._text_content(delta.get("content"))
                    raw_reasoning = delta.get("reasoning")
                    if raw_reasoning is None:
                        reasoning = ""
                    elif isinstance(raw_reasoning, str):
                        reasoning = raw_reasoning
                    else:
                        raise OpenRouterError(
                            "openrouter_response",
                            "model returned invalid reasoning text",
                        )
                    raw_reasoning_details = delta.get("reasoning_details")
                    if raw_reasoning_details is None:
                        detail_deltas: list[dict[str, Any]] = []
                    elif isinstance(raw_reasoning_details, list) and all(
                        isinstance(detail, Mapping) for detail in raw_reasoning_details
                    ):
                        detail_deltas = [
                            dict(detail) for detail in raw_reasoning_details
                        ]
                    else:
                        raise OpenRouterError(
                            "openrouter_response",
                            "model returned invalid reasoning details",
                        )
                    raw_tool_deltas = delta.get("tool_calls")
                    if raw_tool_deltas is None:
                        tool_deltas = []
                    elif isinstance(raw_tool_deltas, list):
                        tool_deltas = raw_tool_deltas
                    else:
                        raise OpenRouterError(
                            "openrouter_tool_call",
                            "model returned invalid tool-call deltas",
                        )
                    if finish_reason is not None:
                        if text or reasoning or detail_deltas or tool_deltas:
                            raise OpenRouterError(
                                "openrouter_response",
                                "OpenRouter sent model data after stream completion",
                            )
                        if reason not in {None, finish_reason}:
                            raise OpenRouterError(
                                "openrouter_response",
                                "OpenRouter changed the terminal finish reason",
                            )
                        continue
                    if text:
                        text_parts.append(text)
                        yield ModelStreamEvent(text_delta=text)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    reasoning_details.extend(detail_deltas)
                    self._merge_tool_deltas(tool_calls, tool_deltas)
                    if reason is not None:
                        finish_reason = reason
                if saw_done:
                    break

        if not saw_done:
            raise OpenRouterError(
                "openrouter_truncated", "OpenRouter stream ended before [DONE]"
            )
        expected_reason = "tool_calls" if tool_calls else "stop"
        if finish_reason != expected_reason:
            raise OpenRouterError(
                "openrouter_finish_reason",
                (
                    f"OpenRouter stream ended with {finish_reason!r}, "
                    f"expected {expected_reason!r}"
                ),
                finish_reason=finish_reason,
            )
        calls = tuple(
            self._tool_call(
                {
                    "id": item.get("id"),
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments") or "{}",
                    },
                }
            )
            for _, item in sorted(tool_calls.items())
        )
        yield ModelStreamEvent(
            turn=ModelTurn(
                text="".join(text_parts) or None,
                calls=calls,
                reasoning="".join(reasoning_parts) or None,
                reasoning_details=tuple(reasoning_details),
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
                f"{self.base_url}/chat/completions",
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
            choices = payload.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise TypeError("response choices are invalid")
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise TypeError("response choice is not an object")
            if choice.get("finish_reason") not in {None, "stop"}:
                raise TypeError("completion did not stop normally")
            message = choice.get("message")
            if not isinstance(message, Mapping):
                raise TypeError("response message is not an object")
            if message.get("tool_calls"):
                raise TypeError("completion returned tool calls")
            content = message.get("content")
            if not isinstance(content, str):
                raise TypeError("completion content is not text")
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
            if cancel_requested:
                state.cancel_requested = False
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
    def _merge_tool_deltas(accumulated: dict[int, dict[str, Any]], deltas: Any) -> None:
        if not isinstance(deltas, list):
            raise OpenRouterError(
                "openrouter_tool_call", "model returned invalid tool-call deltas"
            )
        for delta in deltas:
            if not isinstance(delta, Mapping) or not isinstance(
                delta.get("index"), int
            ):
                raise OpenRouterError(
                    "openrouter_tool_call", "model returned an invalid tool-call delta"
                )
            item = accumulated.setdefault(
                delta["index"], {"id": "", "name": "", "arguments": ""}
            )
            identifier = delta.get("id")
            if identifier:
                item["id"] += str(identifier)
            function = delta.get("function")
            if function is None:
                continue
            if not isinstance(function, Mapping):
                raise OpenRouterError(
                    "openrouter_tool_call", "model returned an invalid tool-call delta"
                )
            if function.get("name"):
                item["name"] += str(function["name"])
            if function.get("arguments"):
                item["arguments"] += str(function["arguments"])

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
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": parameters,
            },
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
    def _wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        wired = []
        for message in messages:
            role = message["role"]
            if role == "assistant":
                item: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content"),
                }
                calls = message.get("tool_calls") or ()
                if calls:
                    item["tool_calls"] = [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(
                                    OpenRouterModel._to_wire_args(
                                        call["name"], call["args"]
                                    )
                                ),
                            },
                        }
                        for call in calls
                    ]
                reasoning = message.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    item["reasoning"] = reasoning
                reasoning_details = message.get("reasoning_details")
                if reasoning_details:
                    item["reasoning_details"] = reasoning_details
                wired.append(item)
            elif role == "tool":
                content = message.get("content")
                wired.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "name": message.get("name"),
                        "content": (
                            content if isinstance(content, str) else json.dumps(content)
                        ),
                    }
                )
            else:
                wired.append({"role": role, "content": message.get("content", "")})
        return wired

    @staticmethod
    def _tool_call(item: Any) -> ToolCall:
        try:
            if not isinstance(item, Mapping):
                raise TypeError("tool call is not an object")
            function = item["function"]
            if not isinstance(function, Mapping):
                raise TypeError("tool function is not an object")
            arguments = function.get("arguments") or "{}"
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(args, dict):
                raise TypeError("tool arguments are not an object")
            name = function["name"]
            return ToolCall(
                name=name,
                args=OpenRouterModel._from_wire_args(name, args),
                id=item.get("id") or uuid.uuid4().hex,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise OpenRouterError(
                "openrouter_tool_call", "model returned an invalid tool call"
            ) from error

    @staticmethod
    def _text_content(content: Any) -> str | None:
        if content is None or isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "".join(parts) or None
        return str(content)

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
    """OpenRouter transport view bound to one root cancellation domain."""

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
