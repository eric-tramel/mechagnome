"""OpenRouter adapter for the provider-neutral toolbox harness."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from threading import Lock
from typing import Any

import httpx

from mechagnome.harness import ModelStreamEvent, ModelTurn, ToolCall
from mechagnome.kernel import ToolboxError

DEFAULT_MODEL = "z-ai/glm-5.2"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"
MAX_STREAM_BYTES = 4_000_000

DEFAULT_SYSTEM_PROMPT = """\
You are the agent inside Mechagnome, a persistent metaprogrammable toolbox.
You can only directly call help, search_tools, read_tool_source, write_tool, and
call_tool. The toolbox begins with no domain-specific user tools. Search before
creating duplicates; build small reusable Python tools when they improve the
task; call and repair them immediately; and reuse them across later requests.
Tools may call other tools through ctx.call_tool, read current or historical
sessions through ctx.sessions, and use the ordinary Linux/Python environment.
Use help when you need the tool ABI or examples. Core operation source is also
readable and editable, but keep changes deliberate. When the user's task is
complete, return a concise final answer instead of making another tool call.
"""


class OpenRouterError(ToolboxError):
    """A structured provider or response-shape failure."""


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
        self.base_url = DEFAULT_BASE_URL
        self.api_key_env = DEFAULT_KEY_ENV
        self.api_key = api_key or os.environ.get(DEFAULT_KEY_ENV)
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.client = client
        self._active_lock = Lock()
        self._active_client: httpx.Client | None = None
        self._active_response: httpx.Response | None = None
        self._cancel_requested = False

    @property
    def ready(self) -> bool:
        """Whether the configured API-key environment is available."""
        return bool(self.api_key)

    def cancel_current(self) -> None:
        """Close the active streaming response so a blocked read wakes promptly."""
        with self._active_lock:
            self._cancel_requested = True
            response = self._active_response
            client = self._active_client
        if response is not None:
            with suppress(Exception):
                response.close()
        if client is not None:
            with suppress(Exception):
                client.close()

    def reset_cancellation(self) -> None:
        """Discard a cancellation latch when its conversation rollout has ended."""
        with self._active_lock:
            self._cancel_requested = False

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        """Consume a streaming OpenRouter response into one model turn."""
        completed: ModelTurn | None = None
        for event in self.stream(messages, tools):
            if event.turn is not None:
                completed = event.turn
        if completed is None:
            raise OpenRouterError(
                "openrouter_response", "OpenRouter stream ended without a completion"
            )
        return completed

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Iterator[ModelStreamEvent]:
        """Yield OpenRouter text deltas and one assembled final turn."""
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
            "parallel_tool_calls": False,
            "stream": True,
        }
        try:
            if self.client is None:
                with (
                    httpx.Client(timeout=self.timeout) as client,
                    self._active(client=client),
                ):
                    yield from self._stream_response(client, body)
            else:
                with self._active(client=self.client):
                    yield from self._stream_response(self.client, body)
        except (httpx.HTTPError, httpx.StreamError) as error:
            raise OpenRouterError(
                "openrouter_transport", f"OpenRouter request failed: {error}"
            ) from error

    def _stream_response(
        self, client: httpx.Client, body: dict[str, Any]
    ) -> Iterator[ModelStreamEvent]:
        text_parts: list[str] = []
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
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "mechagnome",
                },
                json=body,
            ) as response,
            self._active(response=response),
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
                        if text or tool_deltas:
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
            turn=ModelTurn(text="".join(text_parts) or None, calls=calls)
        )

    @contextmanager
    def _active(
        self,
        *,
        client: httpx.Client | None = None,
        response: httpx.Response | None = None,
    ) -> Iterator[None]:
        with self._active_lock:
            if client is not None:
                self._active_client = client
            if response is not None:
                self._active_response = response
            cancel_requested = self._cancel_requested
            if cancel_requested:
                self._cancel_requested = False
        try:
            if cancel_requested:
                if response is not None:
                    with suppress(Exception):
                        response.close()
                if client is not None:
                    with suppress(Exception):
                        client.close()
                raise OpenRouterError(
                    "openrouter_cancelled", "OpenRouter request was cancelled"
                )
            yield
        finally:
            with self._active_lock:
                if client is not None and self._active_client is client:
                    self._active_client = None
                if response is not None and self._active_response is response:
                    self._active_response = None

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
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }

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
                                "arguments": json.dumps(call["args"]),
                            },
                        }
                        for call in calls
                    ]
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
            return ToolCall(
                name=function["name"],
                args=args,
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
