"""OpenRouter adapter for the provider-neutral toolbox harness."""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib.resources import files
from threading import Event, Lock
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
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 0.5
_RETRY_AFTER_MAX_DELAY = 30.0
_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 429})
_RETRYABLE_ERROR_TYPES = frozenset(
    {
        "rate_limit_exceeded",
        "provider_overloaded",
        "provider_unavailable",
        "server",
        "server_error",
        "timeout",
        "unmapped",
    }
)
_ACTIONABLE_ERROR_TYPES = frozenset(
    {
        "authentication",
        "content_policy_violation",
        "context_length_exceeded",
        "invalid_prompt",
        "invalid_request",
        "max_tokens_exceeded",
        "payment_required",
        "permission_denied",
        "refusal",
        "string_too_long",
        "token_limit_exceeded",
    }
)
_KNOWN_ERROR_TYPES = _RETRYABLE_ERROR_TYPES | _ACTIONABLE_ERROR_TYPES
_SESSION_RECOVERY_ERROR_TYPES = frozenset(
    {"context_length_exceeded", "invalid_prompt", "string_too_long"}
)
_RETRYABLE_FINISH_REASONS = frozenset(
    {"error", "failed", "incomplete", "provider_error", "server_error"}
)
_PERMANENT_FINISH_REASONS = frozenset(
    {
        "content_filter",
        "content_policy",
        "length",
        "max_output_tokens",
        "max_tokens",
        "refusal",
        "safety",
    }
)
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


def _attempt_suffix(details: Mapping[str, Any]) -> str:
    attempts = details.get("attempts")
    if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts > 1:
        return f" after {attempts} attempts"
    return ""


def _public_error_guidance(
    code: str, details: Mapping[str, Any]
) -> tuple[str, tuple[str, ...]]:
    """Build provider-safe diagnostics and user actions from trusted fields."""
    suffix = _attempt_suffix(details)
    status = details.get("status")
    status = (
        status
        if isinstance(status, int)
        and not isinstance(status, bool)
        and 100 <= status <= 599
        else None
    )
    retryable = details.get("retryable") is True
    error_type = details.get("error_type")
    error_type = (
        error_type.lower()
        if isinstance(error_type, str) and error_type.lower() in _KNOWN_ERROR_TYPES
        else None
    )

    if error_type == "context_length_exceeded":
        return (
            "This session's history exceeds the selected model's context window.",
            (
                "Compact the session and continue; repeated retries in the same "
                "session will not help.",
            ),
        )
    if error_type in {"max_tokens_exceeded", "token_limit_exceeded"}:
        return (
            "OpenRouter stopped because the request exceeded a token limit.",
            ("Reduce the request size or output length, then retry.",),
        )
    if error_type == "string_too_long":
        return (
            "One item in the session is too large for the selected model.",
            ("Compact the session or start a new one, then retry.",),
        )
    if error_type in {"invalid_prompt", "invalid_request"}:
        return (
            "OpenRouter rejected an item in the session history.",
            (
                "Compact the session or start a new one; repeated retries will not "
                "help.",
            ),
        )
    if error_type == "authentication":
        return (
            "OpenRouter rejected the configured credentials.",
            (f"Check {DEFAULT_KEY_ENV} and the key's account access, then retry.",),
        )
    if error_type == "payment_required":
        return (
            "OpenRouter rejected the request because the account has insufficient "
            "credits.",
            ("Add credits or choose an available model, then retry.",),
        )
    if error_type == "permission_denied":
        return (
            "OpenRouter denied this request.",
            ("Check the key permissions and guardrails, or switch models.",),
        )
    if error_type in {"content_policy_violation", "refusal"}:
        return (
            "OpenRouter or the selected model blocked the request for content policy.",
            ("Rephrase the request or choose a model appropriate for the content.",),
        )

    if code == "missing_api_key":
        return (
            "OpenRouter cannot start because its API key is missing.",
            (f"Set {DEFAULT_KEY_ENV}, then retry the message.",),
        )
    if code == "openrouter_http":
        if status in {401, 403}:
            return (
                f"OpenRouter rejected the configured credentials (HTTP {status}).",
                (f"Check {DEFAULT_KEY_ENV} and the key's account access, then retry.",),
            )
        if status == 402:
            return (
                "OpenRouter rejected the request because the account has insufficient "
                "credits (HTTP 402).",
                ("Add credits or choose an available model, then retry.",),
            )
        if status == 404:
            return (
                "OpenRouter could not find the selected model or endpoint (HTTP 404).",
                ("Check the selected model name or switch models, then retry.",),
            )
        if status == 429:
            return (
                f"OpenRouter rate-limited the request (HTTP 429){suffix}.",
                (
                    "Retry the message later; if it continues, switch models or check "
                    "OpenRouter's status page.",
                ),
            )
        if status is not None and (status in {408, 409} or status >= 500):
            return (
                f"OpenRouter or its upstream provider returned a temporary HTTP "
                f"{status} error{suffix}.",
                (
                    "Retry the message; if it continues, switch models or check "
                    "OpenRouter's status page.",
                ),
            )
        status_text = f" (HTTP {status})" if status is not None else ""
        return (
            f"OpenRouter rejected the request{status_text}.",
            (
                "Check that the selected model supports the requested tools and "
                "reasoning settings, or switch models. If this repeats only in this "
                "session, compact it or start a new one.",
            ),
        )
    if code == "openrouter_transport":
        return (
            f"Mechagnome could not reach OpenRouter{suffix}.",
            (
                "Check the network connection and retry; if it continues, check "
                "OpenRouter's status page.",
            ),
        )
    if code == "openrouter_timeout":
        return (
            f"OpenRouter did not respond before the request timed out{suffix}.",
            ("Retry the message; if it continues, switch models.",),
        )
    if code == "openrouter_truncated":
        return (
            f"OpenRouter ended the response before it completed{suffix}.",
            ("Retry the message; if it continues, switch models.",),
        )
    if code == "openrouter_finish_reason":
        reason = details.get("finish_reason")
        normalized = reason.lower() if isinstance(reason, str) else None
        if normalized in {"length", "max_output_tokens", "max_tokens"}:
            return (
                "The model stopped because it reached its output limit.",
                ("Send 'continue', or split the request into smaller steps.",),
            )
        if normalized in {"content_filter", "content_policy", "refusal", "safety"}:
            return (
                "The model stopped because of a safety or content-policy decision.",
                (
                    "Rephrase the request or choose a model appropriate for the "
                    "content.",
                ),
            )
        return (
            f"OpenRouter's provider stopped unexpectedly{suffix}.",
            (
                "Retry the message; if it continues, switch models or check "
                "OpenRouter's status page.",
            ),
        )
    if code == "openrouter_stream":
        return (
            f"OpenRouter's response stream failed{suffix}.",
            (
                "Retry the message; if it continues, switch models or check "
                "OpenRouter's status page.",
            ),
        )
    if code in {
        "openrouter_models",
        "openrouter_response",
        "openrouter_response_too_large",
        "openrouter_tool_call",
    }:
        return (
            f"The selected model returned a response Mechagnome could not use{suffix}.",
            ("Retry once; if it continues, switch models and report the session ID.",),
        )
    if code == "openrouter_cancelled":
        return ("The OpenRouter request was cancelled.", ("Send the message again.",))

    message = _PUBLIC_ERROR_MESSAGES.get(code, "OpenRouter request failed")
    actions = ("Retry the message; if it continues, switch models.",)
    if not retryable:
        actions = ("Check the request and selected model, then retry.",)
    return (f"{message}{suffix}.", actions)


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

DEFAULT_SYSTEM_PROMPT = (
    files("mechagnome")
    .joinpath("assets", "default_system_prompt.txt")
    .read_text(encoding="utf-8")
)


class OpenRouterError(ModelTransportError):
    """A structured provider or response-shape failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(
            code,
            message,
            public_message=_PUBLIC_ERROR_MESSAGES.get(code),
            **details,
        )

    def public_error(self) -> dict[str, Any]:
        """Return sanitized diagnostics plus concrete recovery guidance."""
        message, actions = _public_error_guidance(self.code, self.details)
        if self.details.get("retry_suppressed") == "visible_output":
            actions = (
                "The response had already started, so Mechagnome did not retry it "
                "and risk duplicate output; send 'continue' to resume.",
            )
        details: dict[str, Any] = {}
        for name in ("attempts", "max_attempts", "status"):
            value = self.details.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                details[name] = value
        retryable = self.details.get("retryable")
        if isinstance(retryable, bool):
            details["retryable"] = retryable
        suppressed = self.details.get("retry_suppressed")
        if suppressed == "visible_output":
            details["retry_suppressed"] = suppressed
        error_type = self.details.get("error_type")
        if isinstance(error_type, str) and error_type.lower() in _KNOWN_ERROR_TYPES:
            normalized_type = error_type.lower()
            details["error_type"] = normalized_type
            if (
                normalized_type in _SESSION_RECOVERY_ERROR_TYPES
                and suppressed != "visible_output"
            ):
                details["recovery"] = "compact_session"
        return {
            "code": self.code,
            "message": " ".join((message, *actions)),
            "details": details,
        }


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
    cancelled: Event = field(default_factory=Event)
    operations: int = 0


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
        self._retry_sink: ContextVar[Callable[[dict[str, Any]], None] | None] = (
            ContextVar(f"openrouter_retry_sink_{id(self)}", default=None)
        )

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

    @contextmanager
    def capture_retries(self, sink: Callable[[dict[str, Any]], None]) -> Iterator[None]:
        """Route retry notices to the durable session owning this request."""
        token = self._retry_sink.set(sink)
        try:
            yield
        finally:
            self._retry_sink.reset(token)

    def _cancel_session(self, session_key: object) -> None:
        with self._active_lock:
            state = self._active_sessions.setdefault(session_key, _ActiveSession())
            state.cancelled.set()
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
            state.cancelled.clear()
            if not state.operations and not state.clients and not state.responses:
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
        deadline = time.monotonic() + self.timeout
        with self._active(session_key), self._request_client(session_key) as client:
            for retry_index in range(_MAX_RETRIES + 1):
                try:
                    return self._complete_once(client, body, session_key, deadline)
                except OpenRouterError as error:
                    self._raise_if_cancelled(session_key)
                    retryable = self._retryable(error)
                    if retry_index >= _MAX_RETRIES or not retryable:
                        self._finalize_error(
                            error,
                            attempts=retry_index + 1,
                            retryable=retryable,
                        )
                        raise
                    try:
                        self._wait_to_retry(error, retry_index, deadline, session_key)
                    except OpenRouterError as retry_error:
                        self._finalize_error(
                            retry_error,
                            attempts=retry_index + 1,
                            retryable=self._retryable(retry_error),
                        )
                        raise
        raise AssertionError("OpenRouter completion retry loop did not return")

    def _complete_once(
        self,
        client: httpx.Client,
        body: dict[str, Any],
        session_key: object,
        deadline: float,
    ) -> str:
        """Make one text-completion attempt inside a logical request."""
        try:
            return self._completion_response(client, body, session_key, deadline)
        except httpx.TransportError as error:
            raise OpenRouterError(
                "openrouter_transport",
                f"OpenRouter request failed: {error}",
                retryable=True,
            ) from error
        except (httpx.HTTPError, httpx.StreamError) as error:
            raise OpenRouterError(
                "openrouter_transport",
                f"OpenRouter request failed: {error}",
                retryable=False,
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
        deadline = time.monotonic() + self.timeout
        with self._active(session_key), self._request_client(session_key) as client:
            for retry_index in range(_MAX_RETRIES + 1):
                emitted_text = False
                try:
                    for event in self._stream_once(client, body, session_key, deadline):
                        if event.text_delta:
                            emitted_text = True
                        yield event
                    return
                except OpenRouterError as error:
                    self._raise_if_cancelled(session_key)
                    retryable = self._retryable(error)
                    if emitted_text or retry_index >= _MAX_RETRIES or not retryable:
                        self._finalize_error(
                            error,
                            attempts=retry_index + 1,
                            retryable=retryable,
                            retry_suppressed=(
                                "visible_output" if emitted_text else None
                            ),
                        )
                        raise
                    try:
                        self._wait_to_retry(error, retry_index, deadline, session_key)
                    except OpenRouterError as retry_error:
                        self._finalize_error(
                            retry_error,
                            attempts=retry_index + 1,
                            retryable=self._retryable(retry_error),
                        )
                        raise

    def _stream_once(
        self,
        client: httpx.Client,
        body: dict[str, Any],
        session_key: object,
        deadline: float,
    ) -> Iterator[ModelStreamEvent]:
        """Make one streaming attempt inside a logical request."""
        try:
            yield from self._stream_response(client, body, session_key, deadline)
        except httpx.TransportError as error:
            raise OpenRouterError(
                "openrouter_transport",
                f"OpenRouter request failed: {error}",
                retryable=True,
            ) from error
        except (httpx.HTTPError, httpx.StreamError) as error:
            raise OpenRouterError(
                "openrouter_transport",
                f"OpenRouter request failed: {error}",
                retryable=False,
            ) from error

    def _stream_response(
        self,
        client: httpx.Client,
        body: dict[str, Any],
        session_key: object,
        deadline: float,
    ) -> Iterator[ModelStreamEvent]:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        output_items: dict[int, dict[str, Any]] = {}
        total_tokens: int | None = None
        terminal_status: str | None = None
        terminal_finish_reason: str | None = None
        terminal_error_type: str | None = None
        incomplete_reason: str | None = None
        saw_done = False
        stream_bytes = 0
        line_buffer = bytearray()
        with (
            client.stream(
                "POST",
                f"{self.base_url}/responses",
                headers={
                    **self._headers(),
                    "Accept": "text/event-stream",
                },
                json=body,
                timeout=self._remaining_timeout(deadline),
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
                            raise self._stream_error(
                                payload["error"],
                                error_type=payload.get("error_type"),
                            )
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
                        "response.incomplete",
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
                            raise self._stream_error(
                                error,
                                error_type=response_payload.get("error_type"),
                            )
                        finish_reason = response_payload.get("finish_reason")
                        if finish_reason is not None and not isinstance(
                            finish_reason, str
                        ):
                            raise OpenRouterError(
                                "openrouter_response",
                                "OpenRouter returned an invalid finish reason",
                            )
                        terminal_finish_reason = finish_reason
                        error_type = response_payload.get("error_type")
                        if isinstance(error_type, str):
                            terminal_error_type = error_type
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
                status=terminal_status,
                **(
                    {"error_type": terminal_error_type}
                    if terminal_error_type is not None
                    else {}
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
        self,
        client: httpx.Client,
        body: dict[str, Any],
        session_key: object,
        deadline: float,
    ) -> str:
        response_bytes = bytearray()
        with (
            client.stream(
                "POST",
                f"{self.base_url}/responses",
                headers={**self._headers(), "Accept": "application/json"},
                json=body,
                timeout=self._remaining_timeout(deadline),
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
                raise self._stream_error(
                    payload["error"], error_type=payload.get("error_type")
                )
            status = payload.get("status")
            finish_reason = payload.get("finish_reason")
            if status is not None and not isinstance(status, str):
                raise TypeError("completion status is invalid")
            if finish_reason is not None and not isinstance(finish_reason, str):
                raise TypeError("completion finish reason is invalid")
            details = payload.get("incomplete_details")
            incomplete_reason = (
                details.get("reason")
                if isinstance(details, Mapping)
                and isinstance(details.get("reason"), str)
                else None
            )
            if status != "completed" or finish_reason not in {None, "stop"}:
                reason = incomplete_reason or finish_reason or status
                error_type = payload.get("error_type")
                raise OpenRouterError(
                    "openrouter_finish_reason",
                    f"OpenRouter returned an invalid completion: {reason}",
                    finish_reason=reason,
                    status=status,
                    **(
                        {"error_type": error_type}
                        if isinstance(error_type, str)
                        else {}
                    ),
                )
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
    def _request_client(self, session_key: object) -> Iterator[httpx.Client]:
        """Keep one HTTP client alive for every attempt in a logical request."""
        if self.client is not None:
            yield self.client
            return
        with (
            httpx.Client(timeout=self.timeout) as client,
            self._active(
                session_key,
                client=client,
                close_client_on_cancel=True,
            ),
        ):
            yield client

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
            state.operations += 1
            if client is not None and close_client_on_cancel:
                state.clients[id(client)] = client
            if response is not None:
                state.responses[id(response)] = response
            cancelled = state.cancelled
        try:
            if cancelled.is_set():
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
                    state.operations -= 1
                    if client is not None and state.clients.get(id(client)) is client:
                        state.clients.pop(id(client), None)
                    if (
                        response is not None
                        and state.responses.get(id(response)) is response
                    ):
                        state.responses.pop(id(response), None)
                    if (
                        not state.cancelled.is_set()
                        and not state.operations
                        and not state.clients
                        and not state.responses
                    ):
                        self._active_sessions.pop(session_key, None)

    @staticmethod
    def _stream_error(error: Any, *, error_type: Any = None) -> OpenRouterError:
        message = error.get("message") if isinstance(error, Mapping) else None
        normalized_type = error_type if isinstance(error_type, str) else None
        if isinstance(error, Mapping):
            for field_name in ("type", "code"):
                value = error.get(field_name)
                if isinstance(value, str):
                    normalized_type = normalized_type or value
        return OpenRouterError(
            "openrouter_stream",
            str(message or "OpenRouter returned a stream error"),
            **({"error_type": normalized_type} if normalized_type else {}),
        )

    def _raise_if_cancelled(self, session_key: object) -> None:
        with self._active_lock:
            state = self._active_sessions.get(session_key)
            cancelled = state is not None and state.cancelled.is_set()
        if cancelled:
            raise OpenRouterError(
                "openrouter_cancelled", "OpenRouter request was cancelled"
            )

    @staticmethod
    def _remaining_timeout(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OpenRouterError(
                "openrouter_timeout", "OpenRouter request exceeded its timeout"
            )
        return remaining

    @staticmethod
    def _retryable(error: OpenRouterError) -> bool:
        error_type = error.details.get("error_type")
        if isinstance(error_type, str) and error_type.lower() in _KNOWN_ERROR_TYPES:
            return error_type.lower() in _RETRYABLE_ERROR_TYPES
        if error.code == "openrouter_transport":
            return error.details.get("retryable") is True
        if error.code in {"openrouter_timeout", "openrouter_truncated"}:
            return True
        if error.code == "openrouter_http":
            status = error.details.get("status")
            return isinstance(status, int) and (
                status in _RETRYABLE_HTTP_STATUSES or status >= 500
            )
        if error.code == "openrouter_stream":
            error_type = error.details.get("error_type")
            return (
                isinstance(error_type, str)
                and error_type.lower() in _RETRYABLE_ERROR_TYPES
            )
        if error.code != "openrouter_finish_reason":
            return False
        reason = error.details.get("finish_reason")
        normalized_reason = reason.lower() if isinstance(reason, str) else None
        if normalized_reason in _PERMANENT_FINISH_REASONS:
            return False
        status = error.details.get("status")
        return status in {"failed", "incomplete"} or (
            normalized_reason in _RETRYABLE_FINISH_REASONS
        )

    @staticmethod
    def _retry_delay(error: OpenRouterError, retry_index: int) -> float:
        delay = _RETRY_BASE_DELAY * (2**retry_index)
        retry_after = error.details.get("retry_after")
        if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool):
            delay = max(delay, min(float(retry_after), _RETRY_AFTER_MAX_DELAY))
        return delay

    @staticmethod
    def _finalize_error(
        error: OpenRouterError,
        *,
        attempts: int,
        retryable: bool,
        retry_suppressed: str | None = None,
    ) -> None:
        """Annotate a terminal failure with safe retry diagnostics."""
        error.details["attempts"] = attempts
        error.details["max_attempts"] = _MAX_RETRIES + 1
        error.details["retryable"] = retryable
        if retry_suppressed is not None:
            error.details["retry_suppressed"] = retry_suppressed

    def _emit_retry(
        self, error: OpenRouterError, retry_index: int, delay: float
    ) -> None:
        sink = self._retry_sink.get()
        if sink is None:
            return
        attempt = retry_index + 2
        max_attempts = _MAX_RETRIES + 1
        reason, _ = _public_error_guidance(error.code, error.details)
        delay = round(delay, 3)
        timing = "immediately" if delay <= 0 else f"in {delay:g} seconds"
        sink(
            {
                "attempt": attempt,
                "max_attempts": max_attempts,
                "delay_seconds": delay,
                "code": error.code,
                "reason": reason,
                "action": (
                    f"Retrying automatically {timing} "
                    f"(attempt {attempt} of {max_attempts})."
                ),
            }
        )

    def _wait_to_retry(
        self,
        error: OpenRouterError,
        retry_index: int,
        deadline: float,
        session_key: object,
    ) -> None:
        self._raise_if_cancelled(session_key)
        remaining = self._remaining_timeout(deadline)
        delay = self._retry_delay(error, retry_index)
        if delay >= remaining:
            raise OpenRouterError(
                "openrouter_timeout", "OpenRouter retry window exceeded its timeout"
            )
        self._emit_retry(error, retry_index, delay)
        with self._active_lock:
            state = self._active_sessions.get(session_key)
            cancelled = state.cancelled if state is not None else None
        if cancelled is not None and cancelled.wait(delay):
            raise OpenRouterError(
                "openrouter_cancelled", "OpenRouter request was cancelled"
            )
        self._raise_if_cancelled(session_key)
        self._remaining_timeout(deadline)

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
        error_type: str | None = None
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
            candidates: tuple[Any, ...] = (
                payload.get("error_type") if isinstance(payload, Mapping) else None,
                (
                    provider_error.get("error_type")
                    if isinstance(provider_error, Mapping)
                    else None
                ),
                (
                    provider_error.get("code")
                    if isinstance(provider_error, Mapping)
                    else None
                ),
                (
                    provider_error.get("metadata", {}).get("error_type")
                    if isinstance(provider_error, Mapping)
                    and isinstance(provider_error.get("metadata"), Mapping)
                    else None
                ),
            )
            error_type = next(
                (
                    value.lower()
                    for value in candidates
                    if isinstance(value, str) and value.lower() in _KNOWN_ERROR_TYPES
                ),
                None,
            )
        except ValueError:
            message = None
        details: dict[str, Any] = {"status": response.status_code}
        if error_type is not None:
            details["error_type"] = error_type
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                seconds = float(retry_after)
            except ValueError:
                pass
            else:
                if math.isfinite(seconds) and seconds >= 0:
                    details["retry_after"] = min(seconds, _RETRY_AFTER_MAX_DELAY)
        return OpenRouterError(
            "openrouter_http",
            message or f"OpenRouter returned HTTP {response.status_code}",
            **details,
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

    @contextmanager
    def capture_retries(self, sink: Callable[[dict[str, Any]], None]) -> Iterator[None]:
        with self._model.capture_retries(sink):
            yield

    def cancel_current(self) -> None:
        self._model._cancel_session(self._session_key)

    def reset_cancellation(self) -> None:
        self._model._reset_session(self._session_key)
