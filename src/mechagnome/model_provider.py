"""Credential-opaque model-provider capability for authored tools."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol

from mechagnome.kernel import InvocationScope, Kernel, ToolboxError

MAX_MODEL_CALLS = 8
MAX_MODEL_MESSAGES = 64
MAX_MODEL_REQUEST_BYTES = 256 * 1024
MAX_MODEL_RESPONSE_BYTES = 1024 * 1024
_MAX_MODEL_FRAME_BYTES = MAX_MODEL_RESPONSE_BYTES + 4096

_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
_HEADER = struct.Struct("!I")
_ERROR_MESSAGES = {
    "model_provider_unavailable": "no model provider is available to this tool",
    "invalid_model_request": "model provider messages are invalid",
    "model_provider_limit": "model provider call limit exceeded",
    "model_provider_failed": "model provider request failed",
    "invalid_model_provider": "model provider must support cancellation",
    "model_provider_protocol": "model provider connection failed",
}


@dataclass(frozen=True)
class ToolCall:
    """One model-requested core operation."""

    name: str
    args: dict[str, Any]
    id: str


@dataclass(frozen=True)
class ModelTurn:
    """A provider-adapted model response."""

    text: str | None = None
    calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    reasoning: str | None = None
    reasoning_details: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    response_items: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    total_tokens: int | None = None


@dataclass(frozen=True)
class ModelStreamEvent:
    """One text delta or completed provider-neutral model turn."""

    text_delta: str = ""
    turn: ModelTurn | None = None


class ModelTransportError(ToolboxError):
    """A provider failure with a separately curated durable-session message."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        public_message: str | None = None,
        **details: Any,
    ) -> None:
        super().__init__(code, message, **details)
        self.public_message = public_message

    def public_error(self) -> dict[str, Any]:
        """Return provider-safe diagnostics for durable events and the UI."""
        if self.public_message is None:
            return _error("model_provider_failed").to_dict()["error"]
        return {
            "code": self.code,
            "message": self.public_message,
            "details": {},
        }


class ModelTransport(Protocol):
    """Raw provider transport hidden behind the session-aware gateway."""

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        """Return one provider-neutral model turn."""
        ...


class CompletionTransport(Protocol):
    """Raw synchronous text-completion surface used by child sessions."""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return text for a validated chat-message sequence."""
        ...

    def cancel_current(self) -> None:
        """Cause an active ``complete`` call to return promptly."""
        ...

    def reset_cancellation(self) -> None:
        """Make the provider reusable after cancellation."""
        ...


class _CompletionProvider(Protocol):
    """Internal model capability already bound to a durable parent session."""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return text for a validated chat-message sequence."""
        ...

    def run_agent(self, prompt: str) -> str:
        """Run a tool-capable child agent in a durable conversation session."""
        ...

    def for_origin(self, call_id: str) -> _CompletionProvider:
        """Bind descendants to the authored call that requested them."""
        ...

    def for_scope(self, scope: InvocationScope) -> _CompletionProvider:
        """Bind descendants to one frozen parent invocation scope."""
        ...

    def cancel_current(self) -> None:
        """Cancel active model work."""
        ...

    def reset_cancellation(self) -> None:
        """Make the capability reusable after cancellation."""
        ...

    @property
    def supports_cancellation(self) -> bool:
        """Whether active work can be cooperatively cancelled."""
        ...


def _bind_session_transport(transport: Any, session_id: str) -> Any:
    """Bind an opt-in transport to one conversation cancellation domain."""
    bind = getattr(transport, "for_session", None)
    return bind(session_id) if callable(bind) else transport


class ToolModelProvider:
    """Narrow model capability exposed to authored tools.

    Binding operations intentionally stay on the trusted invocation layer. A tool
    can request work, but cannot select its own parent session or origin call.
    """

    def __init__(self, capability: _BoundedModelProvider) -> None:
        self._capability = capability

    async def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Run a one-shot completion in a durable child session."""
        return await asyncio.to_thread(self._capability.complete, messages)

    async def run_agent(self, prompt: str) -> str:
        """Run a tool-capable agent in a durable child session."""
        return await asyncio.to_thread(self._capability.run_agent, prompt)


_USE_ROOT_TRANSPORT = object()


class ModelProvider:
    """Host-owned session gateway for every system model invocation."""

    def __init__(
        self,
        kernel: Kernel,
        transport: ModelTransport,
        *,
        completion_transport: CompletionTransport | None | object = _USE_ROOT_TRANSPORT,
        allow_tool_agents: bool = True,
    ) -> None:
        self.kernel = kernel
        self.transport = transport
        self.allow_tool_agents = allow_tool_agents
        if completion_transport is _USE_ROOT_TRANSPORT:
            candidate = transport
            self.completion_transport = (
                candidate if callable(getattr(candidate, "complete", None)) else None
            )
        else:
            self.completion_transport = completion_transport
        self._session_transport_lock = Lock()
        self._session_transports: dict[tuple[int, str], Any] = {}

    def _transport_for_session(self, transport: Any, session_id: str) -> Any:
        """Return one cached transport view per source and conversation."""
        if transport is None:
            return None
        key = (id(transport), session_id)
        with self._session_transport_lock:
            if key not in self._session_transports:
                self._session_transports[key] = _bind_session_transport(
                    transport, session_id
                )
            return self._session_transports[key]

    def _release_session(self, session_id: str) -> None:
        """Release cached transport views owned by one terminal conversation."""
        with self._session_transport_lock:
            keys = [key for key in self._session_transports if key[1] == session_id]
            for key in keys:
                self._session_transports.pop(key, None)

    @classmethod
    def from_transport(
        cls,
        kernel: Kernel,
        transport: ModelTransport,
        *,
        completions: CompletionTransport | None = None,
    ) -> ModelProvider:
        """Normalize legacy transports without implicitly granting tool spend."""
        return cls(
            kernel,
            transport,
            completion_transport=completions,
            allow_tool_agents=completions is not None,
        )

    @classmethod
    def from_completion_transport(
        cls, kernel: Kernel, transport: CompletionTransport
    ) -> ModelProvider:
        """Adapt a legacy tool-only transport at a host boundary."""
        return cls(
            kernel,
            _UnavailableConversationTransport(),
            completion_transport=transport,
            allow_tool_agents=False,
        )

    def start_session(
        self,
        *,
        session_id: str | None = None,
        parent_scope: InvocationScope | None = None,
        origin_call_id: str | None = None,
    ) -> ModelSession:
        """Create a root/child conversation or resume an existing root."""
        if parent_scope is not None:
            if session_id is not None:
                raise ToolboxError(
                    "invalid_session", "child session IDs are generated by the host"
                )
            identifier = self.kernel.create_child_session(
                parent_scope,
                kind="conversation",
                origin_call_id=origin_call_id,
            )
        else:
            identifier = self.kernel.create_session(session_id, kind="conversation")
            metadata = self.kernel.session_metadata(identifier)
            if metadata["kind"] != "conversation":
                raise ToolboxError(
                    "invalid_session",
                    f"a {metadata['kind']} session cannot be resumed as a conversation",
                )
        return ModelSession(self, identifier)

    def _complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        parent_scope: InvocationScope,
        origin_call_id: str | None,
    ) -> str:
        normalized = _normalized_messages(messages)
        child_id = self.kernel.create_child_session(
            parent_scope,
            kind="completion",
            origin_call_id=origin_call_id,
        )
        self.kernel.append_event(
            child_id,
            "model_input",
            {"messages": normalized},
        )
        completion_transport = self._transport_for_session(
            self.completion_transport, parent_scope.session_id
        )
        complete = (
            getattr(completion_transport, "complete", None)
            if completion_transport is not None
            else None
        )
        if not callable(complete):
            error = _error("model_provider_unavailable")
            self.kernel.append_event(child_id, "model_failed", error.to_dict()["error"])
            raise error
        try:
            result = complete(normalized)
            if not isinstance(result, str):
                raise _error("model_provider_failed")
            if len(result.encode("utf-8")) > MAX_MODEL_RESPONSE_BYTES:
                raise _error("model_provider_limit")
        except Exception as error:
            if isinstance(error, ModelTransportError):
                record = error.public_error()
                failure = _error("model_provider_failed")
            else:
                failure = (
                    error
                    if isinstance(error, ToolboxError) and error.code in _ERROR_MESSAGES
                    else _error("model_provider_failed")
                )
                record = failure.to_dict()["error"]
            self.kernel.append_event(child_id, "model_failed", record)
            raise failure from error
        self.kernel.append_event(
            child_id,
            "model",
            {"text": result, "calls": []},
        )
        self.kernel.append_event(child_id, "final", {"content": result})
        return result


class ModelSession:
    """A provider bound to one durable, potentially multi-turn agent session."""

    def __init__(self, provider: ModelProvider, session_id: str) -> None:
        self.provider = provider
        self.session_id = session_id
        self._closed = False
        self._bound_transports: dict[int, Any] = {}
        for transport in (provider.transport, provider.completion_transport):
            if transport is None or id(transport) in self._bound_transports:
                continue
            self._bound_transports[id(transport)] = provider._transport_for_session(
                transport, session_id
            )

    @property
    def transport(self) -> ModelTransport:
        return self._bound_transports[id(self.provider.transport)]

    @property
    def _completion_transport(self) -> CompletionTransport | None:
        transport = self.provider.completion_transport
        if transport is None:
            return None
        return self._bound_transports[id(transport)]

    @property
    def supports_detached_agents(self) -> bool:
        """Whether this conversation has an isolated cancellation binding."""
        if not self.provider.allow_tool_agents:
            return False
        if not callable(getattr(self.provider.transport, "for_session", None)):
            return False
        return all(
            callable(getattr(self.transport, name, None))
            for name in ("cancel_current", "reset_cancellation")
        )

    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        completed: ModelTurn | None = None
        for event in self.stream(messages, tools):
            if event.turn is not None:
                completed = event.turn
        assert completed is not None
        return completed

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        on_record: Callable[[str, dict[str, Any], int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[ModelStreamEvent]:
        yield from self._stream(
            messages,
            tools,
            on_record=on_record,
            cancelled=cancelled,
            input_recorded=False,
        )

    def _stream_with_recorded_input(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        on_record: Callable[[str, dict[str, Any], int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[ModelStreamEvent]:
        """Dispatch after the conversation has durably recorded its input."""
        yield from self._stream(
            messages,
            tools,
            on_record=on_record,
            cancelled=cancelled,
            input_recorded=True,
        )

    def _stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        on_record: Callable[[str, dict[str, Any], int], None] | None,
        cancelled: Callable[[], bool] | None,
        input_recorded: bool,
    ) -> Iterator[ModelStreamEvent]:
        if not input_recorded:
            self._record("model_input", {"messages": messages}, on_record)
        completed: ModelTurn | None = None
        try:
            stream = getattr(self.transport, "stream", None)
            events = (
                stream(messages, tools)
                if callable(stream)
                else iter(
                    (ModelStreamEvent(turn=self.transport.respond(messages, tools)),)
                )
            )
            for event in events:
                if event.turn is not None:
                    if completed is not None:
                        raise ToolboxError(
                            "invalid_model_stream",
                            "model stream completed more than once",
                        )
                    completed = event.turn
                else:
                    yield event
            if completed is None:
                raise ToolboxError(
                    "invalid_model_stream",
                    "model stream ended without a completed turn",
                )
            if cancelled is not None and cancelled():
                raise ToolboxError("cancelled", "rollout stopped")
        except Exception as error:
            is_cancelled = (
                isinstance(error, ToolboxError) and error.code == "cancelled"
            ) or (cancelled is not None and cancelled())
            if not is_cancelled:
                failure = (
                    error.public_error()
                    if isinstance(error, ModelTransportError)
                    else _error("model_provider_failed").to_dict()["error"]
                )
                self._record("model_failed", failure, on_record)
            raise

        payload = _model_payload(completed)
        self._record("model", payload, on_record)
        if not completed.calls:
            self._record("final", {"content": completed.text or ""}, on_record)
        yield ModelStreamEvent(turn=completed)

    def completion_provider(
        self,
        scope: InvocationScope | None = None,
        *,
        agent_runner: Callable[[InvocationScope, str | None, str], str] | None = None,
    ) -> _BoundedModelProvider:
        active_scope = (
            scope
            if scope is not None
            else self.provider.kernel.snapshot_scope(self.session_id)
        )
        return _bind_model_provider(
            _SessionCompletionProvider(
                self.provider,
                active_scope,
                agent_runner=agent_runner,
            )
        )

    def _record(
        self,
        kind: str,
        payload: dict[str, Any],
        sink: Callable[[str, dict[str, Any], int], None] | None,
    ) -> None:
        sequence = self.provider.kernel.append_event(self.session_id, kind, payload)
        if sink is not None:
            sink(kind, payload, sequence)

    def cancel_current(self) -> None:
        for target in self._bound_transports.values():
            cancel = getattr(target, "cancel_current", None)
            if callable(cancel):
                cancel()

    def reset_cancellation(self) -> None:
        for target in self._bound_transports.values():
            reset = getattr(target, "reset_cancellation", None)
            if callable(reset):
                reset()

    def close(self) -> None:
        """Release provider-side bindings for a terminal conversation."""
        if self._closed:
            return
        self._closed = True
        self.provider._release_session(self.session_id)


class _UnavailableConversationTransport:
    def respond(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        raise _error("model_provider_unavailable")


def _model_payload(turn: ModelTurn) -> dict[str, Any]:
    calls = [
        {"id": call.id, "name": call.name, "args": call.args} for call in turn.calls
    ]
    payload: dict[str, Any] = {"text": turn.text, "calls": calls}
    if turn.reasoning:
        payload["reasoning"] = turn.reasoning
    if turn.reasoning_details:
        payload["reasoning_details"] = list(turn.reasoning_details)
    if turn.response_items:
        payload["response_items"] = list(turn.response_items)
    if turn.total_tokens is not None:
        payload["total_tokens"] = turn.total_tokens
    return payload


class _SessionCompletionProvider:
    """Restricted child-session capability bound to trusted host context."""

    def __init__(
        self,
        provider: ModelProvider,
        scope: InvocationScope,
        origin_call_id: str | None = None,
        agent_runner: Callable[[InvocationScope, str | None, str], str] | None = None,
    ) -> None:
        self._provider = provider
        self._scope = scope
        self._origin_call_id = origin_call_id
        self._agent_runner = agent_runner

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        return self._provider._complete(
            messages,
            parent_scope=self._scope,
            origin_call_id=self._origin_call_id,
        )

    def for_origin(self, call_id: str) -> _SessionCompletionProvider:
        return _SessionCompletionProvider(
            self._provider,
            self._scope,
            call_id,
            self._agent_runner,
        )

    def for_scope(self, scope: InvocationScope) -> _SessionCompletionProvider:
        return _SessionCompletionProvider(
            self._provider,
            scope,
            self._origin_call_id,
            self._agent_runner,
        )

    def run_agent(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise _error("invalid_model_request")
        if self._agent_runner is None:
            raise _error("model_provider_unavailable")
        if not self._provider.allow_tool_agents:
            raise _error("model_provider_unavailable")
        return self._agent_runner(
            self._scope,
            self._origin_call_id,
            prompt,
        )

    def cancel_current(self) -> None:
        ModelSession(self._provider, self._scope.session_id).cancel_current()

    def reset_cancellation(self) -> None:
        ModelSession(self._provider, self._scope.session_id).reset_cancellation()

    @property
    def supports_cancellation(self) -> bool:
        session = ModelSession(self._provider, self._scope.session_id)
        targets = []
        if self._provider.completion_transport is not None:
            targets.append(session._completion_transport)
        if self._agent_runner is not None and self._provider.allow_tool_agents:
            targets.append(session.transport)
        return all(
            all(
                callable(getattr(target, name, None))
                for name in ("cancel_current", "reset_cancellation")
            )
            for target in targets
        )


def _error(code: str) -> ToolboxError:
    return ToolboxError(code, _ERROR_MESSAGES[code])


def _normalized_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise _error("invalid_model_request")
    if not 1 <= len(messages) <= MAX_MODEL_MESSAGES:
        raise _error("invalid_model_request")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise _error("invalid_model_request")
        role = message.get("role")
        content = message.get("content")
        if (
            not isinstance(role, str)
            or role not in _ALLOWED_ROLES
            or not isinstance(content, str)
        ):
            raise _error("invalid_model_request")
        normalized.append({"role": role, "content": content})
    try:
        encoded = json.dumps(
            {"messages": normalized},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise _error("invalid_model_request") from error
    if len(encoded) > MAX_MODEL_REQUEST_BYTES:
        raise _error("invalid_model_request")
    return normalized


class _UnavailableModelProvider:
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        raise _error("model_provider_unavailable")

    def run_agent(self, prompt: str) -> str:
        raise _error("model_provider_unavailable")

    def for_origin(self, call_id: str) -> _UnavailableModelProvider:
        return self

    def for_scope(self, scope: InvocationScope) -> _UnavailableModelProvider:
        return self

    def cancel_current(self) -> None:
        pass

    def reset_cancellation(self) -> None:
        pass

    @property
    def supports_cancellation(self) -> bool:
        return True


class _BoundedModelProvider:
    def __init__(
        self,
        provider: _CompletionProvider,
        *,
        budget: _ModelCallBudget | None = None,
    ) -> None:
        self._provider = provider
        self._budget = budget or _ModelCallBudget()

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        self._budget.consume()
        normalized = _normalized_messages(messages)
        try:
            result = self._provider.complete(normalized)
        except ToolboxError as error:
            code = (
                error.code if error.code in _ERROR_MESSAGES else "model_provider_failed"
            )
            raise _error(code) from error
        except Exception as error:
            raise _error("model_provider_failed") from error
        if not isinstance(result, str):
            raise _error("model_provider_failed")
        try:
            size = len(result.encode("utf-8"))
        except UnicodeError as error:
            raise _error("model_provider_failed") from error
        if size > MAX_MODEL_RESPONSE_BYTES:
            raise _error("model_provider_limit")
        return result

    def run_agent(self, prompt: str) -> str:
        self._budget.consume()
        if not isinstance(prompt, str) or not prompt:
            raise _error("invalid_model_request")
        try:
            result = self._provider.run_agent(prompt)
        except ToolboxError as error:
            code = (
                error.code if error.code in _ERROR_MESSAGES else "model_provider_failed"
            )
            raise _error(code) from error
        except Exception as error:
            raise _error("model_provider_failed") from error
        if not isinstance(result, str):
            raise _error("model_provider_failed")
        return result

    def for_origin(self, call_id: str) -> _BoundedModelProvider:
        provider = self._provider.for_origin(call_id)
        return _BoundedModelProvider(provider, budget=self._budget)

    def for_scope(self, scope: InvocationScope) -> _BoundedModelProvider:
        provider = self._provider.for_scope(scope)
        return _BoundedModelProvider(provider, budget=self._budget)

    def cancel_current(self) -> None:
        self._provider.cancel_current()

    def reset_cancellation(self) -> None:
        self._provider.reset_cancellation()

    @property
    def supports_cancellation(self) -> bool:
        return self._provider.supports_cancellation


class _ModelCallBudget:
    def __init__(self) -> None:
        self._calls = 0
        self._lock = Lock()

    def consume(self) -> None:
        with self._lock:
            self._calls += 1
            if self._calls > MAX_MODEL_CALLS:
                raise _error("model_provider_limit")


def _bind_model_provider(
    provider: _CompletionProvider | None,
) -> _BoundedModelProvider:
    """Create a fresh per-call-tree facade around a host provider."""
    if isinstance(provider, _BoundedModelProvider):
        return provider
    if provider is None:
        provider = _UnavailableModelProvider()
    return _BoundedModelProvider(provider)


def _encode_frame(payload: dict[str, Any]) -> bytes:
    try:
        data = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise _error("model_provider_protocol") from error
    if len(data) > _MAX_MODEL_FRAME_BYTES:
        raise _error("model_provider_protocol")
    return _HEADER.pack(len(data)) + data


def _send_frame(connection: socket.socket, payload: dict[str, Any]) -> None:
    try:
        connection.sendall(_encode_frame(payload))
    except OSError as error:
        raise _error("model_provider_protocol") from error


def _receive_exact(
    connection: socket.socket, size: int, *, allow_initial_eof: bool = False
) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        try:
            chunk = connection.recv(size - len(chunks))
        except OSError as error:
            raise _error("model_provider_protocol") from error
        if not chunk:
            if allow_initial_eof and not chunks:
                return None
            raise _error("model_provider_protocol")
        chunks.extend(chunk)
    return bytes(chunks)


def _receive_frame(
    connection: socket.socket, *, allow_initial_eof: bool = False
) -> dict[str, Any] | None:
    header = _receive_exact(
        connection,
        _HEADER.size,
        allow_initial_eof=allow_initial_eof,
    )
    if header is None:
        return None
    (size,) = _HEADER.unpack(header)
    if size > _MAX_MODEL_FRAME_BYTES:
        raise _error("model_provider_protocol")
    data = _receive_exact(connection, size)
    assert data is not None
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise _error("model_provider_protocol") from error
    if not isinstance(payload, dict):
        raise _error("model_provider_protocol")
    return payload


class _ModelProviderProxy:
    """Worker-local completion capability backed by a host socket."""

    def __init__(
        self,
        connection: socket.socket,
        *,
        origin_call_id: str | None = None,
        lock: Lock | None = None,
    ) -> None:
        self._connection = connection
        self._origin_call_id = origin_call_id
        self._lock = lock or Lock()

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        return self._request("complete", messages=messages)

    def run_agent(self, prompt: str) -> str:
        return self._request("run_agent", prompt=prompt)

    def _request(
        self,
        operation: str,
        *,
        messages: Sequence[Mapping[str, str]] | None = None,
        prompt: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "op": operation,
            "origin_call_id": self._origin_call_id,
        }
        if operation == "complete":
            assert messages is not None
            payload["messages"] = _normalized_messages(messages)
        else:
            if not isinstance(prompt, str) or not prompt:
                raise _error("invalid_model_request")
            payload["prompt"] = prompt
        with self._lock:
            _send_frame(self._connection, payload)
            response = _receive_frame(self._connection)
        if response is None or response.get("ok") not in {True, False}:
            raise _error("model_provider_protocol")
        if response["ok"] is True:
            if set(response) != {"ok", "text"} or not isinstance(
                response.get("text"), str
            ):
                raise _error("model_provider_protocol")
            return response["text"]
        if set(response) != {"ok", "error"} or not isinstance(
            response.get("error"), dict
        ):
            raise _error("model_provider_protocol")
        error_payload = response["error"]
        code = error_payload.get("code")
        if set(error_payload) != {"code"} or code not in _ERROR_MESSAGES:
            raise _error("model_provider_protocol")
        raise _error(str(code))

    def for_origin(self, call_id: str) -> _ModelProviderProxy:
        return _ModelProviderProxy(
            self._connection,
            origin_call_id=call_id,
            lock=self._lock,
        )

    def for_scope(self, scope: InvocationScope) -> _ModelProviderProxy:
        """Keep the scope fixed by the host-side broker binding."""
        return self

    def cancel_current(self) -> None:
        pass

    def reset_cancellation(self) -> None:
        pass

    @property
    def supports_cancellation(self) -> bool:
        return True


class _ModelProviderBroker:
    """Host-side broker retaining the concrete authenticated provider."""

    def __init__(
        self, connection: socket.socket, provider: _CompletionProvider
    ) -> None:
        self._connection = connection
        self._provider = _bind_model_provider(provider)

    def serve(self) -> None:
        """Serve requests until the worker closes its endpoint."""
        try:
            while True:
                request = _receive_frame(
                    self._connection,
                    allow_initial_eof=True,
                )
                if request is None:
                    return
                response = self._handle(request)
                _send_frame(self._connection, response)
        except ToolboxError:
            return
        finally:
            self._connection.close()

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        expected = (
            {"op", "messages", "origin_call_id"}
            if operation == "complete"
            else {"op", "prompt", "origin_call_id"}
            if operation == "run_agent"
            else set()
        )
        if set(request) != expected or (
            request.get("origin_call_id") is not None
            and not isinstance(request.get("origin_call_id"), str)
        ):
            return {"ok": False, "error": {"code": "model_provider_protocol"}}
        try:
            provider = self._provider
            origin_call_id = request["origin_call_id"]
            if origin_call_id is not None:
                provider = provider.for_origin(origin_call_id)
            text = (
                provider.complete(request["messages"])
                if operation == "complete"
                else provider.run_agent(request["prompt"])
            )
        except ToolboxError as error:
            code = (
                error.code if error.code in _ERROR_MESSAGES else "model_provider_failed"
            )
            return {"ok": False, "error": {"code": code}}
        return {"ok": True, "text": text}
