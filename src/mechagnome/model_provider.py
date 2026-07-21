"""Credential-opaque model-provider capability for authored tools."""

from __future__ import annotations

import json
import socket
import struct
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Any, Protocol

from mechagnome.kernel import ToolboxError

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


class ModelProvider(Protocol):
    """Cancellable synchronous text-completion service supplied by the host."""

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
    """Completion-only capability exposed to authored tools."""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return text for a validated chat-message sequence."""
        ...


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


class _BoundedModelProvider:
    def __init__(self, provider: _CompletionProvider) -> None:
        self._provider = provider
        self._calls = 0
        self._lock = Lock()

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        with self._lock:
            self._calls += 1
            if self._calls > MAX_MODEL_CALLS:
                raise _error("model_provider_limit")
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


def _bind_model_provider(
    provider: _CompletionProvider | None,
) -> _CompletionProvider:
    """Create a fresh per-call-tree facade around a host provider."""
    if provider is None:
        return _UnavailableModelProvider()
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

    def __init__(self, connection: socket.socket) -> None:
        self._connection = connection
        self._lock = Lock()

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        normalized = _normalized_messages(messages)
        with self._lock:
            _send_frame(
                self._connection,
                {"op": "complete", "messages": normalized},
            )
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


class _ModelProviderBroker:
    """Host-side broker retaining the concrete authenticated provider."""

    def __init__(self, connection: socket.socket, provider: ModelProvider) -> None:
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
        if set(request) != {"op", "messages"} or request.get("op") != "complete":
            return {"ok": False, "error": {"code": "model_provider_protocol"}}
        try:
            text = self._provider.complete(request["messages"])
        except ToolboxError as error:
            code = (
                error.code if error.code in _ERROR_MESSAGES else "model_provider_failed"
            )
            return {"ok": False, "error": {"code": code}}
        return {"ok": True, "text": text}
