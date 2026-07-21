"""Unit tests for the bounded model-provider capability and socket proxy."""

from __future__ import annotations

import socket
import threading
from collections.abc import Mapping, Sequence

import pytest

from mechagnome import ToolboxError
from mechagnome.model_provider import (
    MAX_MODEL_CALLS,
    MAX_MODEL_RESPONSE_BYTES,
    _bind_model_provider,
    _encode_frame,
    _ModelProviderBroker,
    _ModelProviderProxy,
)


class RecordingProvider:
    def __init__(self, result: str = "completed") -> None:
        self.result = result
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.messages.append([dict(message) for message in messages])
        return self.result

    def cancel_current(self) -> None:
        pass

    def reset_cancellation(self) -> None:
        pass


def test_bounded_provider_validates_normalizes_and_counts_attempts() -> None:
    provider = RecordingProvider()
    bounded = _bind_model_provider(provider)

    invalid_messages = (
        [],
        [{"role": "tool", "content": "no"}],
        [{"role": "user", "content": "ok", "extra": "no"}],
        [{"role": ["user"], "content": "no"}],
    )
    for messages in invalid_messages:
        with pytest.raises(ToolboxError) as error:
            bounded.complete(messages)  # type: ignore[arg-type]
        assert error.value.code == "invalid_model_request"

    for _ in range(MAX_MODEL_CALLS - len(invalid_messages)):
        assert bounded.complete([{"role": "user", "content": "hello"}]) == ("completed")
    with pytest.raises(ToolboxError) as exhausted:
        bounded.complete([{"role": "user", "content": "hello"}])
    assert exhausted.value.code == "model_provider_limit"
    assert len(provider.messages) == MAX_MODEL_CALLS - len(invalid_messages)


def test_bounded_provider_sanitizes_failures_and_enforces_result_limit() -> None:
    secret = "sentinel-provider-secret"

    class FailingProvider:
        def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
            raise RuntimeError(secret)

    failing = _bind_model_provider(FailingProvider())
    with pytest.raises(ToolboxError) as failure:
        failing.complete([{"role": "user", "content": "hello"}])
    assert failure.value.code == "model_provider_failed"
    assert secret not in str(failure.value)

    oversized = _bind_model_provider(
        RecordingProvider("x" * (MAX_MODEL_RESPONSE_BYTES + 1))
    )
    with pytest.raises(ToolboxError) as too_large:
        oversized.complete([{"role": "user", "content": "hello"}])
    assert too_large.value.code == "model_provider_limit"


def test_escaped_text_can_reach_the_documented_frame_limit_early() -> None:
    escaped = "\\" * (MAX_MODEL_RESPONSE_BYTES // 2 + 4096)
    assert len(escaped.encode()) < MAX_MODEL_RESPONSE_BYTES

    with pytest.raises(ToolboxError) as error:
        _encode_frame({"ok": True, "text": escaped})

    assert error.value.code == "model_provider_protocol"


def test_socket_proxy_round_trips_without_serializing_provider() -> None:
    host, worker = socket.socketpair()
    provider = RecordingProvider("from host")
    broker = _ModelProviderBroker(host, provider)
    thread = threading.Thread(target=broker.serve)
    thread.start()
    proxy = _ModelProviderProxy(worker)
    try:
        assert proxy.complete([{"role": "user", "content": "hello"}]) == "from host"
    finally:
        worker.close()
        thread.join(timeout=1)

    assert thread.is_alive() is False
    assert provider.messages == [[{"role": "user", "content": "hello"}]]


def test_socket_proxy_rejects_partial_response_frame() -> None:
    host, worker = socket.socketpair()
    proxy = _ModelProviderProxy(worker)
    host.sendall(b"\x00\x00")
    host.close()
    try:
        with pytest.raises(ToolboxError) as error:
            proxy.complete([{"role": "user", "content": "hello"}])
        assert error.value.code == "model_provider_protocol"
    finally:
        worker.close()
