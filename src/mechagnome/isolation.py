"""Filtered-environment subprocess boundary for authored tool execution."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from threading import Thread
from typing import Any

from mechagnome.kernel import JsonValue, Kernel, ToolboxError
from mechagnome.model_provider import ModelProvider, _ModelProviderBroker

CommittedEventSink = Callable[[dict[str, Any]], None]

_SAFE_ENVIRONMENT = (
    "GIT_ASKPASS",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_SYSTEM",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_SSH_VARIANT",
    "GIT_TERMINAL_PROMPT",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "SSH_ASKPASS",
    "SSH_ASKPASS_REQUIRE",
    "SSH_AUTH_SOCK",
    "TERM",
    "TMPDIR",
    "TZ",
    "XDG_CONFIG_HOME",
)


def _is_safe_environment_variable(name: str) -> bool:
    """Whether a host variable is required for basic runtime or Git access."""
    if name in _SAFE_ENVIRONMENT:
        return True
    for prefix in ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"):
        if name.startswith(prefix):
            return name.removeprefix(prefix).isdecimal()
    return False


class IsolatedToolRunner:
    """Run an entire editable tool call tree outside the provider process."""

    def __init__(self, kernel: Kernel, *, timeout: float = 120.0) -> None:
        self.kernel = kernel
        self.timeout = timeout

    def call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session_id: str,
        on_event: CommittedEventSink | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> JsonValue:
        """Execute one call and relay its committed events while it runs."""
        return self._call(
            name,
            args,
            session_id=session_id,
            on_event=on_event,
            cancelled=cancelled,
            model_provider=None,
        )

    def call_with_model_provider(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session_id: str,
        model_provider: ModelProvider | None,
        on_event: CommittedEventSink | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> JsonValue:
        """Execute one call with a host-brokered model provider capability."""
        return self._call(
            name,
            args,
            session_id=session_id,
            on_event=on_event,
            cancelled=cancelled,
            model_provider=model_provider,
        )

    def _call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session_id: str,
        on_event: CommittedEventSink | None,
        cancelled: Callable[[], bool] | None,
        model_provider: ModelProvider | None,
    ) -> JsonValue:
        self._validate_model_provider(model_provider)
        if cancelled is not None and cancelled():
            raise ToolboxError("cancelled", "rollout stopped")
        after = self._latest_sequence(session_id)
        scope = self.kernel.snapshot_scope(session_id)
        request = {
            "db_path": str(self.kernel.db_path),
            "max_depth": self.kernel.max_depth,
            "max_calls": self.kernel.max_calls,
            "name": name,
            "args": args,
            "session_id": session_id,
            "toolbox_ids": scope.toolbox_ids,
            "cwd": scope.cwd,
        }
        environment = {
            key: value
            for key, value in os.environ.items()
            if _is_safe_environment_variable(key)
        }
        environment["PYTHONIOENCODING"] = "utf-8"

        provider_host: socket.socket | None = None
        provider_worker: socket.socket | None = None
        broker_thread: Thread | None = None
        process: subprocess.Popen[bytes] | None = None
        with tempfile.TemporaryDirectory(prefix="mechagnome-") as directory:
            root = Path(directory)
            request_path = root / "request.json"
            response_path = root / "response.json"
            try:
                if model_provider is not None:
                    provider_host, provider_worker = socket.socketpair()
                    request["model_provider_fd"] = provider_worker.fileno()
                request_path.write_text(json.dumps(request), encoding="utf-8")
                pass_fds = (
                    (provider_worker.fileno(),) if provider_worker is not None else ()
                )
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-P",
                            "-m",
                            "mechagnome.tool_worker",
                            str(request_path),
                            str(response_path),
                        ],
                        cwd=scope.cwd,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                        pass_fds=pass_fds,
                    )
                except OSError as error:
                    raise ToolboxError(
                        "session_cwd_unavailable",
                        f"cannot launch tool worker in session cwd: {scope.cwd}",
                        reason=str(error),
                    ) from error
                if provider_worker is not None:
                    provider_worker.close()
                    provider_worker = None
                if provider_host is not None and model_provider is not None:
                    broker = _ModelProviderBroker(provider_host, model_provider)
                    broker_thread = Thread(
                        target=broker.serve,
                        name="mechagnome-model-provider",
                        daemon=True,
                    )
                    broker_thread.start()
                deadline = time.monotonic() + self.timeout
                while process.poll() is None:
                    after = self._relay(session_id, after, on_event)
                    if cancelled is not None and cancelled():
                        raise ToolboxError("cancelled", "rollout stopped")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ToolboxError(
                            "tool_timeout",
                            f"tool call exceeded {self.timeout:g} seconds",
                        )
                    try:
                        process.wait(timeout=min(0.1, remaining))
                    except subprocess.TimeoutExpired:
                        pass
                after = self._relay(session_id, after, on_event)
                if cancelled is not None and cancelled():
                    raise ToolboxError("cancelled", "rollout stopped")
                if process.returncode != 0 or not response_path.exists():
                    raise ToolboxError(
                        "tool_worker_failed",
                        f"isolated tool worker exited with status {process.returncode}",
                    )
                try:
                    response = json.loads(response_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as error:
                    raise ToolboxError(
                        "tool_worker_failed",
                        "isolated tool worker returned invalid data",
                    ) from error
            finally:
                had_error = sys.exc_info()[0] is not None
                process_active = process is not None and process.poll() is None
                broker_active = broker_thread is not None and broker_thread.is_alive()
                provider_was_cancelled = broker_active
                if provider_was_cancelled:
                    self._cancel_model_provider(model_provider)
                if process_active and process is not None:
                    self._stop_process_group(process)
                if process is not None:
                    try:
                        self._relay(session_id, after, on_event)
                    except Exception:
                        pass
                if provider_worker is not None:
                    provider_worker.close()
                if provider_host is not None:
                    try:
                        provider_host.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    provider_host.close()
                broker_stuck = False
                if broker_thread is not None:
                    broker_thread.join(timeout=min(1.0, max(0.1, self.timeout)))
                    broker_stuck = broker_thread.is_alive()
                if provider_was_cancelled and not broker_stuck:
                    self._reset_model_provider(model_provider)
                if broker_stuck and not had_error:
                    raise ToolboxError(
                        "model_provider_failed",
                        "model provider did not stop after cancellation",
                    )

        if not isinstance(response, dict):
            raise ToolboxError(
                "tool_worker_failed", "isolated tool worker returned invalid data"
            )
        if response.get("ok") is True:
            return response.get("result")
        failure = response.get("error")
        if not isinstance(failure, dict):
            raise ToolboxError("tool_worker_failed", "isolated tool worker failed")
        raise ToolboxError(
            str(failure.get("code") or "tool_failed"),
            str(failure.get("message") or "isolated tool call failed"),
            **(
                failure.get("details")
                if isinstance(failure.get("details"), dict)
                else {}
            ),
        )

    @staticmethod
    def _validate_model_provider(provider: ModelProvider | None) -> None:
        if provider is None:
            return
        if not all(
            callable(getattr(provider, name, None))
            for name in ("complete", "cancel_current", "reset_cancellation")
        ):
            raise ToolboxError(
                "invalid_model_provider",
                "isolated model providers must support cancellation and reset",
            )

    @staticmethod
    def _cancel_model_provider(provider: ModelProvider | None) -> None:
        if provider is None:
            return
        try:
            provider.cancel_current()
        except Exception:
            pass

    @staticmethod
    def _reset_model_provider(provider: ModelProvider | None) -> None:
        if provider is None:
            return
        try:
            provider.reset_cancellation()
        except Exception:
            pass

    @staticmethod
    def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
        """Stop the worker and any subprocesses created by authored tools."""
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait()

    def _latest_sequence(self, session_id: str) -> int:
        page = self.kernel.read_session(session_id, after=0, limit=100)
        latest = page["events"][-1]["seq"] if page["events"] else 0
        while page["next_after"] is not None:
            page = self.kernel.read_session(
                session_id, after=page["next_after"], limit=100
            )
            if page["events"]:
                latest = page["events"][-1]["seq"]
        return latest

    def _relay(
        self,
        session_id: str,
        after: int,
        sink: CommittedEventSink | None,
    ) -> int:
        while True:
            page = self.kernel.read_session(session_id, after=after, limit=100)
            for event in page["events"]:
                after = event["seq"]
                if sink is not None:
                    sink(event)
            if page["next_after"] is None:
                return after
