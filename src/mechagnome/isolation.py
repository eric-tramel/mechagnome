"""Filtered-environment subprocess boundary for authored tool execution."""

from __future__ import annotations

import codecs
import json
import math
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Event, Lock, Thread, current_thread
from typing import Any

from mechagnome.kernel import InvocationScope, JsonValue, Kernel, ToolboxError
from mechagnome.model_provider import (
    ModelProvider,
    ModelSession,
    _BoundedModelProvider,
    _CompletionProvider,
    _ModelProviderBroker,
)

CommittedEventSink = Callable[[dict[str, Any]], None]
ToolRunEventSink = Callable[[str, dict[str, Any]], None]

_MAX_DETACHED_JOBS = 4
_MAX_RETAINED_DETACHED_JOBS = 64
_MAX_DETACHED_RESULT_BYTES = 1024 * 1024
_MAX_OUTPUT_BYTES = 256 * 1024
_MAX_SHUTDOWN_DRAIN_BYTES = 64 * 1024

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


def _deadline_after_milliseconds(timeout_ms: int) -> float:
    """Return a monotonic deadline, saturating unrepresentably large waits."""
    try:
        return time.monotonic() + timeout_ms / 1000
    except OverflowError:
        return math.inf


def _deadline_after_seconds(timeout: float) -> float:
    """Return a monotonic deadline, saturating unrepresentably large waits."""
    try:
        return time.monotonic() + timeout
    except OverflowError:
        return math.inf


class _ControlSanitizer:
    """Strip terminal controls while preserving ordinary whitespace and text."""

    def __init__(self) -> None:
        self._state = "text"

    def feed(self, value: str) -> str:
        output: list[str] = []
        for character in value:
            codepoint = ord(character)
            if self._state == "text":
                if character == "\x1b":
                    self._state = "escape"
                elif character in "\n\r\t" or 32 <= codepoint < 127 or codepoint >= 160:
                    output.append(character)
            elif self._state == "escape":
                if character == "[":
                    self._state = "csi"
                elif character == "]":
                    self._state = "osc"
                else:
                    self._state = "text"
            elif self._state == "csi":
                if "@" <= character <= "~":
                    self._state = "text"
            elif self._state == "osc":
                if character == "\x07":
                    self._state = "text"
                elif character == "\x1b":
                    self._state = "osc_escape"
            elif self._state == "osc_escape":
                self._state = "text" if character == "\\" else "osc"
        return "".join(output)


@dataclass
class _ToolRun:
    """One process-lifetime detached tool invocation and its retained state."""

    run_id: str
    trace_session_id: str
    owner_session_id: str
    name: str
    args: dict[str, Any]
    version: int | None
    sink: ToolRunEventSink | None
    status: str = "running"
    output_tail: str = ""
    truncated: bool = False
    result: JsonValue = None
    error: dict[str, Any] | None = None
    cancelled: Event = field(default_factory=Event)
    thread: Thread | None = None
    generation: int = 0
    stop_reason: str | None = None


class IsolatedToolRunner:
    """Run an entire editable tool call tree outside the provider process."""

    def __init__(self, kernel: Kernel, *, timeout: float | None = None) -> None:
        self.kernel = kernel
        self.timeout = timeout
        self._runs: dict[str, _ToolRun] = {}
        self._completed_run_ids: deque[str] = deque()
        self._runs_lock = Lock()
        self._runs_changed = Condition(self._runs_lock)
        self._foreground_lock = Lock()
        self._foreground_processes: dict[str, set[subprocess.Popen[bytes]]] = {}
        self._foreground_cancelled: set[str] = set()
        self._closed = False

    def start_detached(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session_id: str,
        version: int | None = None,
        on_update: ToolRunEventSink | None = None,
        model_provider: _CompletionProvider | None = None,
        scope: InvocationScope | None = None,
        owner_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Start one call tree and return its process-lifetime ToolRun handle."""
        if not isinstance(name, str) or not name or not isinstance(args, dict):
            raise ToolboxError(
                "invalid_call_tool_request",
                "detached calls require a non-empty name and object args",
            )
        if version is not None and (
            isinstance(version, bool) or not isinstance(version, int) or version < 1
        ):
            raise ToolboxError(
                "invalid_call_tool_request",
                "detached call version must be a positive integer",
            )
        scope = scope or self.kernel.snapshot_scope(session_id)
        owner_session_id = owner_session_id or session_id
        with self._runs_changed:
            if self._closed:
                raise ToolboxError(
                    "detached_runner_closed", "detached tool runner is closed"
                )
            active = sum(
                run.status in {"running", "cancelling"} for run in self._runs.values()
            )
            if active >= _MAX_DETACHED_JOBS:
                raise ToolboxError(
                    "detached_job_limit",
                    f"at most {_MAX_DETACHED_JOBS} detached jobs may run at once",
                )
            trace_session_id = self.kernel.create_child_session(scope, kind="generic")
            run_id = uuid.uuid4().hex
            run = _ToolRun(
                run_id,
                trace_session_id,
                owner_session_id,
                name,
                dict(args),
                version,
                on_update,
            )
            thread = Thread(
                target=self._run_detached,
                args=(run, scope, model_provider),
                name=f"mechagnome-tool-run-{run_id[:8]}",
            )
            run.thread = thread
            self._runs[run_id] = run
            try:
                thread.start()
            except Exception as error:
                start_error = error
            else:
                start_error = None
        if start_error is not None:
            with self._runs_changed:
                self._runs.pop(run_id, None)
                run.sink = None
                self._runs_changed.notify_all()
            raise ToolboxError(
                "detached_start_failed", "could not start detached tool job"
            ) from start_error
        return {
            "run_id": run_id,
            "job_id": run_id,
            "tool_name": name,
            "status": "running",
        }

    def inspect_detached(self, job_id: str, *, session_id: str) -> dict[str, Any]:
        """Compatibility view of a ToolRun using the former detached-job shape."""
        try:
            with self._runs_lock:
                run = self._visible_run_locked(job_id, session_id)
                result = self._event_snapshot_locked(run)
        except ToolboxError as error:
            if error.code == "unknown_tool_run":
                raise ToolboxError(
                    "unknown_detached_job", f"unknown detached job: {job_id}"
                ) from error
            raise
        return result

    def get_tool_run(self, run_id: str, *, session_id: str) -> dict[str, Any]:
        """Return lightweight state for a ToolRun visible to this session."""
        with self._runs_lock:
            run = self._visible_run_locked(run_id, session_id)
            return self._status_locked(run)

    def wait_tool_run(
        self, run_id: str, *, session_id: str, timeout_ms: int | None = None
    ) -> dict[str, Any]:
        """Wait for a visible ToolRun, optionally bounded by ``timeout_ms``."""
        if timeout_ms is not None and (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or timeout_ms < 0
        ):
            raise ToolboxError(
                "invalid_tool_run_request",
                "tool run timeout_ms must be a non-negative integer",
            )
        deadline = (
            None if timeout_ms is None else _deadline_after_milliseconds(timeout_ms)
        )
        with self._runs_changed:
            run = self._visible_run_locked(run_id, session_id)
            while run.status in {"running", "cancelling"}:
                if deadline is None:
                    self._runs_changed.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        result = self._status_locked(run)
                        result["timed_out"] = True
                        return result
                    self._runs_changed.wait(min(1.0, remaining))
            return self._terminal_snapshot_locked(run)

    def cancel_tool_run(self, run_id: str, *, session_id: str) -> dict[str, Any]:
        """Request cancellation without exposing whether an inaccessible run exists."""
        with self._runs_changed:
            run = self._visible_run_locked(run_id, session_id)
            requested = run.status == "running"
            if requested:
                run.status = "cancelling"
                run.stop_reason = "cancel"
                run.generation += 1
                run.cancelled.set()
                payload = self._event_snapshot_locked(run)
                self._runs_changed.notify_all()
            else:
                payload = None
            result = self._status_locked(run)
            result["cancellation_requested"] = requested
        if payload is not None:
            self._notify(run, "tool_run_cancelling", payload)
        return result

    def close(self) -> None:
        """Stop and join every owned detached worker; safe to call repeatedly."""
        with self._runs_changed:
            if self._closed:
                return
            self._closed = True
            runs = list(self._runs.values())
            for run in runs:
                if run.status == "running":
                    run.status = "cancelling"
                    run.stop_reason = "shutdown"
                    run.generation += 1
                    run.cancelled.set()
            self._runs_changed.notify_all()
        for run in runs:
            thread = run.thread
            if (
                thread is not None
                and thread.ident is not None
                and thread is not current_thread()
            ):
                thread.join()

    def _run_detached(
        self,
        run: _ToolRun,
        parent_scope: InvocationScope,
        model_provider: _CompletionProvider | None,
    ) -> None:
        self._notify(run, "tool_run_started", self._event_snapshot(run))
        envelope: dict[str, Any] = {"name": run.name, "args": run.args}
        if run.version is not None:
            envelope["version"] = run.version
        scope = InvocationScope(
            session_id=run.trace_session_id,
            toolbox_ids=parent_scope.toolbox_ids,
            cwd=parent_scope.cwd,
        )
        result: JsonValue = None
        error: dict[str, Any] | None = None
        try:
            result = self._call(
                "call_tool",
                envelope,
                session_id=run.trace_session_id,
                on_event=None,
                cancelled=run.cancelled.is_set,
                model_provider=model_provider,
                scope=scope,
                on_output=lambda text: self._append_run_output(run, text),
                foreground=False,
            )
        except Exception as caught:
            status = "failed"
            if isinstance(caught, ToolboxError):
                failure = caught.to_dict()["error"]
            else:
                failure = {
                    "code": "tool_failed",
                    "message": str(caught),
                    "details": {},
                }
            error = failure
        else:
            status = "succeeded"
        finally:
            release = getattr(model_provider, "release_tool_run", None)
            if callable(release):
                release(run.trace_session_id)
        self._finish_run(run, status=status, result=result, error=error)

    def _append_run_output(self, run: _ToolRun, text: str) -> None:
        if not text:
            return
        with self._runs_changed:
            combined = (run.output_tail + text).encode("utf-8")
            if len(combined) > _MAX_OUTPUT_BYTES:
                combined = combined[-_MAX_OUTPUT_BYTES:]
                while combined and combined[0] & 0xC0 == 0x80:
                    combined = combined[1:]
                run.truncated = True
            run.output_tail = combined.decode("utf-8", errors="replace")
            run.generation += 1
            payload = self._event_snapshot_locked(run)
            self._runs_changed.notify_all()
        self._notify(run, "tool_run_output", payload)

    def _finish_run(
        self,
        run: _ToolRun,
        *,
        status: str,
        result: JsonValue = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        if status == "succeeded":
            try:
                encoded_result = json.dumps(
                    result, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            except UnicodeEncodeError:
                encoded_result = json.dumps(
                    result, ensure_ascii=True, separators=(",", ":")
                ).encode("utf-8")
            if len(encoded_result) > _MAX_DETACHED_RESULT_BYTES:
                status = "failed"
                result = None
                error = {
                    "code": "detached_result_too_large",
                    "message": "detached result exceeds the retained byte limit",
                    "details": {"limit_bytes": _MAX_DETACHED_RESULT_BYTES},
                }
        with self._runs_changed:
            if run.status not in {"running", "cancelling"}:
                return
            if run.stop_reason == "cancel":
                run.status = "cancelled"
                run.result = None
                run.error = {
                    "code": "tool_run_cancelled",
                    "message": "tool run cancelled by request",
                    "details": {},
                }
            elif run.stop_reason == "shutdown":
                run.status = "failed"
                run.result = None
                run.error = {
                    "code": "detached_shutdown",
                    "message": "detached job stopped because its runner closed",
                    "details": {},
                }
            else:
                run.status = status
                run.result = result
                run.error = error
            run.generation += 1
            payload = self._event_snapshot_locked(run)
            self._runs_changed.notify_all()
        self._notify(run, "tool_run_finished", payload)
        with self._runs_lock:
            run.sink = None
            self._completed_run_ids.append(run.run_id)
            while len(self._completed_run_ids) > _MAX_RETAINED_DETACHED_JOBS:
                completed_run_id = self._completed_run_ids.popleft()
                self._runs.pop(completed_run_id, None)

    def _event_snapshot(self, run: _ToolRun) -> dict[str, Any]:
        with self._runs_lock:
            return self._event_snapshot_locked(run)

    @staticmethod
    def _event_snapshot_locked(run: _ToolRun) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": run.run_id,
            "job_id": run.run_id,
            "tool_name": run.name,
            "name": run.name,
            "status": run.status,
            "output_tail": run.output_tail,
            "truncated": run.truncated,
            "generation": run.generation,
        }
        if run.status == "succeeded":
            payload["result"] = run.result
        elif run.status in {"failed", "cancelled"}:
            payload["error"] = dict(run.error or {})
        return payload

    @staticmethod
    def _status_locked(run: _ToolRun) -> dict[str, Any]:
        return {
            "run_id": run.run_id,
            "tool_name": run.name,
            "status": run.status,
        }

    def _terminal_snapshot_locked(self, run: _ToolRun) -> dict[str, Any]:
        payload = self._status_locked(run)
        payload.update(
            {
                "output_tail": run.output_tail,
                "truncated": run.truncated,
            }
        )
        if run.status == "succeeded":
            payload["result"] = run.result
        else:
            payload["error"] = dict(run.error or {})
        return payload

    def _visible_run_locked(self, run_id: str, session_id: str) -> _ToolRun:
        run = self._runs.get(run_id) if isinstance(run_id, str) else None
        if run is None or not self._is_owner_or_ancestor(
            session_id, run.owner_session_id
        ):
            raise ToolboxError("unknown_tool_run", f"unknown tool run: {run_id}")
        return run

    def _is_owner_or_ancestor(self, candidate: str, owner: str) -> bool:
        current: str | None = owner
        while current is not None:
            if current == candidate:
                return True
            try:
                current = self.kernel.session_metadata(current)["parent_session_id"]
            except ToolboxError:
                return False
        return False

    @staticmethod
    def _notify(run: _ToolRun, kind: str, payload: dict[str, Any]) -> None:
        if run.sink is None:
            return
        try:
            run.sink(kind, payload)
        except Exception:
            pass

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
        model_provider: _CompletionProvider | None,
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

    def cancel_foreground(self, session_id: str) -> None:
        """Immediately kill every foreground worker owned by one conversation."""
        with self._foreground_lock:
            self._foreground_cancelled.add(session_id)
            processes = tuple(self._foreground_processes.get(session_id, ()))
        for process in processes:
            self._signal_process_group(process, signal.SIGKILL)

    def reset_foreground_cancellation(self, session_id: str) -> None:
        """Clear a completed rollout's process-registration cancellation latch."""
        with self._foreground_lock:
            self._foreground_cancelled.discard(session_id)

    def _call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session_id: str,
        on_event: CommittedEventSink | None,
        cancelled: Callable[[], bool] | None,
        model_provider: _CompletionProvider | None,
        scope: InvocationScope | None = None,
        on_output: Callable[[str], None] | None = None,
        foreground: bool = True,
    ) -> JsonValue:
        self._validate_model_provider(model_provider)
        if cancelled is not None and cancelled():
            raise ToolboxError("cancelled", "rollout stopped")
        after = self.kernel.latest_event_sequence(session_id)
        scope = scope or self.kernel.snapshot_scope(session_id)
        if scope.session_id != session_id:
            raise ToolboxError("invalid_session", "tool scope session does not match")
        if model_provider is not None and not isinstance(
            model_provider, _BoundedModelProvider
        ):
            gateway = ModelProvider.from_completion_transport(
                self.kernel, model_provider
            )
            model_provider = ModelSession(gateway, session_id).completion_provider(
                scope
            )
        elif isinstance(model_provider, _BoundedModelProvider):
            model_provider = model_provider.for_scope(scope)
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
        if on_output is not None:
            environment["PYTHONUNBUFFERED"] = "1"

        provider_host: socket.socket | None = None
        provider_worker: socket.socket | None = None
        broker_thread: Thread | None = None
        output_thread: Thread | None = None
        output_stop = Event()
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
                        stdout=(
                            subprocess.PIPE
                            if on_output is not None
                            else subprocess.DEVNULL
                        ),
                        stderr=(
                            subprocess.STDOUT
                            if on_output is not None
                            else subprocess.DEVNULL
                        ),
                        start_new_session=True,
                        pass_fds=pass_fds,
                    )
                    if foreground:
                        self._register_foreground(session_id, process)
                except OSError as error:
                    raise ToolboxError(
                        "session_cwd_unavailable",
                        f"cannot launch tool worker in session cwd: {scope.cwd}",
                        reason=str(error),
                    ) from error
                if on_output is not None:
                    assert process.stdout is not None
                    output_thread = Thread(
                        target=self._read_output,
                        args=(process.stdout, on_output, output_stop),
                        name="mechagnome-tool-output",
                    )
                    output_thread.start()
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
                deadline = (
                    None
                    if self.timeout is None
                    else _deadline_after_seconds(self.timeout)
                )
                while process.poll() is None:
                    after = self._relay(session_id, after, on_event)
                    if cancelled is not None and cancelled():
                        raise ToolboxError("cancelled", "rollout stopped")
                    wait_interval = 0.1
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise ToolboxError(
                                "tool_timeout",
                                f"tool call exceeded {self.timeout:g} seconds",
                            )
                        wait_interval = min(wait_interval, remaining)
                    try:
                        process.wait(timeout=wait_interval)
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
                if process is not None and (process_active or on_output is not None):
                    if foreground and self._foreground_was_cancelled(session_id):
                        self._kill_process_group(process)
                    else:
                        self._stop_process_group(process)
                if process is not None and foreground:
                    self._unregister_foreground(session_id, process)
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
                if output_thread is not None:
                    output_stop.set()
                    output_thread.join()
                broker_stuck = False
                if broker_thread is not None:
                    broker_thread.join(timeout=1.0)
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
    def _read_output(stream: Any, sink: Callable[[str], None], stop: Event) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        sanitizer = _ControlSanitizer()
        try:
            descriptor = stream.fileno()
            os.set_blocking(descriptor, False)
            while True:
                readable, _, _ = select.select([descriptor], [], [], 0.1)
                if readable:
                    try:
                        chunk = os.read(descriptor, 4096)
                    except BlockingIOError:
                        chunk = None
                    if chunk == b"":
                        break
                    if chunk:
                        text = sanitizer.feed(decoder.decode(chunk))
                        if text:
                            sink(text)
                if stop.is_set():
                    remaining = _MAX_SHUTDOWN_DRAIN_BYTES
                    while remaining > 0:
                        try:
                            chunk = os.read(descriptor, min(4096, remaining))
                        except BlockingIOError:
                            break
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        text = sanitizer.feed(decoder.decode(chunk))
                        if text:
                            sink(text)
                    break
            final = sanitizer.feed(decoder.decode(b"", final=True))
            if final:
                sink(final)
        finally:
            stream.close()

    @staticmethod
    def _validate_model_provider(provider: _CompletionProvider | None) -> None:
        if provider is None:
            return
        if isinstance(provider, _BoundedModelProvider):
            if provider.supports_cancellation:
                return
            raise ToolboxError(
                "invalid_model_provider",
                "isolated model providers must support cancellation and reset",
            )
        if not all(
            callable(getattr(provider, name, None))
            for name in ("complete", "cancel_current", "reset_cancellation")
        ):
            raise ToolboxError(
                "invalid_model_provider",
                "isolated model providers must support cancellation and reset",
            )

    @staticmethod
    def _cancel_model_provider(provider: _CompletionProvider | None) -> None:
        if provider is None:
            return
        try:
            provider.cancel_current()
        except Exception:
            pass

    @staticmethod
    def _reset_model_provider(provider: _CompletionProvider | None) -> None:
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
        except (PermissionError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, 0)
        except (PermissionError, ProcessLookupError):
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            pass
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _signal_process_group(
        process: subprocess.Popen[bytes], signal_number: signal.Signals
    ) -> None:
        try:
            os.killpg(process.pid, signal_number)
        except (PermissionError, ProcessLookupError):
            pass

    @classmethod
    def _kill_process_group(cls, process: subprocess.Popen[bytes]) -> None:
        cls._signal_process_group(process, signal.SIGKILL)
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    def _register_foreground(
        self, session_id: str, process: subprocess.Popen[bytes]
    ) -> None:
        with self._foreground_lock:
            self._foreground_processes.setdefault(session_id, set()).add(process)
            cancelled = session_id in self._foreground_cancelled
        if cancelled:
            self._signal_process_group(process, signal.SIGKILL)

    def _unregister_foreground(
        self, session_id: str, process: subprocess.Popen[bytes]
    ) -> None:
        with self._foreground_lock:
            processes = self._foreground_processes.get(session_id)
            if processes is None:
                return
            processes.discard(process)
            if not processes:
                del self._foreground_processes[session_id]

    def _foreground_was_cancelled(self, session_id: str) -> bool:
        with self._foreground_lock:
            return session_id in self._foreground_cancelled

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
