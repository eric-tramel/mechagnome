"""Filtered-environment subprocess boundary for authored tool execution."""

from __future__ import annotations

import codecs
import json
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
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
DetachedEventSink = Callable[[str, dict[str, Any]], None]

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
class _DetachedJob:
    """One process-lifetime background call and its bounded observable state."""

    job_id: str
    parent_session_id: str
    name: str
    args: dict[str, Any]
    version: int | None
    sink: DetachedEventSink | None
    status: str = "running"
    output_tail: str = ""
    truncated: bool = False
    result: JsonValue = None
    error: dict[str, Any] | None = None
    cancelled: Event = field(default_factory=Event)
    thread: Thread | None = None


class IsolatedToolRunner:
    """Run an entire editable tool call tree outside the provider process."""

    def __init__(self, kernel: Kernel, *, timeout: float = 120.0) -> None:
        self.kernel = kernel
        self.timeout = timeout
        self._jobs: dict[str, _DetachedJob] = {}
        self._jobs_lock = Lock()
        self._closed = False

    def start_detached(
        self,
        name: str,
        args: dict[str, Any],
        *,
        session_id: str,
        version: int | None = None,
        on_update: DetachedEventSink | None = None,
    ) -> dict[str, Any]:
        """Start one providerless call tree and return its process-lifetime handle."""
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
        scope = self.kernel.snapshot_scope(session_id)
        with self._jobs_lock:
            if self._closed:
                raise ToolboxError(
                    "detached_runner_closed", "detached tool runner is closed"
                )
            active = sum(job.status == "running" for job in self._jobs.values())
            if active >= _MAX_DETACHED_JOBS:
                raise ToolboxError(
                    "detached_job_limit",
                    f"at most {_MAX_DETACHED_JOBS} detached jobs may run at once",
                )
            job_id = self.kernel.create_child_session(scope, kind="generic")
            job = _DetachedJob(
                job_id,
                session_id,
                name,
                dict(args),
                version,
                on_update,
            )
            thread = Thread(
                target=self._run_detached,
                args=(job, scope),
                name=f"mechagnome-detached-{job_id[:8]}",
            )
            job.thread = thread
            self._jobs[job_id] = job
            try:
                thread.start()
            except Exception as error:
                start_error = error
            else:
                start_error = None
        if start_error is not None:
            failure = {
                "code": "detached_start_failed",
                "message": str(start_error),
                "details": {},
            }
            self._finish_job(job, status="failed", error=failure)
            raise ToolboxError(
                "detached_start_failed", "could not start detached tool job"
            ) from start_error
        return {"job_id": job_id, "status": "running"}

    def inspect_detached(self, job_id: str, *, session_id: str) -> dict[str, Any]:
        """Return the latest bounded state for a job owned by this conversation."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None or job.parent_session_id != session_id:
                raise ToolboxError(
                    "unknown_detached_job", f"unknown detached job: {job_id}"
                )
            result: dict[str, Any] = {
                "job_id": job.job_id,
                "status": job.status,
                "output_tail": job.output_tail,
                "truncated": job.truncated,
            }
            if job.status == "succeeded":
                result["result"] = job.result
            elif job.status == "failed":
                result["error"] = dict(job.error or {})
        return result

    def close(self) -> None:
        """Stop and join every owned detached worker; safe to call repeatedly."""
        with self._jobs_lock:
            if self._closed:
                return
            self._closed = True
            jobs = list(self._jobs.values())
            for job in jobs:
                if job.status == "running":
                    job.cancelled.set()
        for job in jobs:
            thread = job.thread
            if (
                thread is not None
                and thread.ident is not None
                and thread is not current_thread()
            ):
                thread.join()

    def _run_detached(self, job: _DetachedJob, parent_scope: InvocationScope) -> None:
        self._notify(job, "detached_started", self._snapshot(job))
        envelope: dict[str, Any] = {"name": job.name, "args": job.args}
        if job.version is not None:
            envelope["version"] = job.version
        scope = InvocationScope(
            session_id=job.job_id,
            toolbox_ids=parent_scope.toolbox_ids,
            cwd=parent_scope.cwd,
        )
        try:
            result = self._call(
                "call_tool",
                envelope,
                session_id=job.job_id,
                on_event=None,
                cancelled=job.cancelled.is_set,
                model_provider=None,
                scope=scope,
                on_output=lambda text: self._append_job_output(job, text),
            )
        except Exception as error:
            if job.cancelled.is_set():
                failure = {
                    "code": "detached_shutdown",
                    "message": "detached job stopped because its runner closed",
                    "details": {},
                }
            elif isinstance(error, ToolboxError):
                failure = error.to_dict()["error"]
            else:
                failure = {
                    "code": "tool_failed",
                    "message": str(error),
                    "details": {},
                }
            self._finish_job(job, status="failed", error=failure)
        else:
            self._finish_job(job, status="succeeded", result=result)

    def _append_job_output(self, job: _DetachedJob, text: str) -> None:
        if not text:
            return
        with self._jobs_lock:
            combined = (job.output_tail + text).encode("utf-8")
            if len(combined) > _MAX_OUTPUT_BYTES:
                combined = combined[-_MAX_OUTPUT_BYTES:]
                while combined and combined[0] & 0xC0 == 0x80:
                    combined = combined[1:]
                job.truncated = True
            job.output_tail = combined.decode("utf-8", errors="replace")
            payload = self._snapshot_locked(job)
        self._notify(job, "detached_output", payload)

    def _finish_job(
        self,
        job: _DetachedJob,
        *,
        status: str,
        result: JsonValue = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        if status == "succeeded":
            encoded_result = json.dumps(
                result, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            if len(encoded_result) > _MAX_DETACHED_RESULT_BYTES:
                status = "failed"
                result = None
                error = {
                    "code": "detached_result_too_large",
                    "message": "detached result exceeds the retained byte limit",
                    "details": {"limit_bytes": _MAX_DETACHED_RESULT_BYTES},
                }
        with self._jobs_lock:
            if job.status != "running":
                return
            job.status = status
            job.result = result
            job.error = error
            payload = self._snapshot_locked(job)
        self._notify(job, "detached_finished", payload)
        with self._jobs_lock:
            job.sink = None
            terminal = [
                job_id
                for job_id, candidate in self._jobs.items()
                if candidate.status != "running"
            ]
            for job_id in terminal[:-_MAX_RETAINED_DETACHED_JOBS]:
                del self._jobs[job_id]

    def _snapshot(self, job: _DetachedJob) -> dict[str, Any]:
        with self._jobs_lock:
            return self._snapshot_locked(job)

    @staticmethod
    def _snapshot_locked(job: _DetachedJob) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": job.job_id,
            "name": job.name,
            "args": job.args,
            "status": job.status,
            "output_tail": job.output_tail,
            "truncated": job.truncated,
        }
        if job.status == "succeeded":
            payload["result"] = job.result
        elif job.status == "failed":
            payload["error"] = dict(job.error or {})
        return payload

    @staticmethod
    def _notify(job: _DetachedJob, kind: str, payload: dict[str, Any]) -> None:
        if job.sink is None:
            return
        try:
            job.sink(kind, payload)
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
    ) -> JsonValue:
        self._validate_model_provider(model_provider)
        if cancelled is not None and cancelled():
            raise ToolboxError("cancelled", "rollout stopped")
        after = self._latest_sequence(session_id)
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
                if process is not None and (process_active or on_output is not None):
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
                if output_thread is not None:
                    output_stop.set()
                    output_thread.join()
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
