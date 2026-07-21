"""Filtered-environment subprocess boundary for authored tool execution."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mechagnome.kernel import JsonValue, Kernel, ToolboxError

CommittedEventSink = Callable[[dict[str, Any]], None]

_SAFE_ENVIRONMENT = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "TERM",
    "TMPDIR",
    "TZ",
)


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
            for key in _SAFE_ENVIRONMENT
            if (value := os.environ.get(key)) is not None
        }
        environment["PYTHONIOENCODING"] = "utf-8"

        with tempfile.TemporaryDirectory(prefix="mechagnome-") as directory:
            root = Path(directory)
            request_path = root / "request.json"
            response_path = root / "response.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
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
                )
            except OSError as error:
                raise ToolboxError(
                    "session_cwd_unavailable",
                    f"cannot launch tool worker in session cwd: {scope.cwd}",
                    reason=str(error),
                ) from error
            deadline = time.monotonic() + self.timeout
            while process.poll() is None:
                after = self._relay(session_id, after, on_event)
                if cancelled is not None and cancelled():
                    self._stop_process_group(process)
                    self._relay(session_id, after, on_event)
                    raise ToolboxError("cancelled", "rollout stopped")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop_process_group(process)
                    self._relay(session_id, after, on_event)
                    raise ToolboxError(
                        "tool_timeout",
                        f"tool call exceeded {self.timeout:g} seconds",
                    )
                try:
                    process.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
            self._relay(session_id, after, on_event)
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
                    "tool_worker_failed", "isolated tool worker returned invalid data"
                ) from error

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
