"""Provider-neutral model loop over editable tools and host agent actions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from threading import Event, Lock, Thread, current_thread
from typing import Any

from mechagnome.bootstrap import CORE_NAMES
from mechagnome.isolation import IsolatedToolRunner
from mechagnome.kernel import InvocationScope, JsonValue, Kernel, ToolboxError
from mechagnome.model_provider import (
    MAX_MODEL_REQUEST_BYTES,
    CompletionTransport,
    ModelProvider,
    ModelSession,
    ModelStreamEvent,  # noqa: F401 - compatibility re-export
    ModelTransport,
    ModelTransportError,
    ModelTurn,
    ToolCall,
)

Model = ModelTransport


@dataclass(frozen=True)
class RunResult:
    """Final answer and durable session identity from a harness run."""

    session_id: str
    answer: str
    turns: int


@dataclass(frozen=True)
class AgentEvent:
    """One agent activity event; ``seq`` is set only when durably committed."""

    kind: str
    payload: dict[str, Any]
    seq: int | None
    call_id: str | None = None
    parent_call_id: str | None = None
    toolbox_id: str | None = None
    tool_name: str | None = None
    tool_version: int | None = None
    tool_version_id: int | None = None


EventSink = Callable[[AgentEvent], None]
DEFAULT_MAX_CALLS_PER_TURN = 16
RUN_AGENT_NAME = "run_agent"
HOST_ACTION_NAMES = (RUN_AGENT_NAME,)
MODEL_ACTION_NAMES = (*CORE_NAMES, *HOST_ACTION_NAMES)
_MAX_DETACHED_AGENT_JOBS = 4
_MAX_RETAINED_DETACHED_AGENT_JOBS = 64
_MAX_DETACHED_AGENT_RESULT_BYTES = 1024 * 1024
_MAX_ACTIVE_FOREGROUND_AGENTS = 16
_MAX_AGENT_LAUNCHES_PER_ROLLOUT = 64

RUN_AGENT_DEFINITION = {
    "name": RUN_AGENT_NAME,
    "description": (
        "Run another full agent with the same actions, detach it for later "
        "inspection, or inspect a detached agent job."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "detach": {"type": "boolean"},
            "job_id": {"type": "string"},
        },
        "additionalProperties": False,
    },
}


def _parse_call_tool_request(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify and validate the flat sync/start/inspect call-tool union."""
    keys = set(args)
    if "job_id" in keys:
        job_id = args.get("job_id")
        if keys == {"job_id"} and isinstance(job_id, str) and job_id:
            return "inspect", {"job_id": job_id}
        raise ToolboxError(
            "invalid_call_tool_request",
            "job inspection requires only a non-empty job_id",
        )

    allowed = {"name", "args", "version", "detach"}
    name = args.get("name")
    tool_args = args.get("args")
    version = args.get("version")
    detach = args.get("detach", False)
    valid_version = version is None or (
        not isinstance(version, bool) and isinstance(version, int) and version >= 1
    )
    if (
        not keys <= allowed
        or not isinstance(name, str)
        or not name
        or not isinstance(tool_args, dict)
        or not valid_version
        or not isinstance(detach, bool)
    ):
        raise ToolboxError(
            "invalid_call_tool_request",
            "call_tool requires name, object args, optional positive version, and "
            "optional boolean detach",
        )
    normalized: dict[str, Any] = {"name": name, "args": tool_args}
    if version is not None:
        normalized["version"] = version
    return ("start" if detach else "sync"), normalized


def _parse_run_agent_request(args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify and validate the flat run/start/inspect agent union."""
    keys = set(args)
    if "job_id" in keys:
        job_id = args.get("job_id")
        if keys == {"job_id"} and isinstance(job_id, str) and job_id:
            return "inspect", {"job_id": job_id}
        raise ToolboxError(
            "invalid_run_agent_request",
            "agent inspection requires only a non-empty job_id",
        )

    prompt = args.get("prompt")
    detach = args.get("detach", False)
    if (
        not keys <= {"prompt", "detach"}
        or not isinstance(prompt, str)
        or not prompt
        or not isinstance(detach, bool)
    ):
        raise ToolboxError(
            "invalid_run_agent_request",
            "run_agent requires a non-empty prompt and optional boolean detach",
        )
    try:
        prompt_size = len(prompt.encode("utf-8"))
    except UnicodeError as error:
        raise ToolboxError(
            "invalid_run_agent_request", "run_agent prompt is not valid UTF-8"
        ) from error
    if prompt_size > MAX_MODEL_REQUEST_BYTES:
        raise ToolboxError(
            "invalid_run_agent_request",
            f"run_agent prompt exceeds {MAX_MODEL_REQUEST_BYTES} bytes",
        )
    return ("start" if detach else "sync"), {"prompt": prompt}


class RunCancelled(ToolboxError):
    """Raised when the user stops the active rollout."""

    def __init__(self) -> None:
        super().__init__("cancelled", "rollout stopped")


class _CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise RunCancelled


class _AgentLaunchBudget:
    """One cumulative launch allowance shared by a recursive rollout tree."""

    def __init__(self) -> None:
        self._launches = 0
        self._lock = Lock()

    def consume(self) -> None:
        with self._lock:
            if self._launches >= _MAX_AGENT_LAUNCHES_PER_ROLLOUT:
                raise ToolboxError(
                    "agent_launch_limit",
                    (
                        "agent rollout exceeded the maximum "
                        f"{_MAX_AGENT_LAUNCHES_PER_ROLLOUT} recursive launches"
                    ),
                )
            self._launches += 1


class _OrderedEventSink:
    """Deduplicate and serialize session records relayed by parallel runners."""

    def __init__(self, sink: EventSink, next_sequence: int) -> None:
        self._sink = sink
        self._next_sequence = next_sequence
        self._pending: dict[int, AgentEvent] = {}
        self._lock = Lock()

    def __call__(self, event: AgentEvent) -> None:
        with self._lock:
            if event.seq is None:
                self._sink(event)
                return
            if event.seq < self._next_sequence:
                return
            self._pending[event.seq] = event
            while self._next_sequence in self._pending:
                self._sink(self._pending.pop(self._next_sequence))
                self._next_sequence += 1


@dataclass
class _DetachedAgentJob:
    """One supervised background agent and its bounded observable state."""

    job_id: str
    owner_session_id: str
    prompt: str
    conversation: Conversation | None
    budget: _AgentLaunchBudget
    sink: EventSink | None
    status: str = "running"
    result: str | None = None
    error: dict[str, Any] | None = None
    thread: Thread | None = None


class _AgentCoordinator:
    """Create and supervise ordinary child conversations for one harness."""

    def __init__(
        self,
        kernel: Kernel,
        *,
        max_calls_per_turn: int,
        tool_runner: IsolatedToolRunner,
    ) -> None:
        self.kernel = kernel
        self.max_calls_per_turn = max_calls_per_turn
        self.tool_runner = tool_runner
        self._jobs: dict[str, _DetachedAgentJob] = {}
        self._completed: list[str] = []
        self._lock = Lock()
        self._closed = False
        self._active_foreground = 0

    def run_foreground(
        self,
        parent: Conversation,
        prompt: str,
        *,
        parent_scope: InvocationScope | None = None,
        origin_call_id: str | None = None,
    ) -> str:
        """Run one child through the ordinary conversation path and wait."""
        budget = parent._agent_budget_for_launch()
        with self._lock:
            if self._closed:
                raise ToolboxError(
                    "agent_runner_closed",
                    "agent runner is closed",
                )
            if self._active_foreground >= _MAX_ACTIVE_FOREGROUND_AGENTS:
                raise ToolboxError(
                    "foreground_agent_limit",
                    (
                        f"at most {_MAX_ACTIVE_FOREGROUND_AGENTS} foreground agents "
                        "may run"
                    ),
                )
            budget.consume()
            self._active_foreground += 1
        try:
            try:
                child = self._create_child(
                    parent,
                    parent_scope=parent_scope,
                    origin_call_id=origin_call_id,
                )
            except ToolboxError:
                raise
            except Exception as error:
                raise _agent_failure(error) from error
            parent._register_foreground_child(child)
            try:
                return child.send(prompt, _agent_budget=budget).answer
            except RunCancelled:
                raise
            except Exception as error:
                raise _agent_failure(error) from error
            finally:
                parent._unregister_foreground_child(child)
                child.close()
        finally:
            with self._lock:
                self._active_foreground -= 1

    def start_detached(
        self,
        parent: Conversation,
        prompt: str,
        *,
        sink: EventSink | None,
    ) -> dict[str, Any]:
        """Create a child synchronously, then run it in a supervised thread."""
        if not parent.model_session.supports_detached_agents:
            raise ToolboxError(
                "detached_agents_unavailable",
                "the model provider cannot isolate detached agent cancellation",
            )
        budget = parent._agent_budget_for_launch()
        with self._lock:
            if self._closed:
                raise ToolboxError(
                    "detached_agent_runner_closed",
                    "detached agent runner is closed",
                )
            active = sum(job.status == "running" for job in self._jobs.values())
            if active >= _MAX_DETACHED_AGENT_JOBS:
                raise ToolboxError(
                    "detached_agent_limit",
                    f"at most {_MAX_DETACHED_AGENT_JOBS} detached agents may run",
                )
            budget.consume()
            child = self._create_child(parent)
            job = _DetachedAgentJob(
                child.session_id,
                parent.session_id,
                prompt,
                child,
                budget,
                sink,
            )
            thread = Thread(
                target=self._run_detached,
                args=(job,),
                name=f"mechagnome-agent-{job.job_id[:8]}",
                daemon=True,
            )
            job.thread = thread
            self._jobs[job.job_id] = job
            try:
                thread.start()
            except Exception as error:
                self._jobs.pop(job.job_id, None)
                child.close()
                raise ToolboxError(
                    "detached_agent_start_failed",
                    "could not start detached agent",
                ) from error
        return {"job_id": job.job_id, "status": "running"}

    def inspect(self, job_id: str, *, session_id: str) -> dict[str, Any]:
        """Inspect a job from its creator conversation or one of its ancestors."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not self._is_owner_or_ancestor(
                session_id, job.owner_session_id
            ):
                raise ToolboxError(
                    "unknown_detached_agent", f"unknown detached agent: {job_id}"
                )
            snapshot: dict[str, Any] = {
                "job_id": job.job_id,
                "status": job.status,
            }
            if job.status == "succeeded":
                snapshot["result"] = job.result
            elif job.status == "failed":
                snapshot["error"] = dict(job.error or {})
            return snapshot

    def close(self) -> None:
        """Reject new jobs, cancel active detached agents, and join them."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.status == "running" and job.conversation is not None:
                job.conversation.close()
        for job in jobs:
            thread = job.thread
            if (
                thread is not None
                and thread.ident is not None
                and thread is not current_thread()
            ):
                thread.join()

    def _create_child(
        self,
        parent: Conversation,
        *,
        parent_scope: InvocationScope | None = None,
        origin_call_id: str | None = None,
    ) -> Conversation:
        provider = parent.model_session.provider
        if not provider.allow_tool_agents:
            raise ToolboxError(
                "model_provider_unavailable",
                "no model provider is available to run another agent",
            )
        scope = parent_scope or self.kernel.snapshot_scope(parent.session_id)
        model_session = provider.start_session(
            parent_scope=scope,
            origin_call_id=origin_call_id,
        )
        return Conversation(
            self.kernel,
            model_session,
            max_calls_per_turn=self.max_calls_per_turn,
            tool_runner=self.tool_runner,
            _agent_coordinator=self,
        )

    def _run_detached(self, job: _DetachedAgentJob) -> None:
        self._notify(job, "detached_started", self._event_snapshot(job))
        try:
            result = job.conversation.send(job.prompt, _agent_budget=job.budget).answer
        except Exception as error:
            if self._closed:
                failure = {
                    "code": "detached_agent_shutdown",
                    "message": "detached agent stopped because its harness closed",
                    "details": {},
                }
            else:
                failure = _agent_failure(error).to_dict()["error"]
            self._finish(job, status="failed", error=failure)
        else:
            self._finish(job, status="succeeded", result=result)
        finally:
            if job.conversation is not None:
                job.conversation.close()
            with self._lock:
                job.conversation = None
                job.prompt = ""
                job.thread = None

    def _finish(
        self,
        job: _DetachedAgentJob,
        *,
        status: str,
        result: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        if status == "succeeded":
            encoded = json.dumps(result, separators=(",", ":")).encode("utf-8")
            if len(encoded) > _MAX_DETACHED_AGENT_RESULT_BYTES:
                status = "failed"
                result = None
                error = {
                    "code": "detached_agent_result_too_large",
                    "message": "detached agent result exceeds the retained byte limit",
                    "details": {"limit_bytes": _MAX_DETACHED_AGENT_RESULT_BYTES},
                }
        with self._lock:
            if job.status != "running":
                return
            job.status = status
            job.result = result
            job.error = error
            self._completed.append(job.job_id)
            payload = self._event_snapshot_locked(job)
        self._notify(job, "detached_finished", payload)
        with self._lock:
            job.sink = None
            while len(self._completed) > _MAX_RETAINED_DETACHED_AGENT_JOBS:
                evicted = self._completed.pop(0)
                self._jobs.pop(evicted, None)

    def _is_owner_or_ancestor(self, candidate: str, owner: str) -> bool:
        current: str | None = owner
        while current is not None:
            if current == candidate:
                return True
            current = self.kernel.session_metadata(current)["parent_session_id"]
        return False

    def _event_snapshot(self, job: _DetachedAgentJob) -> dict[str, Any]:
        with self._lock:
            return self._event_snapshot_locked(job)

    @staticmethod
    def _event_snapshot_locked(job: _DetachedAgentJob) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": job.job_id,
            "job_kind": "agent",
            "name": RUN_AGENT_NAME,
            "args": {"prompt": job.prompt},
            "status": job.status,
            "output_tail": "",
            "truncated": False,
        }
        if job.status == "succeeded":
            payload["result"] = job.result
        elif job.status == "failed":
            payload["error"] = dict(job.error or {})
        return payload

    @staticmethod
    def _notify(job: _DetachedAgentJob, kind: str, payload: dict[str, Any]) -> None:
        if job.sink is None:
            return
        try:
            job.sink(AgentEvent(kind, payload, None))
        except Exception:
            pass


def _agent_failure(error: Exception) -> ToolboxError:
    """Normalize child failures before exposing them to a parent conversation."""
    if isinstance(error, ModelTransportError):
        public = error.public_error()
        return ToolboxError(
            public["code"],
            public["message"],
            **public.get("details", {}),
        )
    return ToolboxError("model_provider_failed", "model provider request failed")


class Conversation:
    """A durable, multi-prompt model conversation over one toolbox session."""

    def __init__(
        self,
        kernel: Kernel,
        model_session: ModelSession,
        *,
        max_calls_per_turn: int,
        tool_runner: IsolatedToolRunner,
        _agent_coordinator: _AgentCoordinator,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.kernel = kernel
        self.model_session = model_session
        self.session_id = model_session.session_id
        self.max_calls_per_turn = max_calls_per_turn
        self.tool_runner = tool_runner
        self._agent_coordinator = _agent_coordinator
        self.messages = messages or []
        self._run_lock = Lock()
        self._current_token: _CancellationToken | None = None
        self._current_agent_budget: _AgentLaunchBudget | None = None
        self._foreground_children: set[Conversation] = set()
        self._closed = False

    def send(
        self,
        prompt: str,
        *,
        on_event: EventSink | None = None,
        _agent_budget: _AgentLaunchBudget | None = None,
    ) -> RunResult:
        """Send one user message and run until the model returns final text."""
        token = _CancellationToken()
        with self._run_lock:
            if self._closed:
                raise RunCancelled
            if self._current_token is not None:
                raise ToolboxError("conversation_busy", "a rollout is already active")
            self._current_token = token
            self._current_agent_budget = _agent_budget or _AgentLaunchBudget()
        message_start = len(self.messages)
        try:
            return self._run(prompt, on_event, token)
        except RunCancelled as error:
            del self.messages[message_start:]
            self._append(on_event, "cancelled", error.to_dict()["error"])
            raise
        finally:
            with self._run_lock:
                if self._current_token is token:
                    try:
                        self.model_session.reset_cancellation()
                    except Exception:
                        pass
                    self._current_token = None
                    self._current_agent_budget = None

    def _agent_budget_for_launch(self) -> _AgentLaunchBudget:
        """Return the current rollout budget or one for a direct host launch."""
        with self._run_lock:
            return self._current_agent_budget or _AgentLaunchBudget()

    def cancel(self) -> bool:
        """Request cancellation of the active model or tool rollout."""
        with self._run_lock:
            token = self._current_token
            if token is None:
                return False
            self._cancel_locked(token)
            return True

    def close(self) -> None:
        """Prevent future rollouts and cancel one that is starting or active."""
        with self._run_lock:
            if self._closed:
                return
            self._closed = True
            if self._current_token is not None:
                self._cancel_locked(self._current_token)
            else:
                for child in tuple(self._foreground_children):
                    child.close()
        self.model_session.close()

    def _cancel_locked(self, token: _CancellationToken) -> None:
        token.cancel()
        for child in tuple(self._foreground_children):
            child.close()
        try:
            self.model_session.cancel_current()
        except Exception:
            pass

    def _register_foreground_child(self, child: Conversation) -> None:
        with self._run_lock:
            if self._closed or (
                self._current_token is not None and self._current_token.cancelled
            ):
                child.close()
                raise RunCancelled
            self._foreground_children.add(child)

    def _unregister_foreground_child(self, child: Conversation) -> None:
        with self._run_lock:
            self._foreground_children.discard(child)

    def _run(
        self,
        prompt: str,
        on_event: EventSink | None,
        token: _CancellationToken,
    ) -> RunResult:
        self.messages.append({"role": "user", "content": prompt})
        self._append(on_event, "user", {"content": prompt})

        for turn_number in count(1):
            token.check()
            tools = [
                *self.kernel.tool_definitions(session_id=self.session_id),
                RUN_AGENT_DEFINITION,
            ]
            try:
                turn = self._respond(tools, on_event, token)
            except Exception as error:
                if token.cancelled or isinstance(error, RunCancelled):
                    raise RunCancelled from error
                raise
            token.check()
            calls = [
                {"id": call.id, "name": call.name, "args": call.args}
                for call in turn.calls
            ]
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": turn.text,
                "tool_calls": calls,
            }
            if turn.reasoning:
                assistant_message["reasoning"] = turn.reasoning
            if turn.reasoning_details:
                details = list(turn.reasoning_details)
                assistant_message["reasoning_details"] = details
            self.messages.append(assistant_message)
            if not turn.calls:
                answer = turn.text or ""
                return RunResult(self.session_id, answer, turn_number)

            if len(turn.calls) > self.max_calls_per_turn:
                observations: list[JsonValue] = [
                    {
                        "error": {
                            "code": "too_many_tool_calls",
                            "message": (
                                f"model requested {len(turn.calls)} operations in one "
                                f"turn; maximum is {self.max_calls_per_turn}; split "
                                "independent work into smaller batches"
                            ),
                        }
                    }
                    for _ in turn.calls
                ]
            else:
                observations = asyncio.run(
                    self._execute_all(tuple(turn.calls), on_event, token)
                )
            for call, observation in zip(turn.calls, observations, strict=True):
                token.check()
                self._append(
                    on_event,
                    "tool_observation",
                    {"model_call_id": call.id, "observation": observation},
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": observation,
                    }
                )

    async def _execute_all(
        self,
        calls: tuple[ToolCall, ...],
        sink: EventSink | None,
        token: _CancellationToken,
    ) -> list[JsonValue]:
        """Run one model-requested tool batch concurrently."""
        ordered_sink = (
            _OrderedEventSink(
                sink, self.kernel.latest_event_sequence(self.session_id) + 1
            )
            if sink is not None and len(calls) > 1
            else sink
        )
        return list(
            await asyncio.gather(
                *(
                    asyncio.to_thread(self._execute, call, ordered_sink, token)
                    for call in calls
                )
            )
        )

    def _respond(
        self,
        tools: list[dict[str, Any]],
        sink: EventSink | None,
        token: _CancellationToken,
    ) -> ModelTurn:
        token.check()
        self._emit_transient(sink, "model_started", {})
        completed: ModelTurn | None = None
        for event in self.model_session._stream_with_recorded_input(
            self.messages,
            tools,
            on_record=lambda kind, payload, seq: self._emit_model_record(
                sink, kind, payload, seq
            ),
            cancelled=lambda: token.cancelled,
        ):
            token.check()
            if event.text_delta:
                self._emit_transient(sink, "model_delta", {"text": event.text_delta})
            if event.turn is not None:
                if completed is not None:
                    raise ToolboxError(
                        "invalid_model_stream", "model stream completed more than once"
                    )
                completed = event.turn
        if completed is None:
            raise ToolboxError(
                "invalid_model_stream", "model stream ended without a completed turn"
            )
        return completed

    @staticmethod
    def _emit_transient(
        sink: EventSink | None, kind: str, payload: dict[str, Any]
    ) -> None:
        if sink is not None:
            sink(AgentEvent(kind, payload, None))

    def _execute(
        self,
        call: ToolCall,
        sink: EventSink | None,
        token: _CancellationToken,
    ) -> JsonValue:
        if call.name not in MODEL_ACTION_NAMES:
            return {
                "error": {
                    "code": "unknown_operation",
                    "message": (
                        f"the model may only call: {', '.join(MODEL_ACTION_NAMES)}"
                    ),
                }
            }
        try:
            if call.name == RUN_AGENT_NAME:
                mode, agent_args = _parse_run_agent_request(call.args)
                if mode == "inspect":
                    return self._agent_coordinator.inspect(
                        agent_args["job_id"], session_id=self.session_id
                    )
                if mode == "start":
                    return self._agent_coordinator.start_detached(
                        self,
                        agent_args["prompt"],
                        sink=sink,
                    )
                return self._agent_coordinator.run_foreground(
                    self, agent_args["prompt"]
                )

            effective_args = call.args
            if call.name == "call_tool":
                mode, effective_args = _parse_call_tool_request(call.args)
                if mode == "inspect":
                    inspect = getattr(self.tool_runner, "inspect_detached", None)
                    if not callable(inspect):
                        raise ToolboxError(
                            "detached_jobs_unavailable",
                            "this tool runner does not support detached jobs",
                        )
                    return inspect(effective_args["job_id"], session_id=self.session_id)
                if mode == "start":
                    start = getattr(self.tool_runner, "start_detached", None)
                    if not callable(start):
                        raise ToolboxError(
                            "detached_jobs_unavailable",
                            "this tool runner does not support detached jobs",
                        )
                    return start(
                        effective_args["name"],
                        effective_args["args"],
                        version=effective_args.get("version"),
                        session_id=self.session_id,
                        on_update=lambda kind, payload: self._emit_transient(
                            sink, kind, payload
                        ),
                    )

            call_with_provider = getattr(
                self.tool_runner,
                "call_with_model_provider",
                None,
            )
            call_kwargs = {
                "session_id": self.session_id,
                "on_event": lambda event: self._emit_record(sink, event),
                "cancelled": lambda: token.cancelled,
            }
            if callable(call_with_provider):
                return call_with_provider(
                    call.name,
                    effective_args,
                    model_provider=self.model_session.completion_provider(
                        agent_runner=self._run_agent_from_tool
                    ),
                    **call_kwargs,
                )
            return self.tool_runner.call(call.name, effective_args, **call_kwargs)
        except ToolboxError as error:
            if token.cancelled or error.code == "cancelled":
                raise RunCancelled from error
            return error.to_dict()
        except Exception as error:  # Generated tools may raise anything.
            return {"error": {"code": "tool_failed", "message": str(error)}}

    def _run_agent_from_tool(
        self,
        parent_scope: InvocationScope,
        origin_call_id: str | None,
        prompt: str,
    ) -> str:
        return self._agent_coordinator.run_foreground(
            self,
            prompt,
            parent_scope=parent_scope,
            origin_call_id=origin_call_id,
        )

    def _append(
        self,
        sink: EventSink | None,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        sequence = self.kernel.append_event(self.session_id, kind, payload)
        if sink is not None:
            sink(AgentEvent(kind, payload, sequence))

    @staticmethod
    def _emit_record(sink: EventSink | None, event: dict[str, Any]) -> None:
        if sink is not None:
            sink(
                AgentEvent(
                    kind=event["kind"],
                    payload=event["payload"],
                    seq=event["seq"],
                    call_id=event["call_id"],
                    parent_call_id=event["parent_call_id"],
                    toolbox_id=event["toolbox_id"],
                    tool_name=event["tool_name"],
                    tool_version=event["tool_version"],
                    tool_version_id=event["tool_version_id"],
                )
            )

    @staticmethod
    def _emit_model_record(
        sink: EventSink | None,
        kind: str,
        payload: dict[str, Any],
        sequence: int,
    ) -> None:
        if sink is not None:
            sink(AgentEvent(kind, payload, sequence))


class Harness:
    """Drive a model over the editable toolbox and host action surface."""

    def __init__(
        self,
        kernel: Kernel,
        *,
        max_calls_per_turn: int = DEFAULT_MAX_CALLS_PER_TURN,
        tool_runner: IsolatedToolRunner | None = None,
    ) -> None:
        self.kernel = kernel
        self.max_calls_per_turn = max_calls_per_turn
        self.tool_runner = tool_runner or IsolatedToolRunner(kernel)
        self._agent_coordinator = _AgentCoordinator(
            kernel,
            max_calls_per_turn=max_calls_per_turn,
            tool_runner=self.tool_runner,
        )

    def close(self) -> None:
        """Release background agent and tool work owned by this harness."""
        self._agent_coordinator.close()
        close = getattr(self.tool_runner, "close", None)
        if callable(close):
            close()

    def start(
        self,
        model: Model | ModelProvider,
        *,
        session_id: str | None = None,
        model_provider: CompletionTransport | None = None,
    ) -> Conversation:
        """Start a persistent conversation suitable for a chat interface."""
        if isinstance(model, ModelProvider):
            if model.kernel is not self.kernel:
                raise ToolboxError(
                    "invalid_model_provider",
                    "model provider belongs to a different kernel",
                )
            if model_provider is not None:
                raise ToolboxError(
                    "invalid_model_provider",
                    "a model provider cannot be combined with a second provider",
                )
            provider = model
        else:
            provider = ModelProvider.from_transport(
                self.kernel,
                model,
                completions=model_provider,
            )
        model_session = provider.start_session(session_id=session_id)
        identifier = model_session.session_id
        return Conversation(
            self.kernel,
            model_session,
            max_calls_per_turn=self.max_calls_per_turn,
            tool_runner=self.tool_runner,
            _agent_coordinator=self._agent_coordinator,
            messages=self._session_messages(identifier),
        )

    def _session_messages(self, session_id: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        call_names: dict[str, str] = {}
        prompt_start: int | None = None
        after = 0
        while True:
            page = self.kernel.read_session(session_id, after=after, limit=100)
            for event in page["events"]:
                after = event["seq"]
                payload = event["payload"]
                if event["kind"] == "user":
                    prompt_start = len(messages)
                    messages.append({"role": "user", "content": payload["content"]})
                elif event["kind"] == "model":
                    calls = payload.get("calls") or []
                    for call in calls:
                        call_names[call["id"]] = call["name"]
                    assistant_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": payload.get("text"),
                        "tool_calls": calls,
                    }
                    if payload.get("reasoning"):
                        assistant_message["reasoning"] = payload["reasoning"]
                    if payload.get("reasoning_details"):
                        assistant_message["reasoning_details"] = payload[
                            "reasoning_details"
                        ]
                    messages.append(assistant_message)
                elif event["kind"] == "tool_observation":
                    call_id = payload["model_call_id"]
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": call_names.get(call_id),
                            "content": payload["observation"],
                        }
                    )
                elif event["kind"] == "final":
                    prompt_start = None
                elif event["kind"] == "cancelled" and prompt_start is not None:
                    del messages[prompt_start:]
                    prompt_start = None
            if page["next_after"] is None:
                return messages

    def run(
        self,
        model: Model | ModelProvider,
        prompt: str,
        *,
        session_id: str | None = None,
        model_provider: CompletionTransport | None = None,
    ) -> RunResult:
        """Run until the model returns a turn with no operation calls."""
        conversation = self.start(
            model,
            session_id=session_id,
            model_provider=model_provider,
        )
        try:
            return conversation.send(prompt)
        finally:
            conversation.close()
