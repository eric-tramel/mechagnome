"""Provider-neutral model loop over the five fixed toolbox operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any

from mechagnome.bootstrap import CORE_NAMES
from mechagnome.isolation import IsolatedToolRunner
from mechagnome.kernel import JsonValue, Kernel, ToolboxError
from mechagnome.model_provider import (
    CompletionTransport,
    ModelProvider,
    ModelSession,
    ModelStreamEvent,  # noqa: F401 - compatibility re-export
    ModelTransport,
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


class Conversation:
    """A durable, multi-prompt model conversation over one toolbox session."""

    def __init__(
        self,
        kernel: Kernel,
        model_session: ModelSession,
        *,
        max_turns: int,
        max_calls_per_turn: int,
        tool_runner: IsolatedToolRunner,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.kernel = kernel
        self.model_session = model_session
        self.session_id = model_session.session_id
        self.max_turns = max_turns
        self.max_calls_per_turn = max_calls_per_turn
        self.tool_runner = tool_runner
        self.messages = messages or []
        self._run_lock = Lock()
        self._current_token: _CancellationToken | None = None
        self._closed = False

    def send(self, prompt: str, *, on_event: EventSink | None = None) -> RunResult:
        """Send one user message and run until the model returns final text."""
        token = _CancellationToken()
        with self._run_lock:
            if self._closed:
                raise RunCancelled
            if self._current_token is not None:
                raise ToolboxError("conversation_busy", "a rollout is already active")
            self._current_token = token
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
            self._cancel_locked(self._current_token)

    def _cancel_locked(self, token: _CancellationToken | None) -> None:
        if token is not None:
            token.cancel()
        try:
            self.model_session.cancel_current()
        except Exception:
            pass

    def _run(
        self,
        prompt: str,
        on_event: EventSink | None,
        token: _CancellationToken,
    ) -> RunResult:
        self.messages.append({"role": "user", "content": prompt})
        self._append(on_event, "user", {"content": prompt})

        for turn_number in range(1, self.max_turns + 1):
            token.check()
            tools = self.kernel.tool_definitions(session_id=self.session_id)
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

            oversized_batch = len(turn.calls) > self.max_calls_per_turn
            for call in turn.calls:
                if oversized_batch:
                    observation: JsonValue = {
                        "error": {
                            "code": "too_many_tool_calls",
                            "message": (
                                f"model requested {len(turn.calls)} operations in one "
                                f"turn; maximum is {self.max_calls_per_turn}; split "
                                "independent work into smaller batches"
                            ),
                        }
                    }
                else:
                    observation = self._execute(call, on_event, token)
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

        error = ToolboxError(
            "max_turns", f"model exceeded the maximum {self.max_turns} turns"
        )
        self._append(on_event, "harness_failed", error.to_dict())
        raise error

    def _respond(
        self,
        tools: list[dict[str, Any]],
        sink: EventSink | None,
        token: _CancellationToken,
    ) -> ModelTurn:
        token.check()
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
        if call.name not in CORE_NAMES:
            return {
                "error": {
                    "code": "unknown_operation",
                    "message": f"the model may only call: {', '.join(CORE_NAMES)}",
                }
            }
        try:
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
                    call.args,
                    model_provider=self.model_session.completion_provider(
                        agent_runner=self._run_child_agent
                    ),
                    **call_kwargs,
                )
            return self.tool_runner.call(call.name, call.args, **call_kwargs)
        except ToolboxError as error:
            if token.cancelled or error.code == "cancelled":
                raise RunCancelled from error
            return error.to_dict()
        except Exception as error:  # Generated tools may raise anything.
            return {"error": {"code": "tool_failed", "message": str(error)}}

    def _run_child_agent(self, model_session: ModelSession, prompt: str) -> str:
        child = Conversation(
            self.kernel,
            model_session,
            max_turns=self.max_turns,
            max_calls_per_turn=self.max_calls_per_turn,
            tool_runner=self.tool_runner,
        )
        return child.send(prompt).answer

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
    """Drive a model that can only see the five metaprogramming operations."""

    def __init__(
        self,
        kernel: Kernel,
        *,
        max_turns: int = 50,
        max_calls_per_turn: int = DEFAULT_MAX_CALLS_PER_TURN,
        tool_runner: IsolatedToolRunner | None = None,
    ) -> None:
        self.kernel = kernel
        self.max_turns = max_turns
        self.max_calls_per_turn = max_calls_per_turn
        self.tool_runner = tool_runner or IsolatedToolRunner(kernel)

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
            max_turns=self.max_turns,
            max_calls_per_turn=self.max_calls_per_turn,
            tool_runner=self.tool_runner,
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
        return self.start(
            model,
            session_id=session_id,
            model_provider=model_provider,
        ).send(prompt)
