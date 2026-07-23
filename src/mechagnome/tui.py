"""Textual interface for a persistent toolbox agent conversation."""

from __future__ import annotations

import base64
import binascii
import difflib
import json
import math
import re
import shlex
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from threading import Lock
from typing import Any

from PIL import Image as PILImage
from rich.cells import split_graphemes
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Unmount
from textual.screen import ModalScreen, Screen
from textual.timer import Timer
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual_image.widget import Image as TerminalImage

from mechagnome.harness import AgentEvent, Conversation, Harness, Model, RunCancelled
from mechagnome.kernel import Kernel
from mechagnome.model_provider import CompletionTransport, ModelProvider
from mechagnome.openrouter import (
    OpenRouterError,
    OpenRouterModel,
    OpenRouterModelOption,
)

MODEL_INPUT_EMOJIS = (("image", "🖼️"), ("audio", "🎧"))
MAX_TOOL_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOOL_IMAGE_PIXELS = 16 * 1024 * 1024
MAX_TOOL_IMAGE_TOTAL_PIXELS = MAX_TOOL_IMAGE_PIXELS
MAX_TOOL_IMAGES = 8
MAX_TOOL_IMAGE_CONTAINER_CHARS = 16 * 1024 * 1024
TOOL_IMAGE_MARKDOWN_PATTERN = re.compile(
    r"!\[(?P<alt>[^\]\r\n]*)\]\("
    r"(?P<uri>data:(?P<mime>image/[A-Za-z0-9.+-]+);base64,"
    r"[A-Za-z0-9+/]+={0,2})\)"
)


def _format_duration(value: Any) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        return ""
    if value < 0.1:
        return "<0.1 ms"
    if value < 10:
        return f"{value:.2f} ms"
    if value < 1000:
        return f"{value:.1f} ms"
    return f"{value / 1000:.2f} s"


def _compact_json(value: Any, limit: int = 1600) -> str:
    try:
        rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = repr(value)
    return rendered if len(rendered) <= limit else f"{rendered[:limit]}\n…"


def _decode_tool_image(
    value: Any, *, max_pixels: int | None = None
) -> PILImage.Image | None:
    """Decode one MCP-style image content block, ignoring malformed data."""
    if not isinstance(value, dict) or value.get("type") != "image":
        return None
    encoded = value.get("data")
    media_type = value.get("mimeType", value.get("mime_type", ""))
    if not isinstance(encoded, str) or not isinstance(media_type, str):
        return None

    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header:
            return None
        data_media_type = header[5:].split(";", 1)[0]
        media_type = media_type or data_media_type
    if media_type and not media_type.startswith("image/"):
        return None
    pixel_limit = (
        MAX_TOOL_IMAGE_PIXELS
        if max_pixels is None
        else min(max_pixels, MAX_TOOL_IMAGE_PIXELS)
    )
    if pixel_limit <= 0:
        return None
    maximum_encoded_size = ((MAX_TOOL_IMAGE_BYTES + 2) // 3) * 4
    if len(encoded) > maximum_encoded_size:
        return None

    try:
        decoded = base64.b64decode(encoded, validate=True)
        if len(decoded) > MAX_TOOL_IMAGE_BYTES:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("error", PILImage.DecompressionBombWarning)
            with PILImage.open(BytesIO(decoded)) as source:
                if source.width * source.height > pixel_limit:
                    return None
                source.load()
                return source.copy()
    except (
        binascii.Error,
        OSError,
        ValueError,
        PILImage.DecompressionBombError,
        PILImage.DecompressionBombWarning,
    ):
        return None


def _structured_tool_string(value: Any) -> dict[str, Any] | list[Any] | None:
    """Decode a JSON-stringified structured result without parsing arbitrary text."""
    if not isinstance(value, str) or len(value) > MAX_TOOL_IMAGE_CONTAINER_CHARS:
        return None
    stripped = value.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, RecursionError):
        return None
    return decoded if isinstance(decoded, (dict, list)) else None


def _tool_images(value: Any) -> tuple[PILImage.Image, ...]:
    """Find valid image content blocks anywhere in a JSON tool result."""
    images: list[PILImage.Image] = []
    total_pixels = 0
    pending = [value]
    while pending and len(images) < MAX_TOOL_IMAGES:
        item = pending.pop()
        decoded = _decode_tool_image(
            item,
            max_pixels=MAX_TOOL_IMAGE_TOTAL_PIXELS - total_pixels,
        )
        if decoded is not None:
            images.append(decoded)
            total_pixels += decoded.width * decoded.height
        elif structured := _structured_tool_string(item):
            pending.append(structured)
        elif isinstance(item, str) and len(item) <= MAX_TOOL_IMAGE_CONTAINER_CHARS:
            for match in TOOL_IMAGE_MARKDOWN_PATTERN.finditer(item):
                decoded = _decode_tool_image(
                    {
                        "type": "image",
                        "data": match.group("uri"),
                        "mimeType": match.group("mime"),
                    },
                    max_pixels=MAX_TOOL_IMAGE_TOTAL_PIXELS - total_pixels,
                )
                if decoded is not None:
                    images.append(decoded)
                    total_pixels += decoded.width * decoded.height
                    if len(images) >= MAX_TOOL_IMAGES:
                        break
        elif isinstance(item, dict):
            pending.extend(reversed(tuple(item.values())))
        elif isinstance(item, list):
            pending.extend(reversed(item))
    return tuple(images)


def _summarize_tool_images(value: Any) -> Any:
    """Keep image metadata in textual details without dumping its base64 body."""
    if isinstance(value, dict):
        summarized = {key: _summarize_tool_images(item) for key, item in value.items()}
        if value.get("type") == "image" and isinstance(value.get("data"), str):
            media_type = value.get("mimeType", value.get("mime_type", "image"))
            summarized["data"] = f"<{media_type} data omitted>"
        return summarized
    if isinstance(value, list):
        return [_summarize_tool_images(item) for item in value]
    if structured := _structured_tool_string(value):
        return _summarize_tool_images(structured)
    if isinstance(value, str):
        if len(value) > MAX_TOOL_IMAGE_CONTAINER_CHARS:
            return "<tool response omitted: exceeds image container limit>"
        return TOOL_IMAGE_MARKDOWN_PATTERN.sub(
            lambda match: (
                f"![{match.group('alt')}](<{match.group('mime')} data omitted>)"
            ),
            value,
        )
    return value


class ToolEvent(Collapsible):
    """One quiet, expandable tool invocation or observation."""

    SYMBOLS = {"call": "→", "response": "✓", "error": "✕"}
    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    MAX_DETACHED_TAIL_BYTES = 16 * 1024

    def __init__(
        self,
        kind: str,
        tool_name: str,
        detail: str,
        argument_summary: str = "",
        outcome_summary: str = "",
    ) -> None:
        self.kind = kind
        self.tool_name = tool_name
        self.detail = detail
        self.call_detail = detail
        self.argument_summary = argument_summary
        self.outcome_summary = outcome_summary
        self.detached_job_id: str | None = None
        self.detached_output_tail = ""
        self.detached_output_truncated = False
        self.processing = kind == "call"
        self.detail_widget = Static(
            Text(detail, style="dim"), classes="tool-event-detail"
        )
        super().__init__(
            self.detail_widget,
            title=self._render_title(self.SYMBOLS[kind]),
            collapsed=True,
            classes=f"tool-event tool-{kind}",
        )
        self._spinner_index = 0
        self._spinner_timer = None

    def on_mount(self) -> None:
        """Start the spinner animation for in-progress calls."""
        if self.processing:
            self._start_spinner()

    def on_unmount(self) -> None:
        """Release the widget-owned animation timer."""
        self.stop_spinner()

    def _start_spinner(self) -> None:
        self._spinner_timer = self.set_interval(0.08, self._spin)

    def _spin(self) -> None:
        if not self.processing:
            self.stop_spinner()
            return
        frame = self.SPINNER_FRAMES[self._spinner_index % len(self.SPINNER_FRAMES)]
        self._spinner_index += 1
        self.title = self._render_title(frame)

    def stop_spinner(self) -> None:
        """Stop the spinner and restore the static symbol."""
        self.processing = False
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
        self.title = self._render_title(self.SYMBOLS[self.kind])

    def update_call(
        self, tool_name: str, detail: str, argument_summary: str = ""
    ) -> None:
        """Replace a pending dispatcher row with its confirmed target call."""
        self.tool_name = tool_name
        self.detail = detail
        self.call_detail = detail
        self.argument_summary = argument_summary
        self.detail_widget.update(Text(detail, style="dim"))
        if self._spinner_timer is None:
            self.title = self._render_title(self.SYMBOLS["call"])

    def finish(
        self, kind: str, tool_name: str, detail: str, outcome_summary: str = ""
    ) -> None:
        """Turn an in-progress invocation into its final outcome in place."""
        call_detail = self.detail
        self.kind = kind
        self.tool_name = tool_name
        self.outcome_summary = outcome_summary
        self.remove_class("tool-call")
        self.add_class(f"tool-{kind}")
        outcome = "response" if kind == "response" else "error"
        self.detail = f"arguments\n{call_detail}\n\n{outcome}\n{detail}"
        self.detail_widget.update(Text(self.detail, style="dim"))
        self.stop_spinner()

    def attach_detached(self, job_id: str) -> None:
        """Keep this call active under one process-lifetime detached handle."""
        self.detached_job_id = job_id
        self.detail = self._detached_detail()
        self.detail_widget.update(Text(self.detail, style="dim"))

    def set_detached_output(self, text: str, *, truncated: bool = False) -> None:
        """Replace the displayed output with the supervisor's latest tail."""
        encoded = text.encode("utf-8")
        if len(encoded) > self.MAX_DETACHED_TAIL_BYTES:
            encoded = encoded[-self.MAX_DETACHED_TAIL_BYTES :]
            while encoded and encoded[0] & 0xC0 == 0x80:
                encoded = encoded[1:]
            truncated = True
        self.detached_output_truncated = truncated
        self.detached_output_tail = encoded.decode("utf-8", errors="replace")
        self.detail = self._detached_detail()
        self.detail_widget.update(Text(self.detail, style="dim"))

    def finish_detached(self, payload: dict[str, Any]) -> None:
        """Finalize one detached row with its final tail and typed outcome."""
        self.set_detached_output(
            str(payload.get("output_tail") or ""),
            truncated=bool(payload.get("truncated")),
        )
        succeeded = payload.get("status") == "succeeded"
        self.kind = "response" if succeeded else "error"
        self.remove_class("tool-call")
        self.add_class(f"tool-{self.kind}")
        label = "response" if succeeded else "error"
        value = payload.get("result") if succeeded else payload.get("error")
        if succeeded:
            value = _summarize_tool_images(value)
        self.detail = f"{self._detached_detail()}\n\n{label}\n{_compact_json(value)}"
        self.detail_widget.update(Text(self.detail, style="dim"))
        self.stop_spinner()

    def _detached_detail(self) -> str:
        output = self.detached_output_tail or "(waiting for output)"
        if self.detached_output_truncated:
            output = f"… output truncated to latest tail …\n{output}"
        return (
            f"arguments\n{self.call_detail}\n\n"
            f"job\n{self.detached_job_id or 'starting'}\n\noutput\n{output}"
        )

    def _render_title(self, symbol: str) -> str:
        return escape(
            f"{symbol} {self.tool_name}{self.argument_summary}{self.outcome_summary}"
        )


class ModelActivity(Static):
    """One animated line showing the latest model output while it runs."""

    SPINNER_FRAMES = ToolEvent.SPINNER_FRAMES
    MAX_PREVIEW_CHARS = 512

    def __init__(self) -> None:
        self._spinner_index = 0
        self._animation_timer: Timer | None = None
        self._active = True
        self._current_line = ""
        self._latest_line = ""
        super().__init__(
            self._render_line(width=80),
            classes="chat-entry streaming-response",
        )

    def on_mount(self) -> None:
        """Start animating unless a fast completion already finalized the row."""
        if not self._active:
            return
        self._animation_timer = self.set_interval(0.08, self._spin)
        self._refresh_line()

    def on_unmount(self) -> None:
        """Stop the widget-owned timer when its chat pane is removed."""
        self.stop_animation()

    def append_text(self, text: str) -> None:
        """Incrementally retain the newest non-empty logical output line."""
        if not text:
            return
        parts = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self._current_line = self._bounded(self._current_line + parts[0])
        if self._current_line.strip():
            self._latest_line = self._current_line
        for part in parts[1:]:
            self._current_line = self._bounded(part)
            if self._current_line.strip():
                self._latest_line = self._current_line
        self._refresh_line()

    def finish(self, renderable: Any) -> None:
        """Turn this transient row into the normal completed response entry."""
        self.stop_animation()
        self.remove_class("streaming-response")
        self.update(renderable)

    def stop_animation(self) -> None:
        """Idempotently stop future animation updates."""
        self._active = False
        if self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer = None

    def _spin(self) -> None:
        if not self._active:
            self.stop_animation()
            return
        self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)
        self._refresh_line()

    def _refresh_line(self) -> None:
        if self._active:
            self.update(self._render_line(), layout=False)

    def _render_line(self, *, width: int | None = None) -> Text:
        frame = self.SPINNER_FRAMES[self._spinner_index]
        message = self._latest_line or "thinking…"
        if width is None:
            width = self.content_size.width or self.size.width or 80
        message = self._cell_suffix(message, max(1, width - 2))
        return Text(
            f"{frame} {message}",
            style="italic bright_magenta",
            no_wrap=True,
            overflow="crop",
        )

    @classmethod
    def _bounded(cls, text: str) -> str:
        if len(text) <= cls.MAX_PREVIEW_CHARS:
            return text
        return text[-cls.MAX_PREVIEW_CHARS :]

    @staticmethod
    def _cell_suffix(text: str, width: int) -> str:
        spans, cell_width = split_graphemes(text)
        if cell_width <= width:
            return text
        if width == 1:
            return "…"
        remaining = width - 1
        start = len(text)
        for span_start, _span_end, span_width in reversed(spans):
            if span_width > remaining:
                break
            start = span_start
            remaining -= span_width
        return f"…{text[start:]}"


class ChatFeed(VerticalScroll):
    """Scrollable chat entries with interactive tool events."""

    def write(self, renderable: Any, *, classes: str = "") -> Static:
        entry_classes = "chat-entry"
        if classes:
            entry_classes += f" {classes}"
        entry = Static(renderable, classes=entry_classes)
        self.mount(entry)
        self.call_after_refresh(self.scroll_end, animate=False)
        return entry

    def write_activity(self) -> ModelActivity:
        """Mount one animated model-activity row at the end of the feed."""
        entry = ModelActivity()
        self.mount(entry)
        self.call_after_refresh(self.scroll_end, animate=False)
        return entry

    def write_image(self, image: PILImage.Image) -> TerminalImage:
        """Mount an image returned by a tool as a visible chat entry."""
        entry = TerminalImage(image, classes="chat-entry tool-response-image")
        self.mount(entry)
        self.call_after_refresh(self.scroll_end, animate=False)
        return entry

    def write_tool(
        self,
        kind: str,
        tool_name: str,
        detail: str,
        argument_summary: str = "",
        outcome_summary: str = "",
    ) -> ToolEvent:
        event = ToolEvent(
            kind,
            tool_name,
            detail,
            argument_summary,
            outcome_summary,
        )
        self.mount(event)
        self.call_after_refresh(self.scroll_end, animate=False)
        return event

    def clear(self) -> None:
        self.remove_children()


@dataclass
class SessionTab:
    """UI and conversation state owned by one session tab."""

    conversation: Conversation
    pane_id: str
    label: str
    chat: ChatFeed
    draft: str = ""
    user_history: list[str] = field(default_factory=list)
    history_index: int | None = None
    history_draft: str = ""
    running: bool = False
    status: str = "ready"
    total_tokens: int | None = None
    context_model: str | None = None
    streamed_text: str = ""
    pending_stream_text: list[str] = field(default_factory=list)
    stream_timer: Timer | None = None
    stream_entry: ModelActivity | None = None
    forwarded_targets: dict[str, str] = field(default_factory=dict)
    forwarded_events: dict[str, ToolEvent] = field(default_factory=dict)
    forwarded_children: dict[str, str] = field(default_factory=dict)
    active_tool_events: dict[str, ToolEvent] = field(default_factory=dict)
    detached_tool_events: dict[str, ToolEvent] = field(default_factory=dict)
    ignored_detached_jobs: set[str] = field(default_factory=set)


class DeleteToolScreen(ModalScreen[bool]):
    """Require an explicit confirmation before removing an active binding."""

    CSS = """
    DeleteToolScreen {
        align: center middle;
        background: #0008;
    }

    #delete-dialog {
        width: 58;
        height: auto;
        border: round #ef4444;
        background: #111923;
        padding: 1 2;
    }

    #delete-question {
        height: auto;
        margin-bottom: 1;
    }

    #delete-actions {
        height: 3;
        align-horizontal: right;
    }

    #delete-actions Button {
        margin-left: 1;
    }
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.tool_name = name

    def compose(self) -> ComposeResult:
        with Vertical(id="delete-dialog"):
            yield Static(
                f"Delete {self.tool_name!r} from the active toolbox?\n"
                "Its immutable versions and session history will be retained.",
                id="delete-question",
            )
            with Horizontal(id="delete-actions"):
                yield Button("Cancel", id="cancel-delete")
                yield Button("Delete tool", id="confirm-delete", variant="error")

    @on(Button.Pressed, "#cancel-delete")
    def cancel_delete(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-delete")
    def confirm_delete(self) -> None:
        self.dismiss(True)


class NamespaceNameScreen(ModalScreen[str | None]):
    """Collect a toolbox name for Blank or Save as."""

    CSS = """
    NamespaceNameScreen {
        align: center middle;
        background: #0008;
    }

    #namespace-dialog {
        width: 58;
        height: auto;
        border: round #38bdf8;
        background: #111923;
        padding: 1 2;
    }

    #namespace-question, #namespace-name {
        margin-bottom: 1;
    }

    #namespace-actions {
        height: 3;
        align-horizontal: right;
    }

    #namespace-actions Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        *,
        title: str,
        action_label: str,
        initial_name: str = "",
    ) -> None:
        super().__init__()
        self.dialog_title = title
        self.action_label = action_label
        self.initial_name = initial_name

    def compose(self) -> ComposeResult:
        with Vertical(id="namespace-dialog"):
            yield Static(self.dialog_title, id="namespace-question")
            yield Input(
                value=self.initial_name,
                placeholder="Namespace name",
                id="namespace-name",
            )
            with Horizontal(id="namespace-actions"):
                yield Button("Cancel", id="cancel-namespace")
                yield Button(
                    self.action_label,
                    id="confirm-namespace",
                    variant="primary",
                )

    def on_mount(self) -> None:
        name = self.query_one("#namespace-name", Input)
        name.focus()
        name.action_end()

    @on(Input.Submitted, "#namespace-name")
    def submit_name(self) -> None:
        self._submit()

    @on(Button.Pressed, "#cancel-namespace")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#confirm-namespace")
    def confirm(self) -> None:
        self._submit()

    def _submit(self) -> None:
        name = self.query_one("#namespace-name", Input).value.strip()
        if name:
            self.dismiss(name)


class ModelSelectionScreen(ModalScreen[str | None]):
    """Select a catalog model or enter an OpenRouter model slug directly."""

    CSS = """
    ModelSelectionScreen {
        align: center middle;
        background: #0008;
    }

    #model-dialog {
        width: 76;
        height: auto;
        max-height: 24;
        border: round #38bdf8;
        background: #111923;
        padding: 1 2;
    }

    #model-title {
        text-style: bold;
        color: #7dd3fc;
        margin-bottom: 1;
    }

    #model-name, #model-picker, #model-capability {
        margin-bottom: 1;
    }

    #model-capability {
        height: auto;
        color: #8fa5ba;
    }

    #model-actions {
        height: 3;
        align-horizontal: right;
    }

    #model-actions Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        current_model: str,
        options: list[OpenRouterModelOption],
    ) -> None:
        super().__init__()
        self.current_model = current_model
        self.options = sorted(
            options,
            key=lambda option: (option.name.casefold(), option.id.casefold()),
        )
        self.options_by_id = {option.id: option for option in options}
        self._received_initial_input_change = False

    def compose(self) -> ComposeResult:
        picker_options = self._picker_options()
        current = (
            self.current_model
            if self.current_model in self.options_by_id
            else Select.NULL
        )
        with Vertical(id="model-dialog"):
            yield Static("Change model", id="model-title")
            yield Input(
                value=self.current_model,
                placeholder="provider/model-slug",
                id="model-name",
            )
            yield Select(
                picker_options,
                value=current,
                allow_blank=True,
                prompt="Choose a tool-capable OpenRouter model",
                id="model-picker",
            )
            yield Static(id="model-capability")
            with Horizontal(id="model-actions"):
                yield Button("Cancel", id="cancel-model")
                yield Button("Use model", id="confirm-model", variant="primary")

    def on_mount(self) -> None:
        self._show_capability(self.current_model)
        model_name = self.query_one("#model-name", Input)
        model_name.focus()
        model_name.action_end()

    @on(Select.Changed, "#model-picker")
    def select_model(self, event: Select.Changed) -> None:
        if event.value is Select.NULL:
            return
        model_id = str(event.value)
        self.query_one("#model-name", Input).value = model_id
        self._show_capability(model_id)

    @on(Input.Changed, "#model-name")
    def edit_model(self, event: Input.Changed) -> None:
        model_query = event.value.strip()
        self._show_capability(model_query)
        if not self._received_initial_input_change:
            self._received_initial_input_change = True
            if model_query == self.current_model:
                return
        picker = self.query_one("#model-picker", Select)
        picker.set_options(self._picker_options(model_query))
        if model_query in self.options_by_id:
            picker.value = model_query

    @on(Input.Submitted, "#model-name")
    def submit_model(self) -> None:
        self._submit()

    @on(Button.Pressed, "#cancel-model")
    def cancel_model(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#confirm-model")
    def confirm_model(self) -> None:
        self._submit()

    def _submit(self) -> None:
        model_id = self.query_one("#model-name", Input).value.strip()
        if not model_id:
            return
        self.dismiss(model_id)

    def _show_capability(self, model_id: str) -> None:
        option = self.options_by_id.get(model_id)
        if option is None:
            message = "Custom slug; reasoning support is unknown."
        elif option.reasoning_efforts:
            message = "Reasoning efforts: " + ", ".join(option.reasoning_efforts) + "."
        else:
            message = "Does not expose configurable reasoning effort."
        self.query_one("#model-capability", Static).update(message)

    def _picker_options(self, query: str = "") -> list[tuple[str, str]]:
        normalized_query = query.casefold()
        picker_options: list[tuple[str, str]] = []
        for option in self.options:
            if (
                normalized_query
                and normalized_query not in option.name.casefold()
                and normalized_query not in option.id.casefold()
            ):
                continue
            modalities = {modality.casefold() for modality in option.input_modalities}
            badges = " ".join(
                emoji
                for modality, emoji in MODEL_INPUT_EMOJIS
                if modality in modalities
            )
            prefix = f"{badges}  " if badges else ""
            picker_options.append((f"{prefix}{option.name}  ·  {option.id}", option.id))
        return picker_options


class ReasoningEffortScreen(ModalScreen[str | None]):
    """Choose the reasoning effort sent with subsequent model requests."""

    CSS = """
    ReasoningEffortScreen {
        align: center middle;
        background: #0008;
    }

    #reasoning-dialog {
        width: 58;
        height: auto;
        border: round #a855f7;
        background: #111923;
        padding: 1 2;
    }

    #reasoning-title, #reasoning-picker, #reasoning-help {
        margin-bottom: 1;
    }

    #reasoning-title {
        text-style: bold;
        color: #c084fc;
    }

    #reasoning-help {
        height: auto;
        color: #8fa5ba;
    }

    #reasoning-actions {
        height: 3;
        align-horizontal: right;
    }

    #reasoning-actions Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        current_effort: str | None,
        efforts: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.current_effort = current_effort
        self.efforts = efforts

    def compose(self) -> ComposeResult:
        options = [("Automatic (provider default)", "automatic")]
        options.extend((effort, effort) for effort in self.efforts)
        with Vertical(id="reasoning-dialog"):
            yield Static("Reasoning effort", id="reasoning-title")
            yield Select(
                options,
                value=self.current_effort or "automatic",
                allow_blank=False,
                id="reasoning-picker",
            )
            yield Static(
                "The selected effort applies to the next model request. "
                "Automatic uses the provider default; models that allow reasoning "
                "to be disabled also offer none.",
                id="reasoning-help",
            )
            with Horizontal(id="reasoning-actions"):
                yield Button("Cancel", id="cancel-reasoning")
                yield Button("Set effort", id="confirm-reasoning", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#reasoning-picker", Select).focus()

    @on(Button.Pressed, "#cancel-reasoning")
    def cancel_reasoning(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#confirm-reasoning")
    def confirm_reasoning(self) -> None:
        value = self.query_one("#reasoning-picker", Select).value
        if value is not Select.NULL:
            self.dismiss(str(value))


class ToolManagerScreen(Screen[None]):
    """Inspect source, history, diffs, provenance, usage, and bindings."""

    TITLE = "mechagnome"
    SUB_TITLE = "tool management"
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("ctrl+d", "delete_tool", "Delete"),
    ]

    CSS = """
    ToolManagerScreen {
        background: #0b0f14;
        color: #d7e0ea;
    }

    #tool-manager {
        height: 1fr;
        padding: 1;
    }

    #namespace-toolbar, #tool-toolbar {
        height: 3;
    }

    #namespace-toolbar {
        margin-bottom: 1;
    }

    #namespace-label {
        width: 12;
        height: 3;
        content-align: left middle;
        color: #7dd3fc;
        text-style: bold;
    }

    #namespace-picker {
        width: 1fr;
        margin-right: 1;
    }

    #blank-namespace {
        width: 12;
        margin-right: 1;
    }

    #save-as-namespace {
        width: 14;
    }

    #tool-toolbar {
        margin-bottom: 1;
    }

    #tool-picker {
        width: 2fr;
        margin-right: 1;
    }

    #version-picker {
        width: 1fr;
        margin-right: 1;
    }

    #delete-tool {
        width: 16;
    }

    #tool-summary {
        height: auto;
        min-height: 4;
        border: round #34465a;
        padding: 0 1;
        margin-bottom: 1;
        background: #101821;
    }

    #tool-tabs {
        height: 1fr;
    }

    .inspection-scroll {
        height: 1fr;
        padding: 0 1;
        scrollbar-color: #516b85;
    }

    #tool-source, #tool-diff, #tool-usage {
        width: 1fr;
        height: auto;
    }

    #manager-status {
        height: 1;
        color: #8fa5ba;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        kernel: Kernel,
        *,
        session_id: str,
        on_toolbox_changed: Any,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.session_id = session_id
        self.on_toolbox_changed = on_toolbox_changed
        self.selected_namespace = self.kernel.active_toolboxes(self.session_id)[0][
            "name"
        ]
        self.inventory = self._inventory()
        self.selected_name = self.inventory[0]["name"]
        self.history = self.kernel.tool_history(
            self.selected_name, session_id=self.session_id
        )
        self.selected_version = self._initial_version()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="tool-manager"):
            with Horizontal(id="namespace-toolbar"):
                yield Static("TOOLBOX", id="namespace-label")
                yield Select(
                    self._namespace_options(),
                    value=self.selected_namespace,
                    allow_blank=False,
                    id="namespace-picker",
                )
                yield Button("Blank", id="blank-namespace")
                yield Button("Save as…", id="save-as-namespace")
            with Horizontal(id="tool-toolbar"):
                yield Select(
                    self._tool_options(),
                    value=self.selected_name,
                    allow_blank=False,
                    id="tool-picker",
                )
                yield Select(
                    self._version_options(),
                    value=self.selected_version,
                    allow_blank=False,
                    id="version-picker",
                )
                yield Button("Delete tool", id="delete-tool", variant="error")
            yield Static(id="tool-summary")
            with TabbedContent(id="tool-tabs"):
                with TabPane("Source", id="source-tab"):
                    with VerticalScroll(classes="inspection-scroll"):
                        yield Static(id="tool-source")
                with TabPane("Diff", id="diff-tab"):
                    with VerticalScroll(classes="inspection-scroll"):
                        yield Static(id="tool-diff")
                with TabPane("Usage & sessions", id="usage-tab"):
                    with VerticalScroll(classes="inspection-scroll"):
                        yield Static(id="tool-usage")
            yield Static(id="manager-status")
        yield Footer()

    def on_mount(self) -> None:
        self._render_selection()
        self.query_one("#namespace-picker", Select).focus()

    @on(Select.Changed, "#namespace-picker")
    def select_namespace(self, event: Select.Changed) -> None:
        if event.value is Select.NULL:
            return
        name = str(event.value)
        if name == self.selected_namespace:
            return
        try:
            self.kernel.select_toolboxes(self.session_id, [name], mode="use")
        except Exception as error:
            self._status(f"Toolbox change failed: {error}")
            self.query_one("#namespace-picker", Select).value = self.selected_namespace
            return
        self.selected_namespace = name
        self._refresh_after_namespace_change(f"Switched to toolbox {name}.")

    @on(Button.Pressed, "#blank-namespace")
    def blank_namespace(self) -> None:
        self.app.push_screen(
            NamespaceNameScreen(
                title="Create a blank toolbox",
                action_label="Create",
            ),
            self._finish_blank_namespace,
        )

    @on(Button.Pressed, "#save-as-namespace")
    def save_as_namespace(self) -> None:
        self.app.push_screen(
            NamespaceNameScreen(
                title=f"Rename toolbox {self.selected_namespace!r}",
                action_label="Save as",
                initial_name=self.selected_namespace,
            ),
            self._finish_save_as_namespace,
        )

    def _finish_blank_namespace(self, name: str | None) -> None:
        if name is None:
            return
        try:
            self.kernel.create_toolbox(name)
            self.kernel.select_toolboxes(self.session_id, [name], mode="use")
        except Exception as error:
            self._status(f"Blank toolbox failed: {error}")
            return
        self.selected_namespace = name
        self._refresh_after_namespace_change(f"Created blank toolbox {name}.")

    def _finish_save_as_namespace(self, name: str | None) -> None:
        if name is None:
            return
        if name == self.selected_namespace:
            self._status("Toolbox name unchanged.")
            return
        old_name = self.selected_namespace
        try:
            self.kernel.rename_toolbox(old_name, name)
        except Exception as error:
            self._status(f"Save as failed: {error}")
            return
        self.selected_namespace = name
        self._refresh_after_namespace_change(f"Renamed toolbox {old_name} to {name}.")

    @on(Select.Changed, "#tool-picker")
    def select_tool(self, event: Select.Changed) -> None:
        if event.value is Select.NULL:
            return
        name = str(event.value)
        if name not in {item["name"] for item in self.inventory}:
            return
        self.selected_name = name
        self.history = self.kernel.tool_history(
            self.selected_name, session_id=self.session_id
        )
        self.selected_version = self._initial_version()
        picker = self.query_one("#version-picker", Select)
        with self.prevent(Select.Changed):
            picker.set_options(self._version_options())
            picker.value = self.selected_version
        self._render_selection()

    @on(Select.Changed, "#version-picker")
    def select_version(self, event: Select.Changed) -> None:
        if event.value is Select.NULL:
            return
        version = int(event.value)
        if version not in {item["version"] for item in self.history["versions"]}:
            return
        self.selected_version = version
        self._render_selection()

    @on(Button.Pressed, "#delete-tool")
    def press_delete(self) -> None:
        self.action_delete_tool()

    def action_close(self) -> None:
        self.app.pop_screen()

    def action_delete_tool(self) -> None:
        if self.history["kind"] == "core":
            self._status("Core tools cannot be deleted.")
            return
        if self.history["active_version"] is None:
            self._status("This tool is already absent from the active toolbox.")
            return
        self.app.push_screen(DeleteToolScreen(self.selected_name), self._finish_delete)

    def _finish_delete(self, confirmed: bool | None) -> None:
        if not confirmed:
            return
        try:
            self.kernel.delete_tool(self.selected_name, session_id=self.session_id)
        except Exception as error:
            self._status(f"Delete failed: {error}")
            return
        self.on_toolbox_changed()
        self.inventory = self._inventory()
        self.history = self.kernel.tool_history(
            self.selected_name, session_id=self.session_id
        )
        self.selected_version = self._initial_version()
        tool_picker = self.query_one("#tool-picker", Select)
        version_picker = self.query_one("#version-picker", Select)
        with self.prevent(Select.Changed):
            tool_picker.set_options(self._tool_options())
            tool_picker.value = self.selected_name
            version_picker.set_options(self._version_options())
            version_picker.value = self.selected_version
        self._render_selection()
        self._status(f"Deleted {self.selected_name} from the active toolbox.")

    def _inventory(self) -> list[dict[str, Any]]:
        return sorted(
            self.kernel.tool_inventory(session_id=self.session_id),
            key=lambda item: (item["kind"] == "core", item["name"]),
        )

    def _namespace_options(self) -> list[tuple[str, str]]:
        return [
            (
                f"{toolbox['name']}  ·  cwd default"
                if toolbox["default"]
                else toolbox["name"],
                toolbox["name"],
            )
            for toolbox in self.kernel.list_toolboxes()
        ]

    def _refresh_after_namespace_change(self, status: str) -> None:
        self.on_toolbox_changed()
        self.inventory = self._inventory()
        available = {item["name"] for item in self.inventory}
        if self.selected_name not in available:
            self.selected_name = self.inventory[0]["name"]
        self.history = self.kernel.tool_history(
            self.selected_name, session_id=self.session_id
        )
        self.selected_version = self._initial_version()

        namespace_picker = self.query_one("#namespace-picker", Select)
        tool_picker = self.query_one("#tool-picker", Select)
        version_picker = self.query_one("#version-picker", Select)
        with self.prevent(Select.Changed):
            namespace_picker.set_options(self._namespace_options())
            namespace_picker.value = self.selected_namespace
            tool_picker.set_options(self._tool_options())
            tool_picker.value = self.selected_name
            version_picker.set_options(self._version_options())
            version_picker.value = self.selected_version
        self._render_selection()
        self._status(status)

    def _tool_options(self) -> list[tuple[str, str]]:
        options = []
        for item in self.inventory:
            state = (
                f"v{item['active_version']}"
                if item["active_version"] is not None
                else "deleted"
            )
            options.append(
                (
                    f"{item['name']}  ·  {state}  ·  {item['call_count']} calls",
                    item["name"],
                )
            )
        return options

    def _initial_version(self) -> int:
        return int(
            self.history["active_version"] or self.history["versions"][0]["version"]
        )

    def _version_options(self) -> list[tuple[str, int]]:
        options = []
        for version in self.history["versions"]:
            active = " · active" if version["active"] else ""
            options.append(
                (
                    f"v{version['version']}{active} · {version['call_count']} calls",
                    int(version["version"]),
                )
            )
        return options

    def _render_selection(self) -> None:
        version = self._selected_version_record()
        active = self.history["active_version"]
        state = f"active v{active}" if active is not None else "deleted"
        creator = version["created_session_id"] or "host / unknown"
        summary = Text()
        summary.append(self.selected_name, style="bold cyan")
        summary.append(
            f"  ·  {self.history['kind']}  ·  {self.history['toolbox']}  ·  {state}\n"
        )
        summary.append(
            "namespaces: " + ", ".join(self.history["namespaces"]) + "\n",
            style="dim",
        )
        summary.append(str(version["description"]))
        summary.append(
            f"\nv{version['version']} created {version['created_at'][:19]}  ·  "
            f"session {creator[:16]}"
        )
        self.query_one("#tool-summary", Static).update(summary)
        self.query_one("#tool-source", Static).update(
            Syntax(
                version["source"],
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=False,
            )
        )
        self.query_one("#tool-diff", Static).update(self._source_diff(version))
        self.query_one("#tool-usage", Static).update(self._usage_table(version))
        delete = self.query_one("#delete-tool", Button)
        delete.disabled = self.history["kind"] == "core" or active is None
        self._status(
            f"Viewing {self.selected_name} v{version['version']}  ·  "
            "select another tool or version above"
        )

    def _selected_version_record(self) -> dict[str, Any]:
        return next(
            version
            for version in self.history["versions"]
            if version["version"] == self.selected_version
        )

    def _source_diff(self, version: dict[str, Any]) -> Any:
        earlier = next(
            (
                item
                for item in self.history["versions"]
                if item["version"] == version["version"] - 1
            ),
            None,
        )
        if earlier is None:
            return Text(
                f"v{version['version']} is the first recorded version.", style="dim"
            )
        lines = difflib.unified_diff(
            earlier["source"].splitlines(),
            version["source"].splitlines(),
            fromfile=f"{self.selected_name}@v{earlier['version']}",
            tofile=f"{self.selected_name}@v{version['version']}",
            lineterm="",
        )
        diff_lines = list(lines)
        diff = "\n".join(diff_lines) + "\n" if diff_lines else "(source unchanged)\n"
        return Syntax(diff, "diff", theme="monokai", word_wrap=False)

    def _usage_table(self, version: dict[str, Any]) -> Table:
        table = Table(box=None, expand=True, show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Selected version")
        table.add_column("All versions")
        table.add_row("Version ID", str(version["tool_version_id"]), "—")
        table.add_row(
            "Calls", str(version["call_count"]), str(self.history["call_count"])
        )
        table.add_row("Timed calls", str(version["timed_call_count"]), "—")
        table.add_row(
            "Average duration",
            _format_duration(version["average_duration_ms"]) or "unavailable",
            "—",
        )
        table.add_row(
            "Succeeded",
            str(version["success_count"]),
            str(self.history["success_count"]),
        )
        table.add_row(
            "Failed",
            str(version["failure_count"]),
            str(self.history["failure_count"]),
        )
        table.add_row(
            "Sessions",
            str(version["session_count"]),
            str(len(self.history["sessions"])),
        )
        table.add_row(
            "Last called",
            str(version["last_called_at"] or "never")[:19],
            "",
        )
        table.add_section()
        table.add_row("Calling session", "Calls", "Last called")
        if not self.history["sessions"]:
            table.add_row("No calls recorded", "", "")
        for session in self.history["sessions"]:
            table.add_row(
                str(session["session_id"])[:16],
                str(session["call_count"]),
                str(session["last_called_at"])[:19],
            )
        return table

    def _status(self, message: str) -> None:
        self.query_one("#manager-status", Static).update(message)


class ToolboxApp(App[None]):
    """Chat with an agent while watching its toolbox grow."""

    TITLE = "mechagnome"
    SUB_TITLE = "persistent metaprogrammable agent"
    BINDINGS = [
        Binding("escape", "stop_rollout", "Stop", priority=True),
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+n", "new_session", "New session"),
        Binding("tab", "next_session", "Next session", show=False, priority=True),
        Binding("up", "previous_prompt", "Previous prompt", show=False),
        Binding("down", "next_prompt", "Next prompt", show=False),
        ("ctrl+t", "manage_tools", "Manage tools"),
        ("f1", "show_help", "Help"),
    ]

    CSS = """
    Screen {
        background: #0b0f14;
        color: #d7e0ea;
    }

    Header {
        background: #111923;
        color: #d7e0ea;
    }

    #workspace {
        height: 1fr;
    }

    #session-tabs {
        width: 1fr;
        height: 1fr;
    }

    #session-tabs > ContentTabs {
        background: #111923;
    }

    #session-tabs TabPane {
        height: 1fr;
        padding: 0;
    }

    .session-chat {
        width: 1fr;
        height: 1fr;
        border: round #34465a;
        padding: 0 1;
        background: #0d131b;
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-color: #516b85;
    }

    .chat-entry {
        width: 1fr;
        height: auto;
        margin-bottom: 1;
    }

    .tool-response-image {
        width: auto;
        max-width: 100%;
        height: auto;
        max-height: 24;
    }

    .streaming-response {
        height: 1;
        min-height: 1;
        margin-bottom: 0;
        text-wrap: nowrap;
        text-overflow: clip;
    }

    .tool-event {
        width: 1fr;
        height: auto;
        margin-bottom: 0;
        padding: 0;
        border-top: none;
        background: transparent;
    }

    .tool-event:focus-within {
        background-tint: #d7e0ea 3%;
    }

    .tool-event CollapsibleTitle {
        height: 1;
        padding: 0 1;
        color: #7890a6;
        text-style: italic;
    }

    .tool-call CollapsibleTitle {
        color: #7dd3fc;
    }

    .tool-response CollapsibleTitle {
        color: #789b83;
    }

    .tool-error CollapsibleTitle {
        color: #b98989;
    }

    .tool-event Contents {
        padding: 0 0 0 2;
    }

    .tool-event-detail {
        width: 1fr;
        height: auto;
        padding: 0 1;
        border: round #34465a;
        background: #101821;
        color: #a8b8c8;
    }

    #sidebar {
        width: 42;
        min-width: 36;
        border: round #34465a;
        background: #101821;
    }

    .sidebar-title {
        height: 1;
        padding: 0 1;
        color: #7dd3fc;
        text-style: bold;
    }

    #model-info {
        height: 5;
        padding: 0 1;
        color: #a8b8c8;
    }

    #tools {
        height: 1fr;
        padding: 1 2;
        background: transparent;
        color: #d7e0ea;
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-color: #516b85;
        scrollbar-background: transparent;
    }

    #status {
        height: 1;
        padding: 0 1;
        background: #111923;
        color: #8fa5ba;
    }

    #status-message, #status-context, #status-session, .status-separator {
        width: auto;
        height: 1;
    }

    .status-separator {
        margin: 0 1;
    }

    .status-control {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0;
        border: none;
        background: transparent;
        color: #7dd3fc;
        text-style: bold;
    }

    .status-control:hover, .status-control:focus {
        background: #243447;
        color: #bae6fd;
        text-style: bold underline;
    }

    #reasoning-selector {
        color: #c084fc;
    }

    #prompt {
        height: 3;
        border: round #516b85;
        background: #0d131b;
    }

    #prompt:focus {
        border: round #38bdf8;
    }

    Footer {
        background: #111923;
    }
    """

    def __init__(
        self,
        kernel: Kernel,
        model: Model | ModelProvider,
        *,
        model_name: str,
        model_provider: CompletionTransport | None = None,
    ) -> None:
        super().__init__()
        self._accept_background_events = True
        self._detached_event_lock = Lock()
        self._staged_detached_events: dict[
            tuple[str, str], tuple[SessionTab, AgentEvent]
        ] = {}
        self._detached_event_timer: Timer | None = None
        self.kernel = kernel
        self.model = model.transport if isinstance(model, ModelProvider) else model
        self.model_name = model_name
        self.harness = Harness(kernel)
        initial_conversation = self.harness.start(
            model,
            model_provider=model_provider,
        )
        self.model_provider = initial_conversation.model_session.provider
        self._tab_counter = 0
        initial = self._make_session_tab(conversation=initial_conversation)
        self.session_tabs = [initial]
        self._active_pane_id = initial.pane_id
        self.model_options: list[OpenRouterModelOption] = []

    def _make_session_tab(
        self, *, conversation: Conversation | None = None
    ) -> SessionTab:
        self._tab_counter += 1
        number = self._tab_counter
        return SessionTab(
            conversation=conversation or self.harness.start(self.model_provider),
            pane_id=f"session-{number}",
            label=f"Session {number}",
            chat=ChatFeed(id=f"chat-{number}", classes="session-chat"),
        )

    @property
    def active_session(self) -> SessionTab:
        return next(
            state
            for state in self.session_tabs
            if state.pane_id == self._active_pane_id
        )

    @property
    def conversation(self) -> Conversation:
        """The active conversation, retained as the synchronous command surface."""
        return self.active_session.conversation

    @property
    def chat(self) -> ChatFeed:
        return self.active_session.chat

    @property
    def busy(self) -> bool:
        return any(state.running for state in self.session_tabs)

    @property
    def streamed_text(self) -> str:
        return self.active_session.streamed_text

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            initial = self.active_session
            with TabbedContent(id="session-tabs", initial=initial.pane_id):
                with TabPane(initial.label, id=initial.pane_id):
                    yield initial.chat
            with Vertical(id="sidebar"):
                yield Static("MODEL", classes="sidebar-title")
                yield Static(id="model-info")
                yield Static("TOOLBOX", classes="sidebar-title")
                yield RichLog(id="tools", wrap=True, markup=False, min_width=1)
        with Horizontal(id="status"):
            yield Static(id="status-message")
            yield Static("·", classes="status-separator")
            yield Button(
                self._active_model_name,
                id="model-selector",
                classes="status-control",
            )
            yield Static("·", id="reasoning-separator", classes="status-separator")
            yield Button(
                "reasoning: automatic",
                id="reasoning-selector",
                classes="status-control",
            )
            yield Static("·", id="context-separator", classes="status-separator")
            yield Static(id="status-context")
            yield Static("·", classes="status-separator")
            yield Static(id="status-session")
        yield Input(
            placeholder="Ask the agent to build or use whatever it needs…",
            id="prompt",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Populate the initial panes and focus the prompt."""
        self._detached_event_timer = self.set_interval(
            0.05, self._flush_staged_detached_events
        )
        self._show_welcome()
        self._refresh_sidebar()
        self._refresh_model_controls()
        self._refresh_active_controls()
        self.query_one("#prompt", Input).focus()
        self.load_model_options()

    def on_unmount(self, event: Unmount) -> None:
        """Release a synchronous rollout before asyncio joins worker threads."""
        with self._detached_event_lock:
            self._accept_background_events = False
            self._staged_detached_events.clear()
        if self._detached_event_timer is not None:
            self._detached_event_timer.stop()
            self._detached_event_timer = None
        for state in self.session_tabs:
            self._reset_stream_state(state)
            state.conversation.close()
        self.harness.close()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Leave TAB available for focus traversal on pushed screens."""
        if action == "next_session" and len(self.screen_stack) != 1:
            return False
        if action in {"previous_prompt", "next_prompt"}:
            return len(self.screen_stack) == 1 and self.focused is self.query_one(
                "#prompt", Input
            )
        return super().check_action(action, parameters)

    @on(TabbedContent.TabActivated, "#session-tabs")
    def activate_session_tab(self, event: TabbedContent.TabActivated) -> None:
        """Restore session-local controls when a tab is clicked or selected."""
        pane_id = event.pane.id
        if pane_id is None or not any(
            state.pane_id == pane_id for state in self.session_tabs
        ):
            return
        if self.is_mounted and pane_id != self._active_pane_id:
            self.active_session.draft = self.query_one("#prompt", Input).value
        self._active_pane_id = pane_id
        if self.is_mounted:
            self._refresh_active_controls()

    @on(Input.Changed, "#prompt")
    def save_session_draft(self, event: Input.Changed) -> None:
        """Keep unsent input attached to the active session."""
        self.active_session.draft = event.value

    @work(thread=True, exclusive=True, group="model-catalog", exit_on_error=False)
    def load_model_options(self) -> None:
        """Load OpenRouter model capabilities without blocking the TUI."""
        if not isinstance(self.model, OpenRouterModel):
            return
        try:
            options = self.model.available_models()
        except OpenRouterError:
            return
        self.call_from_thread(self._set_model_options, options)

    def _set_model_options(self, options: list[OpenRouterModelOption]) -> None:
        self.model_options = options
        current = self._current_model_option()
        effort = self.model.reasoning_effort
        if current is None or (
            effort is not None and effort not in current.reasoning_efforts
        ):
            self.model.reasoning_effort = None
        self._refresh_model_controls()
        self._refresh_active_status()

    @on(Button.Pressed, "#model-selector")
    def choose_model(self) -> None:
        """Open the model picker when no rollout is active."""
        if self.busy:
            self._set_status("stop the active rollout before changing models")
            return
        if not isinstance(self.model, OpenRouterModel):
            self._set_status("this model adapter cannot be changed at runtime")
            return
        self.push_screen(
            ModelSelectionScreen(self._active_model_name, self.model_options),
            self._finish_model_selection,
        )

    def _finish_model_selection(self, selection: str | None) -> None:
        if selection is None:
            return
        self.model.model = selection
        current = self._current_model_option()
        effort = self.model.reasoning_effort
        if current is None or (
            effort is not None and effort not in current.reasoning_efforts
        ):
            self.model.reasoning_effort = None
        self._refresh_sidebar()
        self._refresh_model_controls()
        self._set_status(f"model → {selection}")

    @on(Button.Pressed, "#reasoning-selector")
    def choose_reasoning_effort(self) -> None:
        """Open the reasoning-effort picker when the model supports it."""
        if self.busy:
            self._set_status("stop the active rollout before changing reasoning")
            return
        if not isinstance(self.model, OpenRouterModel):
            return
        current = self._current_model_option()
        if current is None or not current.reasoning_efforts:
            return
        self.push_screen(
            ReasoningEffortScreen(
                self.model.reasoning_effort,
                current.reasoning_efforts,
            ),
            self._finish_reasoning_effort,
        )

    def _finish_reasoning_effort(self, selection: str | None) -> None:
        if selection is None or not isinstance(self.model, OpenRouterModel):
            return
        self.model.reasoning_effort = None if selection == "automatic" else selection
        self._refresh_model_controls()
        self._set_status(f"reasoning → {selection}")

    @on(Input.Submitted, "#prompt")
    async def submit_prompt(self, event: Input.Submitted) -> None:
        """Handle a slash command or send one message to the agent worker."""
        prompt = event.value.strip()
        state = self.active_session
        if not prompt or state.running:
            return
        state.user_history.append(prompt)
        state.history_index = None
        state.history_draft = ""
        state.draft = ""
        with self.prevent(Input.Changed):
            event.input.value = ""
        if prompt.startswith("/") and await self._command(prompt):
            return
        self._write_user(prompt, state)
        self._start_rollout(state)
        self.run_agent(prompt, state)

    @work(thread=True, group="agent", exit_on_error=False)
    def run_agent(self, prompt: str, state: SessionTab) -> None:
        """Run the synchronous model/tool loop without blocking the terminal UI."""
        error_reported = False

        def relay(event: AgentEvent) -> None:
            nonlocal error_reported
            if event.kind.startswith("detached_"):
                self._stage_detached_event(state, event)
                return
            if event.kind == "model_delta":
                self.call_from_thread(
                    self._queue_stream_delta,
                    state,
                    str(event.payload.get("text") or ""),
                )
                return
            error_reported = error_reported or event.kind in {
                "model_failed",
                "harness_failed",
            }
            self.call_from_thread(self._display_event, state, event)

        try:
            state.conversation.send(prompt, on_event=relay)
        except RunCancelled:
            pass
        except (
            Exception
        ) as error:  # Provider and generated tool errors are user-visible.
            if not error_reported:
                self.call_from_thread(self._write_error, str(error), state)
        finally:
            self.call_from_thread(self._finish_rollout, state)

    def _display_event(self, state: SessionTab, event: AgentEvent) -> None:
        if event.kind == "user":
            return
        if event.kind == "model_started":
            self._show_model_activity(state)
            self._set_status("thinking…", state)
        elif event.kind == "model_delta":
            text = str(event.payload.get("text") or "")
            state.streamed_text += text
            self._render_stream(state, text)
            self._set_status("streaming…", state)
        elif event.kind == "model":
            total_tokens = event.payload.get("total_tokens")
            if (
                isinstance(total_tokens, bool)
                or not isinstance(total_tokens, int)
                or total_tokens <= 0
            ):
                state.total_tokens = None
                state.context_model = None
            else:
                state.total_tokens = total_tokens
                state.context_model = self._active_model_name
            content = str(event.payload.get("text") or "")
            if content:
                self._finish_stream(state, content)
            else:
                self._clear_stream(state)
            self._set_status(
                "planning" if event.payload.get("calls") else "answering", state
            )
        elif event.kind == "call_started":
            name = str(event.tool_name or "tool")
            args = event.payload.get("args")
            if self._forwarded_child(state, event, name, args):
                return
            target: str | None = None
            if name == "call_tool" and isinstance(args, dict):
                requested = args.get("name")
                if isinstance(requested, str):
                    target = requested
            displayed = state.chat.write_tool(
                "call",
                name,
                self._compact(args),
                self._argument_summary(args),
            )
            if target and event.call_id:
                state.forwarded_targets[event.call_id] = target
                state.forwarded_events[event.call_id] = displayed
            if event.call_id:
                state.active_tool_events[event.call_id] = displayed
            self._set_status(f"calling {name}", state)
        elif event.kind == "call_succeeded":
            display = self._tool_response(state, event)
            if display is None:
                return
            name, detail, outcome_summary = display
            active = state.active_tool_events.pop(event.call_id, None)
            if active is None:
                state.chat.write_tool(
                    "response", name, detail, outcome_summary=outcome_summary
                )
            else:
                active.finish("response", name, detail, outcome_summary)
            for image in _tool_images(event.payload.get("result")):
                state.chat.write_image(image)
            if state is self.active_session:
                self._refresh_sidebar()
        elif event.kind == "binding_changed":
            name = str(event.payload.get("name") or event.tool_name or "tool")
            version = event.payload.get("to_version")
            state.chat.write(
                Panel(
                    Text(f"active version → v{version}"),
                    title=f"toolbox · {name}",
                    title_align="left",
                    border_style="blue",
                )
            )
            if state is self.active_session:
                self._refresh_sidebar()
        elif event.kind == "call_failed":
            display = self._tool_response(state, event, failed=True)
            if display is None:
                return
            name, detail, outcome_summary = display
            active = state.active_tool_events.pop(event.call_id, None)
            if active is None:
                state.chat.write_tool(
                    "error", name, detail, outcome_summary=outcome_summary
                )
            else:
                active.finish("error", name, detail, outcome_summary)
            self._set_status(f"{name} failed", state)
        elif event.kind in {"model_failed", "harness_failed"}:
            self._stop_active_tool_events(state)
            self._clear_stream(state)
            self._write_error(str(event.payload.get("message") or event.payload), state)
        elif event.kind == "cancelled":
            self._stop_active_tool_events(state)
            partial = state.streamed_text + "".join(state.pending_stream_text)
            content = partial or str(event.payload.get("message") or "Rollout stopped.")
            self._finish_stream(state, content, stopped=True)
            self._set_status("stopped", state)

    def _display_detached_event(self, state: SessionTab, event: AgentEvent) -> None:
        job_id = event.payload.get("job_id")
        if not isinstance(job_id, str):
            return
        terminal = event.payload.get("status") in {"succeeded", "failed"}
        if job_id in state.ignored_detached_jobs:
            if terminal:
                state.ignored_detached_jobs.discard(job_id)
            return
        displayed = state.detached_tool_events.get(job_id)
        if displayed is None:
            name = str(event.payload.get("name") or "tool")
            args = event.payload.get("args")
            displayed = state.chat.write_tool(
                "call",
                name,
                self._compact(args),
                self._argument_summary(args),
            )
            displayed.attach_detached(job_id)
            state.detached_tool_events[job_id] = displayed
        if terminal:
            displayed.finish_detached(event.payload)
            state.detached_tool_events.pop(job_id, None)
            if event.payload.get("status") == "succeeded":
                for image in _tool_images(event.payload.get("result")):
                    state.chat.write_image(image)
        else:
            displayed.set_detached_output(
                str(event.payload.get("output_tail") or ""),
                truncated=bool(event.payload.get("truncated")),
            )

    def _stage_detached_event(self, state: SessionTab, event: AgentEvent) -> None:
        """Coalesce worker-thread updates without waiting for the UI loop."""
        job_id = event.payload.get("job_id")
        if not isinstance(job_id, str):
            return
        with self._detached_event_lock:
            if self._accept_background_events:
                self._staged_detached_events[(state.pane_id, job_id)] = (state, event)

    def _flush_staged_detached_events(self) -> None:
        with self._detached_event_lock:
            staged = list(self._staged_detached_events.values())
            self._staged_detached_events.clear()
        for state, event in staged:
            self._display_detached_event(state, event)

    def _queue_stream_delta(self, state: SessionTab, text: str) -> None:
        if not text:
            return
        state.pending_stream_text.append(text)
        if state.stream_timer is None:
            state.stream_timer = self.set_timer(
                0.05, lambda: self._flush_stream_delta(state)
            )

    def _flush_stream_delta(self, state: SessionTab) -> None:
        state.stream_timer = None
        if not state.pending_stream_text:
            return
        text = "".join(state.pending_stream_text)
        state.streamed_text += text
        state.pending_stream_text.clear()
        self._render_stream(state, text)
        self._set_status("streaming…", state)

    def _show_model_activity(self, state: SessionTab) -> ModelActivity:
        if state.stream_entry is None:
            state.stream_entry = state.chat.write_activity()
        return state.stream_entry

    def _render_stream(self, state: SessionTab, text: str) -> None:
        self._show_model_activity(state).append_text(text)
        state.chat.call_after_refresh(state.chat.scroll_end, animate=False)

    def _finish_stream(
        self, state: SessionTab, content: str, *, stopped: bool = False
    ) -> None:
        self._reset_stream_state(state)
        title = self._active_model_name
        border_style = "bright_magenta"
        if stopped:
            title += " · stopped"
            border_style = "yellow"
        panel = self._model_panel(content, title=title, border_style=border_style)
        if state.stream_entry is None:
            state.chat.write(panel)
        else:
            state.stream_entry.finish(panel)
            state.chat.call_after_refresh(state.chat.scroll_end, animate=False)
            state.stream_entry = None

    def _clear_stream(self, state: SessionTab) -> None:
        self._reset_stream_state(state)
        if state.stream_entry is not None:
            state.stream_entry.remove()
            state.stream_entry = None

    def _reset_stream_state(self, state: SessionTab) -> None:
        if state.stream_timer is not None:
            state.stream_timer.stop()
            state.stream_timer = None
        if state.stream_entry is not None:
            state.stream_entry.stop_animation()
        state.pending_stream_text.clear()
        state.streamed_text = ""

    def _model_panel(
        self,
        content: str,
        *,
        title: str | None = None,
        border_style: str = "bright_magenta",
    ) -> Panel:
        return Panel(
            Markdown(content),
            title=title or self._active_model_name,
            title_align="left",
            border_style=border_style,
        )

    async def _command(self, prompt: str) -> bool:
        try:
            parts = shlex.split(prompt)
        except ValueError as error:
            self._write_error(str(error))
            return True
        command = parts[0]
        arguments = parts[1:]
        if command in {"/quit", "/q"}:
            self.action_quit()
        elif command == "/new":
            await self.action_new_session()
        elif command in {"/clear", "/reset"}:
            await self.action_clear_session()
        elif command == "/end":
            await self.action_end_session()
        elif command == "/tools" or (command == "/toolbox" and not arguments):
            self.action_manage_tools()
        elif command == "/toolbox":
            self._toolbox_command(arguments)
        elif command == "/sessions":
            self._show_sessions()
        elif command in {"/help", "/?"}:
            self.action_show_help()
        else:
            self._write_error(f"unknown command: {command}")
        return True

    def action_quit(self) -> None:
        """Cancel synchronous work before asking Textual to exit."""
        for state in self.session_tabs:
            state.conversation.close()
        self.exit()

    def _toolbox_command(self, arguments: list[str]) -> None:
        if self.busy:
            self._set_status("stop the active rollout before changing toolboxes")
            return
        action, *values = arguments
        try:
            if action == "list":
                self._show_toolboxes()
                return
            if action == "create":
                if not 1 <= len(values) <= 2:
                    raise ValueError("usage: /toolbox create NAME [CWD]")
                cwd = values[1] if len(values) == 2 else self.kernel.cwd
                self.kernel.create_toolbox(values[0], cwd=cwd)
                self._set_status(f"created toolbox {values[0]}")
            elif action in {"use", "add", "remove"}:
                if not values:
                    raise ValueError(f"usage: /toolbox {action} NAME [NAME ...]")
                self.kernel.select_toolboxes(
                    self.conversation.session_id, values, mode=action
                )
                self._set_status(f"toolbox selection {action}d")
            elif action == "default":
                if values:
                    raise ValueError("usage: /toolbox default")
                self.kernel.reset_toolboxes(self.conversation.session_id)
                self._set_status("restored cwd-default toolbox")
            elif action == "set-default":
                if not 1 <= len(values) <= 2:
                    raise ValueError("usage: /toolbox set-default NAME [CWD]")
                self.kernel.set_cwd_default(
                    values[0], cwd=values[1] if len(values) == 2 else None
                )
                self._set_status(f"cwd default → {values[0]}")
            else:
                raise ValueError(f"unknown toolbox command: {action}")
        except Exception as error:
            self._write_error(str(error))
            return
        self._refresh_sidebar()

    def _show_toolboxes(self) -> None:
        selected = {
            item["id"]: item
            for item in self.kernel.active_toolboxes(self.conversation.session_id)
        }
        table = Table("Toolbox", "Active", "Cwd", box=None, expand=True)
        for toolbox in self.kernel.list_toolboxes():
            active = selected.get(toolbox["id"])
            marker = (
                "primary" if active and active["primary"] else "yes" if active else ""
            )
            table.add_row(toolbox["name"], marker, str(toolbox["cwd"] or "—"))
        self.chat.write(Panel(table, title="toolboxes", border_style="blue"))

    async def action_new_session(self) -> None:
        """Open and activate a fresh model conversation in a new tab."""
        state = self._make_session_tab()
        self.session_tabs.append(state)
        tabs = self.query_one("#session-tabs", TabbedContent)
        await tabs.add_pane(
            TabPane(state.label, state.chat, id=state.pane_id),
            after=self.session_tabs[-2].pane_id,
        )
        tabs.active = state.pane_id
        self._show_welcome(state)
        self._set_status("new session", state)

    def action_previous_prompt(self) -> None:
        """Replace the active draft with the previous submitted prompt."""
        state = self.active_session
        if not state.user_history:
            return
        prompt = self.query_one("#prompt", Input)
        if state.history_index is None:
            state.history_draft = prompt.value
            state.history_index = len(state.user_history) - 1
        elif state.history_index > 0:
            state.history_index -= 1
        self._show_history_prompt(state)

    def action_next_prompt(self) -> None:
        """Replace the active draft with the next submitted prompt."""
        state = self.active_session
        if state.history_index is None:
            return
        if state.history_index < len(state.user_history) - 1:
            state.history_index += 1
            self._show_history_prompt(state)
            return
        state.history_index = None
        self._set_prompt_value(state, state.history_draft)
        state.history_draft = ""

    def _show_history_prompt(self, state: SessionTab) -> None:
        """Display the history entry selected for a session."""
        assert state.history_index is not None
        self._set_prompt_value(state, state.user_history[state.history_index])

    def _set_prompt_value(self, state: SessionTab, value: str) -> None:
        """Update the prompt without treating history navigation as an edit."""
        prompt = self.query_one("#prompt", Input)
        state.draft = value
        with self.prevent(Input.Changed):
            prompt.value = value
            prompt.cursor_position = len(value)

    def action_next_session(self) -> None:
        """Activate the next open session tab, wrapping at the end."""
        if len(self.session_tabs) < 2:
            return
        index = self.session_tabs.index(self.active_session)
        target = self.session_tabs[(index + 1) % len(self.session_tabs)]
        self.query_one("#session-tabs", TabbedContent).active = target.pane_id

    async def action_clear_session(self) -> None:
        """Reset the active tab with a new durable session."""
        state = self.active_session
        if state.running:
            self._set_status("stop the active rollout before clearing a session")
            return
        state.conversation.close()
        self._reset_session_ui(state)
        state.conversation = self.harness.start(self.model_provider)
        state.chat.clear()
        with self.prevent(Input.Changed):
            self.query_one("#prompt", Input).value = ""
        self._show_welcome(state)
        self._refresh_sidebar()
        self._set_status("session cleared", state)

    async def action_end_session(self) -> None:
        """Close the active tab, exiting when it is the final session."""
        state = self.active_session
        if state.running:
            self._set_status("stop the active rollout before ending a session")
            return
        state.conversation.close()
        self._reset_session_ui(state)
        if len(self.session_tabs) == 1:
            self.exit()
            return
        index = self.session_tabs.index(state)
        successor = (
            self.session_tabs[index + 1]
            if index + 1 < len(self.session_tabs)
            else self.session_tabs[index - 1]
        )
        self._active_pane_id = successor.pane_id
        self.session_tabs.remove(state)
        await self.query_one("#session-tabs", TabbedContent).remove_pane(state.pane_id)
        self.query_one("#session-tabs", TabbedContent).active = successor.pane_id

    def action_stop_rollout(self) -> None:
        """Stop the active tab's model stream or tool subprocess."""
        state = self.active_session
        if state.running:
            if state.conversation.cancel():
                self._set_status("stopping…", state)
                self._refresh_active_status()
            return
        if isinstance(self.screen, DeleteToolScreen):
            self.screen.dismiss(False)
            return
        if isinstance(self.screen, ToolManagerScreen):
            self.pop_screen()

    def action_show_tools(self) -> None:
        """Refresh the toolbox and report its active contents in chat."""
        self._refresh_sidebar()
        bindings = self.kernel.bindings(
            session_id=self.conversation.session_id, include_origin=True
        )
        table = Table("Tool", "Version", "Kind", "Toolbox", box=None, expand=True)
        for binding in bindings:
            table.add_row(
                binding["name"],
                str(binding["active_version"]),
                binding["kind"],
                binding["toolbox"],
            )
        self.chat.write(Panel(table, title="active toolbox", border_style="blue"))

    def action_manage_tools(self) -> None:
        """Toggle the source, history, usage, and deletion manager."""
        if isinstance(self.screen, ToolManagerScreen):
            self.pop_screen()
            return
        self.push_screen(
            ToolManagerScreen(
                self.kernel,
                session_id=self.conversation.session_id,
                on_toolbox_changed=self._refresh_sidebar,
            )
        )

    def action_show_help(self) -> None:
        """Show local TUI commands."""
        self.chat.write(
            Panel(
                Markdown(
                    "**Commands**\n\n"
                    "- `Esc` — stop the active tab's rollout\n"
                    "- `Ctrl+N` or `/new` — open a new session tab\n"
                    "- `TAB` or click — switch session tabs\n"
                    "- `/clear` — reset the active tab with a fresh session\n"
                    "- `/end` — close the active session tab\n"
                    "- `/tools` or `Ctrl+T` — toggle tool management\n"
                    "- `/toolbox list` — list toolboxes and active order\n"
                    "- `/toolbox create NAME [CWD]` — create a toolbox\n"
                    "- `/toolbox use|add|remove NAME...` — change this session\n"
                    "- `/toolbox default` — restore the session cwd default\n"
                    "- `/sessions` — list saved sessions\n"
                    "- `/quit` — exit\n\n"
                    "The agent itself can call `help` for tool-authoring docs."
                ),
                title="help",
                border_style="blue",
            )
        )

    def _show_sessions(self) -> None:
        sessions = self.kernel.list_sessions(limit=15)["sessions"]
        table = Table(
            "Session", "Kind", "Parent", "Events", "Created", box=None, expand=True
        )
        for session in sessions:
            table.add_row(
                session["id"],
                session["kind"],
                (session["parent_session_id"] or "—")[:12],
                str(session["event_count"]),
                session["created_at"][:19],
            )
        self.chat.write(Panel(table, title="saved sessions", border_style="blue"))

    def _show_welcome(self, state: SessionTab | None = None) -> None:
        state = state or self.active_session
        readiness = "ready" if getattr(self.model, "ready", True) else "API key missing"
        state.chat.write(
            Panel(
                Markdown(
                    f"**{self._active_model_name}** · {readiness}\n\n"
                    "Ask for a task. The agent begins with five editable toolbox "
                    "operations plus run_agent and grows the toolbox as needed. "
                    "Use `/help` for TUI commands."
                ),
                title="mechagnome",
                border_style="bright_blue",
            )
        )
        if isinstance(self.model, OpenRouterModel) and not self.model.ready:
            self._write_error(
                f"Set {self.model.api_key_env} in your environment, then restart.",
                state,
            )

    def _write_user(self, prompt: str, state: SessionTab | None = None) -> None:
        (state or self.active_session).chat.write(
            Panel(
                Markdown(prompt),
                title="you",
                title_align="left",
                border_style="yellow",
            )
        )

    def _write_error(self, message: str, state: SessionTab | None = None) -> None:
        state = state or self.active_session
        state.chat.write(Panel(Text(message), title="error", border_style="red"))
        self._set_status("error", state)

    def _refresh_sidebar(self) -> None:
        bindings = self.kernel.bindings(
            recent_first=True,
            session_id=self.conversation.session_id,
            include_origin=True,
        )
        selected = self.kernel.active_toolboxes(self.conversation.session_id)
        core_count = sum(binding["kind"] == "core" for binding in bindings)
        user_count = len(bindings) - core_count
        selection = " + ".join(item["name"] for item in selected)
        self.query_one("#model-info", Static).update(
            f"{self._active_model_name}\n"
            f"session {self.conversation.session_id[:10]}\n"
            f"{selection}\n"
            f"{core_count} core · {user_count} user"
        )
        toolbox = self.query_one("#tools", RichLog)
        toolbox.clear()
        for index, binding in enumerate(bindings):
            style = "cyan" if binding["kind"] == "core" else "green"
            toolbox.write(
                Text(
                    f"{binding['name']}  v{binding['active_version']}  "
                    f"[{binding['toolbox']}]",
                    style=style,
                    overflow="fold",
                    no_wrap=False,
                )
            )
            toolbox.write(
                Text(
                    binding["description"],
                    style="dim",
                    overflow="fold",
                    no_wrap=False,
                )
            )
            if index < len(bindings) - 1:
                toolbox.write("")

    def _refresh_model_controls(self) -> None:
        selector = self.query_one("#model-selector", Button)
        reasoning = self.query_one("#reasoning-selector", Button)
        separator = self.query_one("#reasoning-separator", Static)
        configurable = isinstance(self.model, OpenRouterModel)
        selector.disabled = not configurable or self.busy
        selector.label = self._active_model_name
        current = self._current_model_option()
        show_reasoning = current is not None and bool(current.reasoning_efforts)
        reasoning.display = show_reasoning
        separator.display = show_reasoning
        reasoning.disabled = self.busy
        if configurable:
            effort = self.model.reasoning_effort or "automatic"
            reasoning.label = f"reasoning: {effort}"

    @property
    def _active_model_name(self) -> str:
        if isinstance(self.model, OpenRouterModel):
            return self.model.model
        return self.model_name

    def _current_model_option(self) -> OpenRouterModelOption | None:
        model_name = self._active_model_name
        return next(
            (option for option in self.model_options if option.id == model_name),
            None,
        )

    def _start_rollout(self, state: SessionTab) -> None:
        state.running = True
        state.status = "thinking…"
        self._show_model_activity(state)
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = self.active_session.running
        self._refresh_model_controls()
        self._refresh_active_status()

    def _finish_rollout(self, state: SessionTab) -> None:
        self._stop_active_tool_events(state)
        if state.stream_entry is not None:
            self._clear_stream(state)
        state.running = False
        if state.status not in {"error", "stopped"}:
            state.status = "ready"
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = self.active_session.running
        self._refresh_model_controls()
        if state is self.active_session:
            self._refresh_sidebar()
        self._refresh_active_status()
        if state is self.active_session:
            prompt.focus()

    def _stop_active_tool_events(self, state: SessionTab) -> None:
        for tool_event in state.active_tool_events.values():
            tool_event.stop_spinner()
        state.active_tool_events.clear()

    def _set_status(self, message: str, state: SessionTab | None = None) -> None:
        state = state or self.active_session
        state.status = message
        if state is self.active_session:
            self._refresh_active_status()

    def _refresh_active_status(self) -> None:
        state = self.active_session
        self.query_one("#status-message", Static).update(state.status)
        context = self.query_one("#status-context", Static)
        context_separator = self.query_one("#context-separator", Static)
        option = self._current_model_option()
        show_context = (
            state.total_tokens is not None
            and state.context_model == self._active_model_name
            and option is not None
            and option.context_length is not None
        )
        if show_context:
            assert state.total_tokens is not None
            assert option is not None and option.context_length is not None
            remaining = max(
                0,
                min(
                    100,
                    (option.context_length - state.total_tokens)
                    * 100
                    // option.context_length,
                ),
            )
            context.update(f"context: {remaining}% left")
        context.display = show_context
        context_separator.display = show_context
        self.query_one("#status-session", Static).update(
            state.conversation.session_id[:10]
        )

    def _refresh_active_controls(self) -> None:
        state = self.active_session
        prompt = self.query_one("#prompt", Input)
        with self.prevent(Input.Changed):
            prompt.value = state.draft
        prompt.disabled = state.running
        self._refresh_sidebar()
        self._refresh_model_controls()
        self._refresh_active_status()
        if not state.running:
            prompt.focus()

    def _reset_session_ui(self, state: SessionTab) -> None:
        self._reset_stream_state(state)
        self._stop_active_tool_events(state)
        state.stream_entry = None
        state.forwarded_targets.clear()
        state.forwarded_events.clear()
        state.forwarded_children.clear()
        state.ignored_detached_jobs.update(state.detached_tool_events)
        state.detached_tool_events.clear()
        with self._detached_event_lock:
            for key in [
                key for key in self._staged_detached_events if key[0] == state.pane_id
            ]:
                _, staged = self._staged_detached_events[key]
                if staged.payload.get("status") in {"succeeded", "failed"}:
                    state.ignored_detached_jobs.discard(key[1])
                else:
                    state.ignored_detached_jobs.add(key[1])
                del self._staged_detached_events[key]
        state.draft = ""
        state.user_history.clear()
        state.history_index = None
        state.history_draft = ""
        state.running = False
        state.status = "ready"
        state.total_tokens = None
        state.context_model = None

    def _forwarded_child(
        self, state: SessionTab, event: AgentEvent, name: str, args: Any
    ) -> bool:
        parent_id = event.parent_call_id
        if (
            event.call_id
            and parent_id
            and state.forwarded_targets.get(parent_id) == name
        ):
            state.forwarded_children[event.call_id] = parent_id
            state.forwarded_events[parent_id].update_call(
                name,
                self._compact(args),
                self._argument_summary(args),
            )
            self._set_status(f"calling {name}", state)
            return True
        return False

    def _tool_response(
        self, state: SessionTab, event: AgentEvent, *, failed: bool = False
    ) -> tuple[str, str, str] | None:
        call_id = event.call_id
        if call_id in state.forwarded_children:
            return None
        if call_id in state.forwarded_targets:
            delegated = any(
                parent_id == call_id for parent_id in state.forwarded_children.values()
            )
            name = (
                state.forwarded_targets[call_id]
                if delegated
                else str(event.tool_name or "tool")
            )
            state.forwarded_targets.pop(call_id)
            state.forwarded_events.pop(call_id, None)
            state.forwarded_children = {
                child_id: parent_id
                for child_id, parent_id in state.forwarded_children.items()
                if parent_id != call_id
            }
        else:
            name = str(event.tool_name or "tool")
        payload = dict(event.payload)
        duration_ms = payload.pop("duration_ms", None)
        value = payload if failed else payload.get("result")
        duration = _format_duration(duration_ms)
        outcome_summary = f" · completed in {duration}" if duration else ""
        return name, self._compact(_summarize_tool_images(value)), outcome_summary

    @staticmethod
    def _compact(value: Any, limit: int = 1600) -> str:
        return _compact_json(value, limit)

    @staticmethod
    def _argument_summary(value: Any, limit: int = 96) -> str:
        if not isinstance(value, dict) or not value:
            return ""

        pairs = []
        for key, item in value.items():
            encoded_key = json.dumps(str(key), ensure_ascii=False)
            rendered_key = ToolboxApp._one_line(encoded_key[1:-1])
            try:
                rendered = json.dumps(item, separators=(",", ":"), ensure_ascii=False)
            except (TypeError, ValueError):
                rendered = repr(item)
            pairs.append(f"{rendered_key}={ToolboxApp._one_line(rendered)}")

        summary = f" [{', '.join(pairs)}]"
        if len(summary) <= limit:
            return summary
        return f"{summary[: limit - 2].rstrip()}…]"

    @staticmethod
    def _one_line(value: str) -> str:
        return "".join(
            character if character.isprintable() else f"\\u{ord(character):04x}"
            for character in value
        )


def run_tui(kernel: Kernel, model: Model, *, model_name: str) -> None:
    """Launch the interactive terminal application."""
    ToolboxApp(
        kernel,
        ModelProvider(kernel, model),
        model_name=model_name,
    ).run()
