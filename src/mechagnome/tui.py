"""Textual interface for a persistent toolbox agent conversation."""

from __future__ import annotations

import difflib
import json
import shlex
from typing import Any

from rich.markdown import Markdown
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

from mechagnome.harness import AgentEvent, Harness, Model, RunCancelled
from mechagnome.kernel import Kernel
from mechagnome.openrouter import (
    OpenRouterError,
    OpenRouterModel,
    OpenRouterModelOption,
)


class ToolEvent(Collapsible):
    """One quiet, expandable tool invocation or observation."""

    SYMBOLS = {"call": "→", "response": "✓", "error": "✕"}

    def __init__(self, kind: str, tool_name: str, detail: str) -> None:
        self.kind = kind
        self.tool_name = tool_name
        self.detail = detail
        self.detail_widget = Static(
            Text(detail, style="dim"), classes="tool-event-detail"
        )
        super().__init__(
            self.detail_widget,
            title=f"{self.SYMBOLS[kind]} {tool_name}",
            collapsed=True,
            classes=f"tool-event tool-{kind}",
        )

    def update_call(self, tool_name: str, detail: str) -> None:
        """Replace a pending dispatcher row with its confirmed target call."""
        self.tool_name = tool_name
        self.detail = detail
        self.title = f"{self.SYMBOLS['call']} {tool_name}"
        self.detail_widget.update(Text(detail, style="dim"))


class ChatFeed(VerticalScroll):
    """Scrollable chat entries with interactive tool events."""

    def write(self, renderable: Any) -> None:
        self.mount(Static(renderable, classes="chat-entry"))
        self.call_after_refresh(self.scroll_end, animate=False)

    def write_tool(self, kind: str, tool_name: str, detail: str) -> ToolEvent:
        event = ToolEvent(kind, tool_name, detail)
        self.mount(event)
        self.call_after_refresh(self.scroll_end, animate=False)
        return event

    def clear(self) -> None:
        self.remove_children()


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
    """Collect a namespace name for Blank or Save as."""

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
        self.options = options
        self.options_by_id = {option.id: option for option in options}

    def compose(self) -> ComposeResult:
        picker_options = [
            (f"{option.name}  ·  {option.id}", option.id) for option in self.options
        ]
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
        self._show_capability(event.value.strip())

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
                yield Static("NAMESPACE", id="namespace-label")
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
            self._status(f"Namespace change failed: {error}")
            self.query_one("#namespace-picker", Select).value = self.selected_namespace
            return
        self.selected_namespace = name
        self._refresh_after_namespace_change(f"Switched to namespace {name}.")

    @on(Button.Pressed, "#blank-namespace")
    def blank_namespace(self) -> None:
        self.app.push_screen(
            NamespaceNameScreen(
                title="Create a blank namespace",
                action_label="Create",
            ),
            self._finish_blank_namespace,
        )

    @on(Button.Pressed, "#save-as-namespace")
    def save_as_namespace(self) -> None:
        self.app.push_screen(
            NamespaceNameScreen(
                title=f"Rename namespace {self.selected_namespace!r}",
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
            self._status(f"Blank namespace failed: {error}")
            return
        self.selected_namespace = name
        self._refresh_after_namespace_change(f"Created blank namespace {name}.")

    def _finish_save_as_namespace(self, name: str | None) -> None:
        if name is None:
            return
        if name == self.selected_namespace:
            self._status("Namespace name unchanged.")
            return
        old_name = self.selected_namespace
        try:
            self.kernel.rename_toolbox(old_name, name)
        except Exception as error:
            self._status(f"Save as failed: {error}")
            return
        self.selected_namespace = name
        self._refresh_after_namespace_change(f"Renamed namespace {old_name} to {name}.")

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
        table.add_row(
            "Calls", str(version["call_count"]), str(self.history["call_count"])
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

    #chat {
        width: 1fr;
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

    .tool-event {
        width: 1fr;
        height: auto;
        margin-bottom: 1;
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

    #status-message, #status-session, .status-separator {
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

    #stream {
        display: none;
        height: auto;
        max-height: 9;
        padding: 0 1;
        border: round #a855f7;
        background: #0d131b;
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
        model: Model,
        *,
        model_name: str,
        max_turns: int = 50,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.model = model
        self.model_name = model_name
        self.harness = Harness(kernel, max_turns=max_turns)
        self.conversation = self.harness.start(model)
        self.busy = False
        self.model_options: list[OpenRouterModelOption] = []
        self.streamed_text = ""
        self.pending_stream_text: list[str] = []
        self.stream_timer: Timer | None = None
        self.forwarded_targets: dict[str, str] = {}
        self.forwarded_events: dict[str, ToolEvent] = {}
        self.forwarded_children: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            yield ChatFeed(id="chat")
            with Vertical(id="sidebar"):
                yield Static("MODEL", classes="sidebar-title")
                yield Static(id="model-info")
                yield Static("TOOLBOX", classes="sidebar-title")
                yield RichLog(id="tools", wrap=True, markup=False, min_width=1)
        yield Static(id="stream")
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
            yield Static("·", classes="status-separator")
            yield Static(id="status-session")
        yield Input(
            placeholder="Ask the agent to build or use whatever it needs…",
            id="prompt",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Populate the initial panes and focus the prompt."""
        self._show_welcome()
        self._refresh_sidebar()
        self._refresh_model_controls()
        self._set_busy(False)
        self.query_one("#prompt", Input).focus()
        self.load_model_options()

    def on_unmount(self, event: Unmount) -> None:
        """Release a synchronous rollout before asyncio joins worker threads."""
        self.conversation.close()

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
    def submit_prompt(self, event: Input.Submitted) -> None:
        """Handle a slash command or send one message to the agent worker."""
        prompt = event.value.strip()
        if not prompt or self.busy:
            return
        event.input.value = ""
        if prompt.startswith("/") and self._command(prompt):
            return
        self._write_user(prompt)
        self._set_busy(True)
        self.run_agent(prompt)

    @work(thread=True, exclusive=True, group="agent", exit_on_error=False)
    def run_agent(self, prompt: str) -> None:
        """Run the synchronous model/tool loop without blocking the terminal UI."""
        error_reported = False

        def relay(event: AgentEvent) -> None:
            nonlocal error_reported
            if event.kind == "model_delta":
                self.call_from_thread(
                    self._queue_stream_delta,
                    str(event.payload.get("text") or ""),
                )
                return
            error_reported = error_reported or event.kind in {
                "model_failed",
                "harness_failed",
            }
            self.call_from_thread(self._display_event, event)

        try:
            self.conversation.send(prompt, on_event=relay)
        except RunCancelled:
            pass
        except (
            Exception
        ) as error:  # Provider and generated tool errors are user-visible.
            if not error_reported:
                self.call_from_thread(self._write_error, str(error))
        finally:
            self.call_from_thread(self._set_busy, False)
            self.call_from_thread(self._refresh_sidebar)

    def _display_event(self, event: AgentEvent) -> None:
        if event.kind == "user":
            return
        if event.kind == "model_delta":
            self.streamed_text += str(event.payload.get("text") or "")
            stream = self.query_one("#stream", Static)
            stream.display = True
            stream.update(Markdown(self.streamed_text))
            self._set_status("streaming…")
        elif event.kind == "model":
            content = str(event.payload.get("text") or "")
            if content:
                self.query_one("#chat", ChatFeed).write(
                    Panel(
                        Markdown(content),
                        title=self._active_model_name,
                        title_align="left",
                        border_style="bright_magenta",
                    )
                )
            self._clear_stream()
            self._set_status("planning" if event.payload.get("calls") else "answering")
        elif event.kind == "call_started":
            name = str(event.tool_name or "tool")
            args = event.payload.get("args")
            if self._forwarded_child(event, name, args):
                return
            target: str | None = None
            if name == "call_tool" and isinstance(args, dict):
                requested = args.get("name")
                if isinstance(requested, str):
                    target = requested
            displayed = self.query_one("#chat", ChatFeed).write_tool(
                "call", name, self._compact(args)
            )
            if target and event.call_id:
                self.forwarded_targets[event.call_id] = target
                self.forwarded_events[event.call_id] = displayed
            self._set_status(f"calling {name}")
        elif event.kind == "call_succeeded":
            display = self._tool_response(event)
            if display is None:
                return
            name, detail = display
            self.query_one("#chat", ChatFeed).write_tool("response", name, detail)
            self._refresh_sidebar()
        elif event.kind == "binding_changed":
            name = str(event.payload.get("name") or event.tool_name or "tool")
            version = event.payload.get("to_version")
            self.query_one("#chat", ChatFeed).write(
                Panel(
                    Text(f"active version → v{version}"),
                    title=f"toolbox · {name}",
                    title_align="left",
                    border_style="blue",
                )
            )
            self._refresh_sidebar()
        elif event.kind == "call_failed":
            display = self._tool_response(event, failed=True)
            if display is None:
                return
            name, detail = display
            self.query_one("#chat", ChatFeed).write_tool("error", name, detail)
            self._set_status(f"{name} failed")
        elif event.kind in {"model_failed", "harness_failed"}:
            self._clear_stream()
            self._write_error(str(event.payload.get("message") or event.payload))
        elif event.kind == "cancelled":
            partial = self.streamed_text + "".join(self.pending_stream_text)
            self._clear_stream()
            content = partial or str(event.payload.get("message") or "Rollout stopped.")
            self.query_one("#chat", ChatFeed).write(
                Panel(
                    Markdown(content),
                    title=f"{self._active_model_name} · stopped",
                    title_align="left",
                    border_style="yellow",
                )
            )
            self._set_status("stopped")

    def _queue_stream_delta(self, text: str) -> None:
        if not text:
            return
        self.pending_stream_text.append(text)
        if self.stream_timer is None:
            self.stream_timer = self.set_timer(0.05, self._flush_stream_delta)

    def _flush_stream_delta(self) -> None:
        self.stream_timer = None
        if not self.pending_stream_text:
            return
        self.streamed_text += "".join(self.pending_stream_text)
        self.pending_stream_text.clear()
        stream = self.query_one("#stream", Static)
        stream.display = True
        stream.update(Markdown(self.streamed_text))
        self._set_status("streaming…")

    def _clear_stream(self) -> None:
        if self.stream_timer is not None:
            self.stream_timer.stop()
            self.stream_timer = None
        self.pending_stream_text.clear()
        self.streamed_text = ""
        stream = self.query_one("#stream", Static)
        stream.update("")
        stream.display = False

    def _command(self, prompt: str) -> bool:
        try:
            parts = shlex.split(prompt)
        except ValueError as error:
            self._write_error(str(error))
            return True
        command = parts[0]
        arguments = parts[1:]
        if command in {"/quit", "/q"}:
            self.action_quit()
        elif command in {"/new", "/reset"}:
            self.action_new_session()
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
        self.conversation.close()
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
        self.query_one("#chat", ChatFeed).write(
            Panel(table, title="toolbox namespaces", border_style="blue")
        )

    def action_new_session(self) -> None:
        """Begin a fresh model conversation without deleting toolbox state."""
        if self.busy:
            return
        self.conversation = self.harness.start(self.model)
        self.query_one("#chat", ChatFeed).clear()
        self.forwarded_targets.clear()
        self.forwarded_events.clear()
        self.forwarded_children.clear()
        self._show_welcome()
        self._refresh_sidebar()
        self._set_status("new session")

    def action_stop_rollout(self) -> None:
        """Stop the active model stream or tool subprocess."""
        if self.busy:
            if self.conversation.cancel():
                self._set_status("stopping…")
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
        self.query_one("#chat", ChatFeed).write(
            Panel(table, title="active toolbox", border_style="blue")
        )

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
        self.query_one("#chat", ChatFeed).write(
            Panel(
                Markdown(
                    "**Commands**\n\n"
                    "- `Esc` — stop the active rollout\n"
                    "- `/new` — start a new saved conversation\n"
                    "- `/tools` or `Ctrl+T` — toggle tool management\n"
                    "- `/toolbox list` — list namespaces and active order\n"
                    "- `/toolbox create NAME [CWD]` — create a namespace\n"
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
        table = Table("Session", "Events", "Created", box=None, expand=True)
        for session in sessions:
            table.add_row(
                session["id"][:12],
                str(session["event_count"]),
                session["created_at"][:19],
            )
        self.query_one("#chat", ChatFeed).write(
            Panel(table, title="saved sessions", border_style="blue")
        )

    def _show_welcome(self) -> None:
        readiness = "ready" if getattr(self.model, "ready", True) else "API key missing"
        self.query_one("#chat", ChatFeed).write(
            Panel(
                Markdown(
                    f"**{self._active_model_name}** · {readiness}\n\n"
                    "Ask for a task. The agent begins with only five core operations "
                    "and grows the toolbox as needed. Use `/help` for TUI commands."
                ),
                title="mechagnome",
                border_style="bright_blue",
            )
        )
        if isinstance(self.model, OpenRouterModel) and not self.model.ready:
            self._write_error(
                f"Set {self.model.api_key_env} in your environment, then restart."
            )

    def _write_user(self, prompt: str) -> None:
        self.query_one("#chat", ChatFeed).write(
            Panel(
                Markdown(prompt),
                title="you",
                title_align="left",
                border_style="yellow",
            )
        )

    def _write_error(self, message: str) -> None:
        self.query_one("#chat", ChatFeed).write(
            Panel(Text(message), title="error", border_style="red")
        )
        self._set_status("error")

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
        selector.disabled = not configurable
        selector.label = self._active_model_name
        current = self._current_model_option()
        show_reasoning = current is not None and bool(current.reasoning_efforts)
        reasoning.display = show_reasoning
        separator.display = show_reasoning
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

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = busy
        if busy:
            self._set_status("thinking…")
        else:
            self._set_status("ready")
            prompt.focus()

    def _set_status(self, message: str) -> None:
        self.query_one("#status-message", Static).update(message)
        self.query_one("#status-session", Static).update(
            self.conversation.session_id[:10]
        )

    def _forwarded_child(self, event: AgentEvent, name: str, args: Any) -> bool:
        parent_id = event.parent_call_id
        if (
            event.call_id
            and parent_id
            and self.forwarded_targets.get(parent_id) == name
        ):
            self.forwarded_children[event.call_id] = parent_id
            self.forwarded_events[parent_id].update_call(name, self._compact(args))
            self._set_status(f"calling {name}")
            return True
        return False

    def _tool_response(
        self, event: AgentEvent, *, failed: bool = False
    ) -> tuple[str, str] | None:
        call_id = event.call_id
        if call_id in self.forwarded_children:
            return None
        if call_id in self.forwarded_targets:
            delegated = any(
                parent_id == call_id for parent_id in self.forwarded_children.values()
            )
            name = (
                self.forwarded_targets[call_id]
                if delegated
                else str(event.tool_name or "tool")
            )
            self.forwarded_targets.pop(call_id)
            self.forwarded_events.pop(call_id, None)
            self.forwarded_children = {
                child_id: parent_id
                for child_id, parent_id in self.forwarded_children.items()
                if parent_id != call_id
            }
        else:
            name = str(event.tool_name or "tool")
        value = event.payload if failed else event.payload.get("result")
        return name, self._compact(value)

    @staticmethod
    def _compact(value: Any, limit: int = 1600) -> str:
        try:
            rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            rendered = repr(value)
        return rendered if len(rendered) <= limit else f"{rendered[:limit]}\n…"


def run_tui(kernel: Kernel, model: Model, *, model_name: str) -> None:
    """Launch the interactive terminal application."""
    ToolboxApp(kernel, model, model_name=model_name).run()
