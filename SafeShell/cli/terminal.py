"""
terminal.py

SafeShell Textual TUI (Member 1).

Collects a raw Bash command from the user, sends it to the Command
Gateway, and displays the result: the normalized AST (from our
Bashlex parser) and the real analysis result from
semantic_fusion.fuse() (risk level, action, explanation, alternative,
matched rule, semantic matches).

This module contains NO parsing logic and NO risk/intent logic of its
own -- it only calls CommandGateway.process() and renders whatever
comes back. It NEVER executes the user's command.
"""

from __future__ import annotations

from typing import Any, Dict

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Static

from src.gateway.command_gateway import CommandGateway

_RISK_COLOR = {
    "low": "green",
    "medium": "yellow",
    "high": "red",
    "critical": "bold red",
}

_ACTION_COLOR = {
    "ALLOW": "green",
    "WARN": "yellow",
    "WARN_CONFIRM": "red",
    "BLOCK": "bold red",
}


class ParsedCommandPanel(Vertical):
    """Displays the normalized AST dict from our Bashlex parser."""

    DEFAULT_CSS = """
    ParsedCommandPanel {
        border: round $primary;
        padding: 1 2;
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Parsed Command", classes="panel-title")
        yield Static("Enter a command and press Analyze.", id="parsed-body")

    def show_parsed(self, ast: Dict[str, Any]) -> None:
        body = self.query_one("#parsed-body", Static)
        lines = [
            f"Command: {ast.get('command', '')}",
            f"Flags: {', '.join(ast.get('flags', [])) or '-'}",
            f"Args: {', '.join(ast.get('args', [])) or '-'}",
            f"Target path: {ast.get('target_path') or '-'}",
            f"Privileged (sudo): {'YES' if ast.get('is_sudo') else 'no'}",
            f"Recursive: {'YES' if ast.get('is_recursive') else 'no'}",
            f"Force: {'YES' if ast.get('is_force') else 'no'}",
        ]
        if ast.get("pipe_to"):
            lines.append(f"Piped to: {ast['pipe_to']}")
        body.update("\n".join(lines))

    def show_error(self, error_type: str, message: str) -> None:
        body = self.query_one("#parsed-body", Static)
        body.update(f"[bold red]PARSE ERROR[/bold red]\n{error_type}: {message}")

    def clear(self) -> None:
        self.query_one("#parsed-body", Static).update("Enter a command and press Analyze.")


class AnalysisPanel(Vertical):
    """Displays semantic_fusion.fuse()'s real analysis result."""

    DEFAULT_CSS = """
    AnalysisPanel {
        border: round $secondary;
        padding: 1 2;
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Backend Analysis", classes="panel-title")
        yield Static("No analysis yet.", id="analysis-body")

    def show_analysis(self, analysis: Dict[str, Any]) -> None:
        body = self.query_one("#analysis-body", Static)
        risk = analysis.get("final_risk", "unknown")
        action = analysis.get("action", "unknown")
        rule = analysis.get("rule_result", {}).get("matched_rule", "-")
        matches = analysis.get("semantic_matches", [])

        risk_color = _RISK_COLOR.get(risk, "white")
        action_color = _ACTION_COLOR.get(action, "white")

        lines = [
            f"Risk: [{risk_color}]{risk.upper()}[/{risk_color}]",
            f"Action: [{action_color}]{action}[/{action_color}]",
            f"Matched rule: {rule}",
            "",
            f"Explanation: {analysis.get('explanation', '-')}",
            f"Suggested alternative: {analysis.get('suggested_alternative', '-')}",
        ]
        if matches:
            lines.append("")
            lines.append(f"Semantic matches: {len(matches)} found")
        body.update("\n".join(lines))

    def clear(self) -> None:
        self.query_one("#analysis-body", Static).update("No analysis yet.")


class StatusPanel(Static):
    """Displays READY / PARSING / ANALYZING / ERROR."""

    DEFAULT_CSS = """
    StatusPanel {
        padding: 0 2;
        height: auto;
        color: $text-muted;
    }
    """

    def set_status(self, status: str, detail: str = "") -> None:
        text = f"Status: {status}"
        if detail:
            text += f" - {detail}"
        self.update(text)


class SafeShellScreen(Screen):
    """The primary SafeShell TUI screen."""

    def __init__(self, gateway: CommandGateway | None = None) -> None:
        super().__init__()
        self.gateway = gateway or CommandGateway()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="input-area"):
            yield Static("Enter Linux command:", classes="section-label")
            yield Input(placeholder="e.g. sudo rm -rf /home/project", id="command-input")
            yield Button("Analyze", id="analyze-button", variant="primary")
        yield ParsedCommandPanel(id="parsed-panel")
        yield AnalysisPanel(id="analysis-panel")
        yield StatusPanel(id="status-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#status-panel", StatusPanel).set_status("READY")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "analyze-button":
            self._run_analysis()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "command-input":
            self._run_analysis()

    def _run_analysis(self) -> None:
        status = self.query_one("#status-panel", StatusPanel)
        parsed_panel = self.query_one("#parsed-panel", ParsedCommandPanel)
        analysis_panel = self.query_one("#analysis-panel", AnalysisPanel)
        raw_command = self.query_one("#command-input", Input).value

        status.set_status("PARSING")
        analysis_panel.clear()

        result = self.gateway.process(raw_command)

        if result["status"] == "error":
            status.set_status("ERROR", result.get("message", ""))
            parsed_panel.show_error(result.get("error_type", "error"), result.get("message", ""))
            return

        parsed_panel.show_parsed(result["parsed_command"])

        status.set_status("ANALYZING")
        analysis_panel.show_analysis(result["analysis"])

        status.set_status("READY", "Analysis complete")


class SafeShellApp(App):
    """SafeShell Terminal UI. Never executes user-submitted commands."""

    TITLE = "SafeShell"

    CSS = """
    #input-area { height: auto; padding: 1 2; border: round $accent; }
    .section-label { color: $text-muted; padding-bottom: 1; }
    #command-input { margin-bottom: 1; }
    .panel-title { text-style: bold; padding-bottom: 1; }
    #parsed-panel, #analysis-panel { margin: 1 2 0 2; }
    #status-panel { margin: 1 2 1 2; }
    """

    def on_mount(self) -> None:
        self.push_screen(SafeShellScreen())


def run() -> None:
    SafeShellApp().run()


if __name__ == "__main__":
    run()