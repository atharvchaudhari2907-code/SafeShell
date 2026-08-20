"""
terminal.py

SafeShell Textual TUI (Member 1).

Collects a raw Bash command from the user, sends it to the Command
Gateway, and displays the result: the normalized AST (from our
Bashlex parser) and the real analysis result from
semantic_fusion.fuse() (risk level, action, explanation, alternative,
matched rule, semantic matches). Also runs the command (unless
BLOCKed) and shows real output.

This module contains NO parsing logic and NO risk/intent logic of its
own -- it only calls CommandGateway.process() and renders whatever
comes back.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SAFESHELL_ROOT = Path(__file__).resolve().parents[1]  # .../SafeShell
_REPO_ROOT = _SAFESHELL_ROOT.parent

for _p in (str(_REPO_ROOT), str(_SAFESHELL_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contextlib
import io
import shlex
import subprocess
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


class OutputPanel(Vertical):
    """Shows real command output after execution (skipped if BLOCKed)."""

    DEFAULT_CSS = """
    OutputPanel {
        border: round $success;
        padding: 1 2;
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Output", classes="panel-title")
        yield Static("Not executed.", id="output-body")

    def show_output(self, stdout: str, stderr: str) -> None:
        body = self.query_one("#output-body", Static)
        text = stdout.strip() or "(no output)"
        if stderr.strip():
            text += f"\n[red]{stderr.strip()}[/red]"
        body.update(text)

    def show_blocked(self, reason: str) -> None:
        body = self.query_one("#output-body", Static)
        body.update(f"[bold red]NOT EXECUTED[/bold red] — {reason}")

    def clear(self) -> None:
        self.query_one("#output-body", Static).update("Not executed.")


class StatusPanel(Static):
    """Displays READY / PARSING / ANALYZING / ERROR / GETTING READY."""

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
            yield Button("Dry Run", id="dryrun-button", variant="warning")
        yield ParsedCommandPanel(id="parsed-panel")
        yield AnalysisPanel(id="analysis-panel")
        yield OutputPanel(id="output-panel")
        yield StatusPanel(id="status-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#status-panel", StatusPanel).set_status("READY")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "analyze-button":
            self._run_analysis(dry_run=False)
        elif event.button.id == "dryrun-button":
            self._run_analysis(dry_run=True)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "command-input":
            self._run_analysis(dry_run=False)

    def _run_analysis(self, dry_run: bool = False) -> None:
        status = self.query_one("#status-panel", StatusPanel)
        parsed_panel = self.query_one("#parsed-panel", ParsedCommandPanel)
        analysis_panel = self.query_one("#analysis-panel", AnalysisPanel)
        output_panel = self.query_one("#output-panel", OutputPanel)
        raw_command = self.query_one("#command-input", Input).value

        status.set_status("PARSING")
        analysis_panel.clear()
        output_panel.clear()

        result = self.gateway.process(raw_command)

        if result["status"] == "error":
            status.set_status("ERROR", result.get("message", ""))
            parsed_panel.show_error(result.get("error_type", "error"), result.get("message", ""))
            return

        parsed_panel.show_parsed(result["parsed_command"])

        status.set_status("ANALYZING")
        analysis_panel.show_analysis(result["analysis"])

        action = result["analysis"].get("action", "BLOCK")

        if action == "BLOCK":
            output_panel.show_blocked("Command blocked by risk analysis.")
        elif dry_run:
            output_panel.show_output(f"[DRY RUN] Would execute: {raw_command}", "")
        else:
            try:
                run_result = subprocess.run(
                    shlex.split(raw_command), capture_output=True, text=True, timeout=10
                )
                output_panel.show_output(run_result.stdout, run_result.stderr)
            except Exception as exc:
                output_panel.show_blocked(f"Execution error: {exc}")

        status.set_status("READY", "Analysis complete")


class SafeShellApp(App):
    """SafeShell Terminal UI."""

    TITLE = "SafeShell"

    CSS = """
    #input-area { height: auto; padding: 1 2; border: round $accent; }
    .section-label { color: $text-muted; padding-bottom: 1; }
    #command-input { margin-bottom: 1; }
    .panel-title { text-style: bold; padding-bottom: 1; }
    #parsed-panel, #analysis-panel, #output-panel { margin: 1 2 0 2; }
    #status-panel { margin: 1 2 1 2; }
    """

    def on_mount(self) -> None:
        self.push_screen(SafeShellScreen())
        self.call_after_refresh(self._warm_up_backend)

    def _warm_up_backend(self) -> None:
        status = self.screen.query_one("#status-panel", StatusPanel)
        status.set_status("GETTING READY", "loading models, one-time...")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            CommandGateway().process("echo warmup")
        status.set_status("READY")


def run() -> None:
    SafeShellApp().run()


if __name__ == "__main__":
    run()