"""
safeshell.py

SafeShell CLI entry point (Member 1).

This is the file a user actually runs to start SafeShell. It handles
argument parsing and launches the Textual TUI (terminal.py). All
parsing/analysis logic lives elsewhere -- this file only wires things
together and starts the app.

Usage
-----
From the repository root:

    python -m SafeShell.cli.safeshell

Or directly:

    python SafeShell/cli/safeshell.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure both the repo root (for semantic_fusion.py etc.) and the
# SafeShell/ package root (for `src.*` imports) are importable,
# regardless of the working directory this script is launched from.
_SAFESHELL_ROOT = Path(__file__).resolve().parents[1]   # .../SafeShell
_REPO_ROOT = _SAFESHELL_ROOT.parent                       # repo root

for path in (str(_REPO_ROOT), str(_SAFESHELL_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="safeshell",
        description="SafeShell -- Context-Aware Linux Command Safety & Intent Analysis",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        help="Analyze a single command non-interactively and exit, "
             "instead of launching the TUI.",
    )
    args = parser.parse_args()

    if args.command:
        _run_single_command(args.command)
    else:
        _run_tui()


def _run_tui() -> None:
    from cli.terminal import run
    run()


def _run_single_command(raw_command: str) -> None:
    """Non-interactive mode: analyze one command and print the result,
    without launching the TUI. Useful for scripting/CI and for a
    quick sanity check without the full terminal UI.
    """
    from src.gateway.command_gateway import CommandGateway

    gateway = CommandGateway()
    result = gateway.process(raw_command)

    if result["status"] == "error":
        print(f"PARSE ERROR [{result['error_type']}]: {result['message']}")
        sys.exit(1)

    analysis = result["analysis"]
    print(f"Command:  {raw_command}")
    print(f"Risk:     {analysis.get('final_risk', 'unknown').upper()}")
    print(f"Action:   {analysis.get('action', 'unknown')}")
    print(f"Rule:     {analysis.get('rule_result', {}).get('matched_rule', '-')}")
    print(f"Explain:  {analysis.get('explanation', '-')}")
    print(f"Alt:      {analysis.get('suggested_alternative', '-')}")


if __name__ == "__main__":
    main()