"""
command_gateway.py

Command Gateway -- the single public entry point that turns raw user
input into a standardized response, by orchestrating:

    raw input -> validation -> Bashlex parser -> AST dict
              -> semantic_fusion.fuse() -> standardized response

This is the ONLY module the TUI (SafeShell/cli/) talks to. It never
exposes Bashlex internals, and it never performs risk/intent analysis
itself -- that all happens inside semantic_fusion.fuse(), which is
owned by other team members.

Wiring note
-----------
semantic_fusion.py (and the modules it depends on: knowledge_base.py,
rules_engine.py, semantic_search.py) currently live at the repository
root, not inside SafeShell/src/. Until the team migrates that logic
into SafeShell/src/fusion/ (currently an empty placeholder package),
this Gateway imports them directly from the repo root by adding it to
sys.path. When that migration happens, only the import block below
needs to change -- nothing else in this file, and nothing in the TUI,
depends on where semantic_fusion.py physically lives.

Response shapes
----------------
Successful response:
    {
        "status": "success",
        "raw_command": "...",
        "parsed_command": {...},   # the AST dict, see command_parser.py
        "analysis": {...},          # semantic_fusion.fuse()'s full result
    }

Error response (empty input / invalid Bash syntax):
    {
        "status": "error",
        "error_type": "...",
        "message": "...",
    }
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# --- repo-root wiring (see module docstring) ---------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import semantic_fusion  # noqa: E402  (repo-root module, path set up above)

from src.parser.ast_utils import CommandParseError  # noqa: E402
from src.parser.command_parser import parse as parse_command  # noqa: E402


class CommandGateway:
    """Orchestrates the parsing + analysis pipeline and returns a
    stable response shape, regardless of success or failure.
    """

    def process(self, raw_command: str) -> Dict[str, Any]:
        """Process a raw command string end-to-end.

        Args:
            raw_command: The exact text the user typed into the TUI.

        Returns:
            A standardized dict response (see module docstring). This
            method never raises -- all parsing failures are caught and
            converted into an error response, and the TUI never needs
            to know the difference between a parse failure and any
            other kind of failure.
        """
        try:
            parsed_command = parse_command(raw_command)
        except CommandParseError as exc:
            return {
                "status": "error",
                "error_type": exc.error_type,
                "message": exc.message,
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            # Absolute last line of defense: the TUI must never crash
            # because of an unexpected parser error on user input.
            return {
                "status": "error",
                "error_type": "unexpected_error",
                "message": f"Unexpected parser error: {exc}",
            }

        try:
            analysis = semantic_fusion.fuse(raw_command, ast=parsed_command)
        except Exception as exc:  # pragma: no cover - defensive fallback
            # The backend (owned by other members) is out of Member 1's
            # control; a failure there must still not crash the TUI.
            return {
                "status": "error",
                "error_type": "analysis_error",
                "message": f"Backend analysis failed: {exc}",
            }

        return {
            "status": "success",
            "raw_command": raw_command,
            "parsed_command": parsed_command,
            "analysis": analysis,
        }