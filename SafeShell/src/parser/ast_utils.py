"""
ast_utils.py

Low-level Bashlex integration for SafeShell's parser layer.

This module's ONLY job is turning a raw command string into Bashlex's
own AST node objects, and normalizing Bashlex's various exceptions
into SafeShell's own exception types. It answers "what is the
structure of this Bash command?" -- nothing else.

It does NOT decide risk, does NOT classify intent, and NEVER executes
anything. Those responsibilities belong to rules_engine.py and
semantic_fusion.py (owned by other team members), which command_parser.py
in this same package feeds into.
"""

from __future__ import annotations

from typing import List

import bashlex
from bashlex.errors import ParsingError


class CommandParseError(Exception):
    """Base class for all parsing-related errors in SafeShell's parser."""

    error_type: str = "parse_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class EmptyCommandError(CommandParseError):
    """Raised when the input is empty or whitespace-only."""

    error_type = "empty_command"


class InvalidSyntaxError(CommandParseError):
    """Raised when Bashlex cannot parse the input (unclosed quotes,
    broken pipes, malformed redirects, etc.)."""

    error_type = "invalid_syntax"


def parse_bash(raw_command: str) -> List:
    """Parse a raw Bash command string into Bashlex AST node(s).

    Raises:
        EmptyCommandError: if the input is empty/whitespace-only.
        InvalidSyntaxError: if Bashlex cannot parse the input.
    """
    if raw_command is None or raw_command.strip() == "":
        raise EmptyCommandError("Command input is empty or whitespace-only.")

    try:
        return bashlex.parse(raw_command)
    except ParsingError as exc:
        raise InvalidSyntaxError(f"Unable to parse Bash syntax: {exc}") from exc
    except NotImplementedError as exc:
        raise InvalidSyntaxError(f"Unsupported Bash syntax: {exc}") from exc
    except (ValueError, IndexError) as exc:
        raise InvalidSyntaxError(f"Invalid Bash syntax: {exc}") from exc


def first_command_node(ast_nodes: List):
    """Return the first meaningful node (command/pipeline/list) from a
    parsed AST, or None if ast_nodes is empty."""
    return ast_nodes[0] if ast_nodes else None