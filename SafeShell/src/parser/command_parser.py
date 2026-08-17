"""
command_parser.py

SafeShell's real Bashlex-based command parser.


...
Public API
----------
parse(raw_command: str) -> dict

The output dict matches the AST contract already consumed by
``rules_engine.check(ast, kb_entry)`` and ``semantic_fusion.fuse(raw,
ast=...)`` exactly:

    {
        "command": str,        # base command name, e.g. "rm"
        "flags": list[str],    # e.g. ["-rf", "-r", "-f"] — combined short
                                # flags expanded, matching their original behavior
        "args": list[str],     # non-flag arguments
        "target_path": str,    # best-guess target file/dir
        "is_sudo": bool,
        "is_recursive": bool,
        "is_force": bool,
        "raw": str,
        "pipe_to": str,        # downstream command name if piped
    }
...


This is a drop-in, more robust replacement for the stopgap
``semantic_fusion.parse_command()`` (a shlex-based parser explicitly
labeled "Bashlex-style AST" in that module's docstring -- a signal
left by the team for this exact module to replace it).

Public API
----------
parse(raw_command: str) -> dict

The output dict matches the AST contract already consumed by
``rules_engine.check(ast, kb_entry)`` and ``semantic_fusion.fuse(raw,
ast=...)`` exactly -- see module docstring in ast_utils.py.

This module NEVER decides risk, NEVER classifies intent, and NEVER
executes anything.
"""

from __future__ import annotations

import os
from typing import List

from src.parser.ast_utils import (
    EmptyCommandError,
    InvalidSyntaxError,
    first_command_node,
    parse_bash,
)

_PRIVILEGE_PREFIXES = {"sudo", "doas", "pkexec"}
_RECURSIVE_MARKERS = ("r", "R", "recursive")
_FORCE_MARKERS = ("f", "force")


def _empty_ast(raw: str, pipe_to: str = "") -> dict:
    return {
        "command": "",
        "flags": [],
        "args": [],
        "target_path": "",
        "is_sudo": False,
        "is_recursive": False,
        "is_force": False,
        "raw": raw,
        "pipe_to": pipe_to,
    }


def _expand_flag(flag: str) -> List[str]:
    """Expand a combined short flag like "-rf" into ["-rf", "-r", "-f"],
    matching the original parser's behavior so downstream rules (which
    check for "-r" / "-f" individually) keep working unchanged."""
    expanded = [flag]
    if flag.startswith("-") and not flag.startswith("--") and len(flag) > 2:
        for char in flag[1:]:
            short_flag = f"-{char}"
            if short_flag not in expanded:
                expanded.append(short_flag)
    return expanded


def _words_from_command_node(node) -> List[str]:
    """Extract word tokens (command name + args/flags) from a Bashlex
    `command` node. Leading environment-variable assignments (e.g.
    "FOO=bar") are Bashlex `assignment` nodes, distinct from `word`
    nodes, and are intentionally excluded here so they're never
    mistaken for the command name."""
    words: List[str] = []
    for part in getattr(node, "parts", []):
        if getattr(part, "kind", None) == "word":
            words.append(part.word)
    return words


def _first_pipe_target(pipeline_node) -> str:
    """Given a Bashlex `pipeline` node, return the command name of the
    first command after the first pipe, or "" if there is none."""
    parts = getattr(pipeline_node, "parts", [])
    seen_pipe = False
    for part in parts:
        kind = getattr(part, "kind", None)
        if kind == "pipe":
            seen_pipe = True
            continue
        if seen_pipe and kind == "command":
            words = _words_from_command_node(part)
            idx = 0
            while idx < len(words) and words[idx] in _PRIVILEGE_PREFIXES:
                idx += 1
            return words[idx] if idx < len(words) else ""
    return ""


def _primary_command_node(node):
    """Given a top-level Bashlex node (command / pipeline / list),
    return the first `command` node to analyze, and the pipe target
    (if any), or (None, "") if none is found."""
    kind = getattr(node, "kind", None)

    if kind == "command":
        return node, ""

    if kind == "pipeline":
        pipe_to = _first_pipe_target(node)
        for part in getattr(node, "parts", []):
            if getattr(part, "kind", None) == "command":
                return part, pipe_to
        return None, pipe_to

    if kind == "list":
        # Multi-command lists (&&, ||, ;): analyze the first segment,
        # matching the original parser's behavior of only looking at
        # what comes before the first "|".
        for part in getattr(node, "parts", []):
            part_kind = getattr(part, "kind", None)
            if part_kind == "command":
                return part, ""
            if part_kind == "pipeline":
                return _primary_command_node(part)
        return None, ""

    # compound nodes (subshells, if/for/while): not unpacked here.
    return None, ""


def _extract_target_path(args: List[str]) -> str:
    """Mirror the original parser's target_path heuristic exactly:
    first arg (or the value of a KEY=value arg) that looks like a
    path, else the last positional argument."""
    for arg in args:
        if "=" in arg:
            val = arg.split("=", 1)[1]
            if val.startswith("/") or val.startswith("."):
                return val
        elif arg.startswith("/") or arg.startswith("."):
            return arg
    return args[-1] if args else ""


def parse(raw_command: str) -> dict:
    """Parse a raw Bash command string into SafeShell's AST dict.

    Raises EmptyCommandError / InvalidSyntaxError for genuinely
    invalid input (propagated from ast_utils.parse_bash) -- the
    Command Gateway is responsible for catching these and converting
    them into a structured error response for the TUI.
    """
    raw = (raw_command or "").strip()

    ast_nodes = parse_bash(raw_command)
    node = first_command_node(ast_nodes)
    if node is None:
        return _empty_ast(raw)

    command_node, pipe_to = _primary_command_node(node)
    if command_node is None:
        return _empty_ast(raw, pipe_to)

    words = _words_from_command_node(command_node)
    if not words:
        return _empty_ast(raw, pipe_to)

    is_sudo = False
    idx = 0
    while idx < len(words) and words[idx] in _PRIVILEGE_PREFIXES:
        is_sudo = True
        idx += 1

    if idx >= len(words):
        result = _empty_ast(raw, pipe_to)
        result["command"] = "sudo"
        result["is_sudo"] = True
        return result

    command = os.path.basename(words[idx])
    rest = words[idx + 1 :]

    flags: List[str] = []
    args: List[str] = []
    is_recursive = False
    is_force = False

    for tok in rest:
        if tok.startswith("-"):
            for expanded in _expand_flag(tok):
                if expanded not in flags:
                    flags.append(expanded)
            if any(marker in tok for marker in _RECURSIVE_MARKERS):
                is_recursive = True
            if any(marker in tok for marker in _FORCE_MARKERS):
                is_force = True
        else:
            args.append(tok)

    return {
        "command": command,
        "flags": flags,
        "args": args,
        "target_path": _extract_target_path(args),
        "is_sudo": is_sudo,
        "is_recursive": is_recursive,
        "is_force": is_force,
        "raw": raw,
        "pipe_to": pipe_to,
    }