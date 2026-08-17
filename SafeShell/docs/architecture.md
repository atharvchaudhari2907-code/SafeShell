# SafeShell — Member 1 Architecture (Parser, Gateway, TUI)

## Scope

This document covers the modules inside `SafeShell/` owned by Member 1:

- `SafeShell/src/parser/` — Bashlex integration + AST normalization
- `SafeShell/src/gateway/` — Command Gateway
- `SafeShell/cli/` — Textual TUI + CLI entry point

## Pipeline
## The AST contract

`command_parser.parse()` returns:

```python
{
    "command": str,
    "flags": list[str],
    "args": list[str],
    "target_path": str,
    "is_sudo": bool,
    "is_recursive": bool,
    "is_force": bool,
    "raw": str,
    "pipe_to": str,
}
```

This exactly matches what `rules_engine.check(ast, kb_entry)` and
`semantic_fusion.fuse(raw, ast=...)` already expect — confirmed by
reading their source directly. This parser is a drop-in replacement
for `semantic_fusion.parse_command()` (the team's shlex-based
stopgap), built on real Bashlex grammar instead of naive string
splitting, so it correctly handles quoting, pipes, redirects, and
multi-command lists.

## Wiring note: where semantic_fusion.py lives

`semantic_fusion.py`, `knowledge_base.py`, `rules_engine.py`, and
`semantic_search.py` currently live at the **repository root**, not
inside `SafeShell/src/`. `command_gateway.py` adds the repo root to
`sys.path` to import them. When the team migrates that logic into
`SafeShell/src/fusion/` (currently an empty placeholder), only the
import block in `command_gateway.py` needs to change.

## What this module does NOT do

- Never decides risk or intent (that's `rules_engine.py` / `semantic_fusion.py`)
- Never executes the user's command, under any circumstances
- Never exposes Bashlex internals past `command_parser.py`

## Testing

- `SafeShell/tests/test_parser.py` — 20 tests, parser only, no ML dependencies, runs in ~0.01s
- `SafeShell/tests/test_gateway.py` — 8 tests, full real pipeline (loads FAISS + sentence-transformers)