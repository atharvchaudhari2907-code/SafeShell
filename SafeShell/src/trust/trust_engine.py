"""
trust_engine.py

Public entry point for the trust/personalization subsystem.

Other modules (like src/execution/executor.py) should import ONLY
from here, not reach into trust_score.py or trust_policy.py directly.
This keeps the internal split between "how trust is scored" and
"how trust affects the final decision" free to change without
breaking callers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.trust.trust_policy import decide
from src.trust.trust_score import TrustScoreStore, command_signature


_default_store = TrustScoreStore()


def evaluate(
    raw_command: str,
    role: str,
    user_id: str,
    analysis: Dict[str, Any],
    parsed_command: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide what should happen with this command, given the real
    risk analysis, the user's role, and their trust history.

    Returns the same dict shape as trust_policy.decide():
        {"final_action": ..., "skip_confirmation": ..., "trust_score": ..., "reason": ...}
    """
    return decide(
        raw_command=raw_command,
        role=role,
        user_id=user_id,
        analysis=analysis,
        parsed_command=parsed_command,
        trust_store=_default_store,
    )


def record_outcome(
    user_id: str,
    parsed_command: Dict[str, Any],
    outcome: str,           # "ALLOWED" / "REJECTED" / "BLOCKED"
    risk_level: str,        # analysis["final_risk"] from the real pipeline
    context: Optional[str] = None,
) -> None:
    """Record what actually happened with a command, so future trust
    scores for this user + command pattern reflect it. Call this AFTER
    executor.py finishes handling the command (whether it ran,
    was rejected, or was blocked).
    """
    signature = command_signature(
        parsed_command.get("command", ""),
        parsed_command.get("flags", []),
        parsed_command.get("target_path", ""),
    )
    _default_store.record(user_id, signature, outcome, risk_level, context)