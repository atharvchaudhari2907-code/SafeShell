"""
trust_policy.py

Policy/Decision Engine (Member 5 spec).

This is where the hard rule lives:

    "Reduce friction but NEVER override a critical current risk."

It combines:
    - the REAL risk verdict from semantic_fusion.fuse() (via
      CommandGateway) -- the current, authoritative risk assessment
    - the user's role-based policy (src/trust/user_profile.py)
    - the user's adaptive trust score (src/trust/trust_score.py)

...into one final decision. Trust score can only ever REDUCE friction
(skip a confirmation prompt) on WARN-level commands for a role that's
allowed to override WARN in the first place. It can NEVER touch a
BLOCK/critical verdict, regardless of score, role, or history.
"""

from __future__ import annotations

from typing import Any, Dict

from src.trust.trust_score import TrustScoreStore, command_signature, compute_trust_score
from src.trust.user_profile import get_user_profile

# Which action levels each role's policy is even ALLOWED to consider
# overriding. This is the ceiling; trust score decides whether to use
# that ceiling on any given command.
_ROLE_OVERRIDE_LEVELS = {
    "strict":   set(),                      # normal user: never overrides anything
    "moderate": {"WARN"},                   # developer: may override WARN only
    "high":     {"WARN", "WARN_CONFIRM"},   # admin: may override WARN + WARN_CONFIRM
}

# Trust score needed before friction is actually reduced (i.e. before
# we skip asking for confirmation), even for a role that's allowed to.
_TRUST_THRESHOLD_TO_SKIP_CONFIRMATION = 0.7


def decide(
    raw_command: str,
    role: str,
    user_id: str,
    analysis: Dict[str, Any],
    parsed_command: Dict[str, Any],
    trust_store: TrustScoreStore | None = None,
) -> Dict[str, Any]:
    """Produce the final execution decision for a command.

    Args:
        analysis: the real result of semantic_fusion.fuse() (from
            CommandGateway.process()["analysis"]) -- the authoritative
            current risk assessment.
        parsed_command: the AST dict from CommandGateway.process()
            (used to build the trust signature).

    Returns:
        {
            "final_action": "ALLOW" | "CONFIRM" | "BLOCK",
            "skip_confirmation": bool,   # True if trust reduced friction
            "trust_score": float,
            "reason": str,
        }
    """
    action = analysis.get("action", "BLOCK")
    profile = get_user_profile(role)
    risk_level_role = profile["risk_level"]

    # HARD RULE: BLOCK from the real pipeline is never overridable,
    # by anyone, regardless of role or trust score.
    if action == "BLOCK":
        return {
            "final_action": "BLOCK",
            "skip_confirmation": False,
            "trust_score": None,
            "reason": "Real analysis pipeline marked this as BLOCK -- "
                      "no role or trust score can override a critical risk.",
        }

    if action == "ALLOW":
        return {
            "final_action": "ALLOW",
            "skip_confirmation": True,
            "trust_score": None,
            "reason": "Real analysis pipeline marked this as low risk.",
        }

    # action is WARN or WARN_CONFIRM: this is the only zone where
    # trust score is allowed to reduce friction, and only if the
    # user's role is permitted to consider overriding this level.
    if action not in _ROLE_OVERRIDE_LEVELS.get(risk_level_role, set()):
        return {
            "final_action": "CONFIRM",
            "skip_confirmation": False,
            "trust_score": None,
            "reason": f"Role '{role}' cannot reduce friction on {action} commands.",
        }

    store = trust_store or TrustScoreStore()
    signature = command_signature(
        parsed_command.get("command", ""),
        parsed_command.get("flags", []),
        parsed_command.get("target_path", ""),
    )
    history = store.get_history(user_id, signature)
    trust_score = compute_trust_score(history)

    if trust_score >= _TRUST_THRESHOLD_TO_SKIP_CONFIRMATION:
        return {
            "final_action": "CONFIRM",
            "skip_confirmation": True,
            "trust_score": trust_score,
            "reason": f"Trust score {trust_score} is high enough to skip the "
                      f"confirmation prompt for this familiar, non-critical command.",
        }

    return {
        "final_action": "CONFIRM",
        "skip_confirmation": False,
        "trust_score": trust_score,
        "reason": f"Trust score {trust_score} is below the threshold "
                  f"({_TRUST_THRESHOLD_TO_SKIP_CONFIRMATION}) -- confirmation still required.",
    }