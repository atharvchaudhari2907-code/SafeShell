"""
trust_score.py

Adaptive Trust Score engine (Member 5 spec).

Combines:
    Usage Frequency + Recency + Previous Decisions
    + Historical Risk + Context Consistency
    -> Trust Score -> Time Decay

The trust score can REDUCE FRICTION (e.g. skip a confirmation prompt
for a command a user has run constantly and safely), but it can NEVER
override a critical/BLOCK verdict -- that hard rule is enforced in
policy_engine.py, not here. This module only ever produces a score;
it never makes the final allow/block decision itself.

Storage is a simple JSON file per install -- no external DB needed
for this project. One entry per (user_id, command_signature) pair.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_STORE_PATH = Path(__file__).resolve().parents[2] / "data" / "trust_scores.json"

# Weights for each factor (must sum to 1.0)
_WEIGHTS = {
    "usage_frequency": 0.25,
    "recency": 0.15,
    "previous_decisions": 0.30,
    "historical_risk": 0.20,
    "context_consistency": 0.10,
}

# How fast trust decays with time since last use (half-life, in days).
_TIME_DECAY_HALF_LIFE_DAYS = 14.0

# risk_level string -> numeric risk, used for the historical_risk factor
_RISK_VALUE = {"low": 0.1, "medium": 0.4, "high": 0.75, "critical": 1.0}


def command_signature(command: str, flags, target_path: str) -> str:
    """Coarse signature grouping "commands like this one" -- same
    command + flags + target *shape* (not exact filename), so
    `rm -rf project_a/` and `rm -rf project_b/` build shared trust.
    """
    flag_sig = ",".join(sorted(flags))
    if target_path in ("/", "") or re.fullmatch(r"/(etc|boot|root|var|usr|bin|sbin|dev)(/.*)?", target_path or ""):
        shape = "SYSTEM_PATH"
    elif (target_path or "").startswith(("/home", "~")):
        shape = "HOME_PATH"
    elif "*" in (target_path or ""):
        shape = "WILDCARD"
    else:
        shape = "PATH"
    return f"{command}|{flag_sig}|{shape}"


class TrustScoreStore:
    """Loads/saves per-user, per-signature interaction history."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or _DEFAULT_STORE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def get_history(self, user_id: str, signature: str) -> Dict[str, Any]:
        user = self._data.get(user_id, {})
        return user.get(signature, {
            "times_seen": 0,
            "times_allowed": 0,
            "times_rejected_or_blocked": 0,
            "risk_values_seen": [],   # numeric risk values, for historical_risk avg
            "last_seen": None,
            "last_context": None,    # e.g. cwd, used for context_consistency
        })

    def record(
        self,
        user_id: str,
        signature: str,
        outcome: str,          # "ALLOWED" / "REJECTED" / "BLOCKED"
        risk_level: str,       # "low"/"medium"/"high"/"critical" from the real pipeline
        context: Optional[str] = None,
    ) -> None:
        user = self._data.setdefault(user_id, {})
        entry = user.setdefault(signature, {
            "times_seen": 0,
            "times_allowed": 0,
            "times_rejected_or_blocked": 0,
            "risk_values_seen": [],
            "last_seen": None,
            "last_context": None,
        })
        entry["times_seen"] += 1
        if outcome == "ALLOWED":
            entry["times_allowed"] += 1
        else:
            entry["times_rejected_or_blocked"] += 1
        entry["risk_values_seen"].append(_RISK_VALUE.get(risk_level, 0.5))
        entry["risk_values_seen"] = entry["risk_values_seen"][-20:]  # cap history size
        entry["last_seen"] = time.time()
        if context:
            entry["last_context"] = context
        self._save()


def _time_decay_factor(last_seen: Optional[float]) -> float:
    """Exponential decay: trust contribution halves every
    _TIME_DECAY_HALF_LIFE_DAYS days since last use. Returns 1.0 for
    "just now", approaching 0.0 for very old / never-seen."""
    if last_seen is None:
        return 0.0
    days_elapsed = (time.time() - last_seen) / 86400.0
    return math.pow(0.5, days_elapsed / _TIME_DECAY_HALF_LIFE_DAYS)


def compute_trust_score(history: Dict[str, Any], current_context: Optional[str] = None) -> float:
    """Compute a 0.0-1.0 trust score from interaction history.

    0.0 = no trust (never seen, or consistently rejected/high-risk)
    1.0 = fully trusted (frequent, recent, consistently allowed,
          historically low risk, consistent context)
    """
    times_seen = history.get("times_seen", 0)
    if times_seen == 0:
        return 0.0

    usage_frequency = min(times_seen / 20.0, 1.0)  # saturates at 20 uses

    recency = _time_decay_factor(history.get("last_seen"))

    times_allowed = history.get("times_allowed", 0)
    previous_decisions = times_allowed / times_seen

    risk_values = history.get("risk_values_seen", [])
    avg_risk = sum(risk_values) / len(risk_values) if risk_values else 1.0
    historical_risk = 1.0 - avg_risk  # low historical risk -> high trust contribution

    last_context = history.get("last_context")
    context_consistency = 1.0 if (current_context and current_context == last_context) else 0.5

    raw_score = (
        _WEIGHTS["usage_frequency"] * usage_frequency
        + _WEIGHTS["recency"] * recency
        + _WEIGHTS["previous_decisions"] * previous_decisions
        + _WEIGHTS["historical_risk"] * historical_risk
        + _WEIGHTS["context_consistency"] * context_consistency
    )

    # Time decay is applied twice deliberately: once inside `recency`
    # (rewards recent use) and once here as an overall dampener on
    # the whole score (an old profile shouldn't feel as confident as
    # a fresh one, even if its historical stats were once great).
    overall_decay = _time_decay_factor(history.get("last_seen"))
    return round(raw_score * (0.5 + 0.5 * overall_decay), 4)  # decay softens, never zeroes fully