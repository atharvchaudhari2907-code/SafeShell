"""
personalization/trust_score.py

Adaptive trust-score calculator for SafeShell.

Implements the architecture-doc formula::

    Usage Frequency  (log-scaled, capped at 1.0)
  + Recency          (exponential decay, 14-day half-life)
  + Previous Decisions (accept / (accept + reject + 1))
  + Historical Risk  (1 − historical_risk_avg)
  + Context Consistency (cosine-sim proxy on context features)
  → weighted sum → exponential time decay → Trust Score

Weights::

    0.25 × frequency
    0.20 × recency
    0.25 × decision
    0.20 × risk_inverse
    0.10 × consistency

CRITICAL RULE (hardcoded, no override path)
-------------------------------------------
If the *current* AI ``risk_level`` is ``"critical"``, the returned
``TrustScoreResult.score`` is forced to **0.0**.  Trust may reduce
friction on low / medium risk commands, but it **never** overrides a
critical risk assessment.  The ``weight_breakdown`` still records the
true computed components so that the audit log can inspect them.

Public API
----------
compute_trust_score(profile, current_context, ...) → TrustScoreResult
apply_time_decay(score, last_used_at, ...)          → float
hash_context(context)                               → str
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any

from safeshell_member5.db.models import TrustProfileRecord
from safeshell_member5.models.schemas import TrustScoreResult

# ── Defaults ──────────────────────────────────────────────────────────
# These are deliberately module-level constants so they can be referenced
# in tests without importing config (the weights live here because *this*
# module owns the formula; config owns profile-storage parameters).

W_FREQ: float = 0.25
W_RECENCY: float = 0.20
W_DECISION: float = 0.25
W_RISK_INV: float = 0.20
W_CONSISTENCY: float = 0.10

# The frequency value at which log-scaling saturates to 1.0.
_FREQ_SATURATION: int = 100

# Default half-life (days) for exponential recency and time decay.
DEFAULT_HALF_LIFE_DAYS: float = 14.0


# =====================================================================
# Public helpers
# =====================================================================

def hash_context(context: dict[str, Any]) -> str:
    """Produce a deterministic hash string from a context dictionary.

    Use this when calling ``record_usage()`` so that the stored hashes
    are consistent with the hashes computed by ``_context_similarity()``.
    """
    serialised = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:16]


# =====================================================================
# Core public function
# =====================================================================

def compute_trust_score(
    profile: TrustProfileRecord,
    current_context: dict[str, Any],
    current_risk_level: str = "low",
    *,
    now: datetime | None = None,
) -> TrustScoreResult:
    """Compute the adaptive trust score for a (user, command_pattern) pair.

    Parameters
    ----------
    profile : TrustProfileRecord
        The persisted trust profile for this (user, pattern).
    current_context : dict
        The ``normalized_context`` from the current ``AIAnalysisResult``.
    current_risk_level : str
        The AI-assigned risk level for the *current* command.
        If ``"critical"`` the returned score is forced to **0.0**.
    now : datetime, optional
        Override "current time" (for deterministic testing).

    Returns
    -------
    TrustScoreResult
        Contains ``score``, ``weight_breakdown``, and
        ``critical_override_active``.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # ── Individual components ─────────────────────────────────────────
    freq_score = _normalize_frequency(profile.usage_frequency)
    recency_score = _recency_decay(profile.last_used_at, now=now)
    decision_score = (
        profile.accept_count
        / (profile.accept_count + profile.reject_count + 1)
    )
    risk_score_inv = 1.0 - profile.historical_risk_avg
    consistency_score = _context_similarity(current_context, profile)

    # ── Weighted sum ──────────────────────────────────────────────────
    raw = (
        W_FREQ * freq_score
        + W_RECENCY * recency_score
        + W_DECISION * decision_score
        + W_RISK_INV * risk_score_inv
        + W_CONSISTENCY * consistency_score
    )

    # ── Exponential time decay ────────────────────────────────────────
    decayed = apply_time_decay(raw, profile.last_used_at, now=now)

    # ── Breakdown (always records the *actual* computation) ───────────
    breakdown: dict[str, float] = {
        "frequency": round(freq_score, 6),
        "recency": round(recency_score, 6),
        "decision": round(decision_score, 6),
        "risk_inverse": round(risk_score_inv, 6),
        "consistency": round(consistency_score, 6),
        "raw_weighted": round(raw, 6),
        "time_decayed": round(decayed, 6),
    }

    # ── CRITICAL RULE ─────────────────────────────────────────────────
    #  Trust NEVER reduces friction when the current risk is critical.
    #  Hardcoded — no config knob, no override path.
    critical_override = current_risk_level == "critical"
    effective_score = 0.0 if critical_override else decayed

    return TrustScoreResult(
        score=round(effective_score, 6),
        weight_breakdown=breakdown,
        critical_override_active=critical_override,
    )


# =====================================================================
# Component functions
# =====================================================================

def _normalize_frequency(usage_frequency: int) -> float:
    """Log-scaled frequency, capped at 1.0.

    ``log₂(1 + n) / log₂(1 + SATURATION)``

    Maps 0 → 0.0, ~15 → 0.58, ~50 → 0.84, 100 → 1.0, 200 → 1.0.
    """
    if usage_frequency <= 0:
        return 0.0
    return min(
        math.log2(1 + usage_frequency) / math.log2(1 + _FREQ_SATURATION),
        1.0,
    )


def _recency_decay(
    last_used_at: datetime | None,
    *,
    now: datetime | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Exponential recency score: 1.0 when just used, halves every *half_life_days*.

    Returns 0.0 when *last_used_at* is ``None`` (cold start).
    """
    if last_used_at is None:
        return 0.0
    if now is None:
        now = datetime.now(timezone.utc)
    last = _ensure_utc(last_used_at)
    days_elapsed = max(0.0, (now - last).total_seconds() / 86400.0)
    return 0.5 ** (days_elapsed / half_life_days)


def apply_time_decay(
    score: float,
    last_used_at: datetime | None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    *,
    now: datetime | None = None,
) -> float:
    """Exponential time decay: ``score × 0.5 ^ (days_elapsed / half_life)``.

    Returns 0.0 when *last_used_at* is ``None`` (cold start — nothing
    to decay from).
    """
    if last_used_at is None:
        return 0.0
    if now is None:
        now = datetime.now(timezone.utc)
    last = _ensure_utc(last_used_at)
    days_elapsed = max(0.0, (now - last).total_seconds() / 86400.0)
    return score * (0.5 ** (days_elapsed / half_life_days))


def _context_similarity(
    current_context: dict[str, Any],
    profile: TrustProfileRecord,
) -> float:
    """Similarity between the current context and the profile's history.

    Computes the match ratio of the current context hash against the
    sliding window of stored context hashes.  This is a cosine-similarity
    proxy for binary feature vectors (each unique hash is a dimension;
    presence is 1, absence is 0).

    Returns 0.0 when the profile has no stored hashes (cold start).
    """
    stored: list[str] = profile.context_hashes or []
    if not stored:
        return 0.0
    current_hash = hash_context(current_context)
    matches = sum(1 for h in stored if h == current_hash)
    return matches / len(stored)


# =====================================================================
# Internal utilities
# =====================================================================

def _ensure_utc(dt: datetime) -> datetime:
    """Attach UTC tzinfo if the datetime is naïve (common with SQLite)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
