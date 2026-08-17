"""
tests/test_trust_score.py

Unit tests for personalization.trust_score — the adaptive trust-score
calculator.

Test groups
-----------
1. Frequency normalisation (log-scaled, capped)
2. Recency decay (exponential, 14-day half-life)
3. Time decay (applied to raw weighted score)
4. Context similarity (hash-based matching)
5. hash_context determinism
6. Critical-override rule (MUST force score to 0.0)
7. End-to-end compute_trust_score (cold start, warm profile, weight sanity)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from safeshell_member5.db.models import TrustProfileRecord
from safeshell_member5.personalization.trust_score import (
    DEFAULT_HALF_LIFE_DAYS,
    W_CONSISTENCY,
    W_DECISION,
    W_FREQ,
    W_RECENCY,
    W_RISK_INV,
    _context_similarity,
    _normalize_frequency,
    _recency_decay,
    apply_time_decay,
    compute_trust_score,
    hash_context,
)


# ── Helpers ───────────────────────────────────────────────────────────

_NOW = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)


def _make_profile(
    *,
    usage_frequency: int = 0,
    last_used_at: datetime | None = None,
    historical_risk_avg: float = 0.0,
    accept_count: int = 0,
    reject_count: int = 0,
    context_consistency_score: float = 0.0,
    current_trust_score: float = 0.0,
    context_hashes: list[str] | None = None,
) -> TrustProfileRecord:
    """Build an in-memory TrustProfileRecord (no DB session needed)."""
    return TrustProfileRecord(
        user_id="test_user",
        command_pattern="test cmd",
        usage_frequency=usage_frequency,
        last_used_at=last_used_at,
        historical_risk_avg=historical_risk_avg,
        accept_count=accept_count,
        reject_count=reject_count,
        context_consistency_score=context_consistency_score,
        current_trust_score=current_trust_score,
        context_hashes=context_hashes if context_hashes is not None else [],
    )


# =====================================================================
# 1. Frequency normalisation
# =====================================================================


class TestNormalizeFrequency:
    """Log-scaled, capped at 1.0."""

    def test_zero_uses(self):
        assert _normalize_frequency(0) == 0.0

    def test_negative_uses(self):
        assert _normalize_frequency(-5) == 0.0

    def test_one_use_is_small_positive(self):
        score = _normalize_frequency(1)
        assert 0.0 < score < 0.2

    def test_monotonically_increasing(self):
        """More uses → higher (or equal) score."""
        prev = 0.0
        for n in [1, 2, 5, 10, 25, 50, 75, 100]:
            score = _normalize_frequency(n)
            assert score >= prev, f"n={n}: {score} < {prev}"
            prev = score

    def test_saturates_at_100(self):
        assert _normalize_frequency(100) == pytest.approx(1.0)

    def test_capped_above_saturation(self):
        assert _normalize_frequency(500) == 1.0


# =====================================================================
# 2. Recency decay
# =====================================================================


class TestRecencyDecay:
    """Exponential decay with a 14-day half-life."""

    def test_none_last_used(self):
        """Cold start → 0.0."""
        assert _recency_decay(None, now=_NOW) == 0.0

    def test_just_used(self):
        """0 days elapsed → ≈ 1.0."""
        score = _recency_decay(_NOW, now=_NOW)
        assert score == pytest.approx(1.0)

    def test_half_life(self):
        """Exactly one half-life → ≈ 0.5."""
        past = _NOW - timedelta(days=14)
        score = _recency_decay(past, now=_NOW)
        assert score == pytest.approx(0.5, abs=1e-6)

    def test_two_half_lives(self):
        """28 days → ≈ 0.25."""
        past = _NOW - timedelta(days=28)
        score = _recency_decay(past, now=_NOW)
        assert score == pytest.approx(0.25, abs=1e-6)

    def test_long_inactivity_approaches_zero(self):
        """140 days (10 half-lives) → ≈ 0.001."""
        past = _NOW - timedelta(days=140)
        score = _recency_decay(past, now=_NOW)
        assert score < 0.002

    def test_naive_datetime_treated_as_utc(self):
        """Timezone-naïve last_used_at should not raise."""
        naive = _NOW.replace(tzinfo=None)
        score = _recency_decay(naive, now=_NOW)
        assert score == pytest.approx(1.0)


# =====================================================================
# 3. Time decay
# =====================================================================


class TestApplyTimeDecay:
    """``score × 0.5 ^ (days / half_life)``."""

    def test_zero_elapsed(self):
        result = apply_time_decay(0.8, _NOW, now=_NOW)
        assert result == pytest.approx(0.8)

    def test_one_half_life(self):
        past = _NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS)
        result = apply_time_decay(1.0, past, now=_NOW)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_none_last_used_returns_zero(self):
        assert apply_time_decay(0.9, None, now=_NOW) == 0.0

    def test_custom_half_life(self):
        past = _NOW - timedelta(days=7)
        result = apply_time_decay(1.0, past, half_life_days=7.0, now=_NOW)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_preserves_zero_score(self):
        past = _NOW - timedelta(days=1)
        assert apply_time_decay(0.0, past, now=_NOW) == 0.0


# =====================================================================
# 4. Context similarity
# =====================================================================


class TestContextSimilarity:
    """Hash-based matching against the stored sliding window."""

    def test_no_stored_hashes_returns_zero(self):
        profile = _make_profile(context_hashes=[])
        assert _context_similarity({"a": 1}, profile) == 0.0

    def test_all_matching(self):
        ctx = {"working_dir": "/home/user"}
        h = hash_context(ctx)
        profile = _make_profile(context_hashes=[h] * 10)
        assert _context_similarity(ctx, profile) == pytest.approx(1.0)

    def test_half_matching(self):
        ctx = {"working_dir": "/home/user"}
        h = hash_context(ctx)
        profile = _make_profile(context_hashes=[h] * 5 + ["other"] * 5)
        assert _context_similarity(ctx, profile) == pytest.approx(0.5)

    def test_none_matching(self):
        ctx = {"working_dir": "/home/user"}
        profile = _make_profile(context_hashes=["x", "y", "z"])
        assert _context_similarity(ctx, profile) == pytest.approx(0.0)


# =====================================================================
# 5. hash_context
# =====================================================================


class TestHashContext:
    """Deterministic, order-independent context hashing."""

    def test_deterministic(self):
        ctx = {"b": 2, "a": 1}
        assert hash_context(ctx) == hash_context(ctx)

    def test_key_order_irrelevant(self):
        assert hash_context({"a": 1, "b": 2}) == hash_context({"b": 2, "a": 1})

    def test_different_values_different_hash(self):
        assert hash_context({"a": 1}) != hash_context({"a": 2})

    def test_returns_string_of_expected_length(self):
        h = hash_context({"x": "y"})
        assert isinstance(h, str)
        assert len(h) == 16


# =====================================================================
# 6. CRITICAL OVERRIDE RULE
# =====================================================================


class TestCriticalOverride:
    """
    CRITICAL RULE: trust_score NEVER reduces risk_level when
    current AI risk_level == 'critical'.

    Even the *maximum possible* trust score (built from a long,
    perfect usage history) must produce score == 0.0 when the
    current command is critical.
    """

    @pytest.fixture()
    def max_trust_profile(self) -> TrustProfileRecord:
        """A profile with the highest achievable trust across all components."""
        ctx_hash = hash_context({"env": "production"})
        return _make_profile(
            usage_frequency=200,               # saturated frequency
            last_used_at=_NOW,                 # just used — no decay
            historical_risk_avg=0.01,          # near-zero historical risk
            accept_count=199,                  # near-100 % approval
            reject_count=1,
            context_consistency_score=0.99,
            current_trust_score=0.99,
            context_hashes=[ctx_hash] * 20,    # perfect consistency
        )

    def test_critical_forces_score_to_zero(self, max_trust_profile):
        """Feed trust_score ≈ 1.0 + risk_level='critical' → score == 0.0."""
        result = compute_trust_score(
            max_trust_profile,
            {"env": "production"},
            current_risk_level="critical",
            now=_NOW,
        )
        assert result.score == 0.0

    def test_critical_override_flag_is_set(self, max_trust_profile):
        result = compute_trust_score(
            max_trust_profile,
            {"env": "production"},
            current_risk_level="critical",
            now=_NOW,
        )
        assert result.critical_override_active is True

    def test_breakdown_still_shows_real_values(self, max_trust_profile):
        """Audit log must see the actual computation, not the overridden zero."""
        result = compute_trust_score(
            max_trust_profile,
            {"env": "production"},
            current_risk_level="critical",
            now=_NOW,
        )
        # The breakdown should contain the true computed values.
        assert result.weight_breakdown["frequency"] > 0.9
        assert result.weight_breakdown["recency"] > 0.9
        assert result.weight_breakdown["decision"] > 0.9
        assert result.weight_breakdown["risk_inverse"] > 0.9
        assert result.weight_breakdown["raw_weighted"] > 0.5
        assert result.weight_breakdown["time_decayed"] > 0.5
        # But the effective score is still zero.
        assert result.score == 0.0

    def test_high_risk_is_not_overridden(self, max_trust_profile):
        """risk_level='high' should produce a real (non-zero) score."""
        result = compute_trust_score(
            max_trust_profile,
            {"env": "production"},
            current_risk_level="high",
            now=_NOW,
        )
        assert result.score > 0.0
        assert result.critical_override_active is False

    def test_low_risk_is_not_overridden(self, max_trust_profile):
        result = compute_trust_score(
            max_trust_profile,
            {"env": "production"},
            current_risk_level="low",
            now=_NOW,
        )
        assert result.score > 0.0
        assert result.critical_override_active is False

    def test_medium_risk_is_not_overridden(self, max_trust_profile):
        result = compute_trust_score(
            max_trust_profile,
            {"env": "production"},
            current_risk_level="medium",
            now=_NOW,
        )
        assert result.score > 0.0
        assert result.critical_override_active is False


# =====================================================================
# 7. End-to-end compute_trust_score
# =====================================================================


class TestComputeTrustScore:
    """Integration tests for the full score-computation pipeline."""

    def test_cold_start_returns_zero(self):
        """Brand-new profile with no history → score ≈ 0.0."""
        profile = _make_profile()  # all defaults
        result = compute_trust_score(profile, {"a": 1}, now=_NOW)
        assert result.score == 0.0

    def test_warm_profile_returns_positive(self):
        """A profile with solid history should have a meaningful score."""
        ctx = {"dir": "/home/user"}
        h = hash_context(ctx)
        profile = _make_profile(
            usage_frequency=30,
            last_used_at=_NOW,
            historical_risk_avg=0.1,
            accept_count=25,
            reject_count=5,
            context_hashes=[h] * 15,
        )
        result = compute_trust_score(profile, ctx, now=_NOW)
        assert result.score > 0.3

    def test_weight_breakdown_has_all_keys(self):
        profile = _make_profile(usage_frequency=5, last_used_at=_NOW)
        result = compute_trust_score(profile, {}, now=_NOW)
        expected_keys = {
            "frequency", "recency", "decision",
            "risk_inverse", "consistency",
            "raw_weighted", "time_decayed",
        }
        assert set(result.weight_breakdown.keys()) == expected_keys

    def test_weights_sum_to_one(self):
        """The five component weights must sum to 1.0."""
        assert W_FREQ + W_RECENCY + W_DECISION + W_RISK_INV + W_CONSISTENCY == pytest.approx(1.0)

    def test_score_decays_with_time(self):
        """Same profile evaluated at a later 'now' should produce lower score."""
        ctx = {"x": 1}
        h = hash_context(ctx)
        profile = _make_profile(
            usage_frequency=20,
            last_used_at=_NOW,
            historical_risk_avg=0.1,
            accept_count=18,
            reject_count=2,
            context_hashes=[h] * 10,
        )
        score_now = compute_trust_score(profile, ctx, now=_NOW).score
        score_later = compute_trust_score(
            profile, ctx, now=_NOW + timedelta(days=14),
        ).score
        # At 14 days later:
        # 1. The recency component (weight 0.20) halves, subtracting 0.10 from raw score
        # 2. The final time decay halves the entire remaining raw score
        expected_later = (score_now - 0.10) * 0.5
        assert score_later == pytest.approx(expected_later, abs=1e-6)

    def test_higher_risk_history_lowers_score(self):
        """Two identical profiles differing only in risk_avg — higher risk → lower score."""
        ctx = {"x": 1}
        h = hash_context(ctx)

        base = dict(
            usage_frequency=30,
            last_used_at=_NOW,
            accept_count=28,
            reject_count=2,
            context_hashes=[h] * 10,
        )
        safe = _make_profile(historical_risk_avg=0.05, **base)
        risky = _make_profile(historical_risk_avg=0.8, **base)

        safe_score = compute_trust_score(safe, ctx, now=_NOW).score
        risky_score = compute_trust_score(risky, ctx, now=_NOW).score
        assert safe_score > risky_score
