"""
tests/test_decision_engine.py

Tests for the Policy Decision Engine.
Covers all combinations of risk_level x trust_score.
"""

from __future__ import annotations

import pytest

from safeshell_member5.models.schemas import AIAnalysisResult, PersonalizedResult, TrustScoreResult
from safeshell_member5.policy.decision_engine import decide


def _make_personalized_result(risk_level: str, trust_score: float) -> PersonalizedResult:
    """Helper to mock a PersonalizedResult for decision testing."""
    # Critical risk forces trust score to 0.0 in real usage, but we test
    # adversarial inputs here (trust=1.0) to prove the safety floor.
    is_critical_override = (risk_level == "critical" and trust_score == 0.0)
    
    analysis = AIAnalysisResult(
        command="test",
        normalized_context={},
        intent="test",
        risk_score=0.5,
        risk_level=risk_level, # type: ignore
        risk_signals=[],
        confidence=0.9
    )
    
    t_score = TrustScoreResult(
        score=trust_score,
        weight_breakdown={},
        critical_override_active=is_critical_override
    )
    
    return PersonalizedResult(
        ai_analysis=analysis,
        trust_score=t_score,
        profile_id=1,
        friction_adjustment="none"
    )


# ── Critical Risk ────────────────────────────────────────────────────────

def test_decide_critical_low_trust():
    res = _make_personalized_result("critical", 0.0)
    decision = decide(res)
    assert decision.action == "BLOCK"
    assert decision.override_possible is False

def test_decide_critical_adversarial_high_trust():
    """Adversarial test: even if trust is maxed out, critical must BLOCK."""
    res = _make_personalized_result("critical", 1.0)
    decision = decide(res)
    assert decision.action == "BLOCK"
    assert decision.override_possible is False


# ── High Risk ────────────────────────────────────────────────────────────

def test_decide_high_risk_low_trust():
    res = _make_personalized_result("high", 0.2)
    decision = decide(res)
    assert decision.action == "WARN"
    assert decision.override_possible is True

def test_decide_high_risk_high_trust():
    """High risk remains WARN even with high trust."""
    res = _make_personalized_result("high", 0.9)
    decision = decide(res)
    assert decision.action == "WARN"
    assert decision.override_possible is True


# ── Medium Risk ──────────────────────────────────────────────────────────

def test_decide_medium_risk_low_trust():
    res = _make_personalized_result("medium", 0.2)
    decision = decide(res)
    assert decision.action == "WARN"
    assert decision.override_possible is True

def test_decide_medium_risk_borderline_trust():
    res = _make_personalized_result("medium", 0.7)
    decision = decide(res)
    assert decision.action == "WARN"
    assert decision.override_possible is True

def test_decide_medium_risk_high_trust():
    """Trust > 0.7 reduces friction for medium risk."""
    res = _make_personalized_result("medium", 0.75)
    decision = decide(res)
    assert decision.action == "ALLOW"
    assert "high trust" in decision.reason.lower()


# ── Low Risk ─────────────────────────────────────────────────────────────

def test_decide_low_risk_low_trust():
    res = _make_personalized_result("low", 0.0)
    decision = decide(res)
    assert decision.action == "ALLOW"

def test_decide_low_risk_high_trust():
    res = _make_personalized_result("low", 1.0)
    decision = decide(res)
    assert decision.action == "ALLOW"



