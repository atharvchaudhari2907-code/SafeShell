"""
tests/test_personalization_engine.py

Tests for the PersonalizationEngine and its helper functions.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from safeshell_member5.db.models import Base
from safeshell_member5.models.schemas import AIAnalysisResult, TrustScoreResult
from safeshell_member5.personalization.engine import (
    PersonalizationEngine,
    derive_friction_adjustment,
    normalize_pattern,
)


@pytest.fixture()
def db_session() -> Session:
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    factory = sessionmaker(bind=eng)
    session = factory()
    yield session
    session.close()


def test_normalize_pattern():
    assert normalize_pattern("rm -rf /tmp/data") == "rm -rf <path>"
    assert normalize_pattern("git status") == "git status"
    assert normalize_pattern("sudo apt-get install python3") == "sudo apt-get install python3"
    assert normalize_pattern("docker run -d nginx") == "docker run -d nginx"


def test_derive_friction_adjustment_critical():
    # CRITICAL RULE: Never reduce friction on critical risk
    score = TrustScoreResult(score=1.0, weight_breakdown={}, critical_override_active=True)
    assert derive_friction_adjustment(score, "critical") == "none"


def test_derive_friction_adjustment_high_trust():
    score = TrustScoreResult(score=0.9, weight_breakdown={}, critical_override_active=False)
    assert derive_friction_adjustment(score, "low") == "reduce"
    assert derive_friction_adjustment(score, "medium") == "reduce"
    assert derive_friction_adjustment(score, "high") == "none" # High risk shouldn't be reduced even with high trust


def test_derive_friction_adjustment_low_trust():
    score = TrustScoreResult(score=0.1, weight_breakdown={}, critical_override_active=False)
    assert derive_friction_adjustment(score, "low") == "increase"
    assert derive_friction_adjustment(score, "medium") == "increase"
    assert derive_friction_adjustment(score, "high") == "increase"


def test_personalization_engine(db_session):
    engine = PersonalizationEngine(db_session)
    
    analysis = AIAnalysisResult(
        command="rm -rf /tmp/data",
        normalized_context={"cwd": "/tmp"},
        intent="delete_files",
        risk_score=0.4,
        risk_level="medium",
        risk_signals=["recursive_delete"],
        confidence=0.9
    )
    
    # Cold start
    result = engine.personalize(analysis, "alice")
    
    assert result.ai_analysis == analysis
    assert result.trust_score.score == 0.0 # Cold start
    assert result.friction_adjustment == "increase" # Score <= 0.25 -> increase
    assert result.profile_id > 0
