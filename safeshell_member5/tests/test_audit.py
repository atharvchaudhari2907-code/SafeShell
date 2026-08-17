"""
tests/test_audit.py

Tests for the Audit Logger and Trust Profile Updater.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from safeshell_member5.audit.logger import AuditLogger
from safeshell_member5.audit.trust_updater import TrustProfileUpdater
from safeshell_member5.db.models import AuditLogRecord, Base
from safeshell_member5.models.schemas import (
    Advisory,
    AIAnalysisResult,
    PersonalizedResult,
    PolicyDecision,
    TrustScoreResult,
)


@pytest.fixture()
def db_session() -> Session:
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    factory = sessionmaker(bind=eng)
    session = factory()
    yield session
    session.close()


@pytest.fixture
def mock_analysis() -> AIAnalysisResult:
    return AIAnalysisResult(
        command="rm -rf /tmp/data",
        normalized_context={"cwd": "/tmp"},
        intent="recursive_delete",
        risk_score=0.6,
        risk_level="medium",
        risk_signals=["recursive"],
        confidence=0.9
    )


@pytest.fixture
def mock_personalized(mock_analysis) -> PersonalizedResult:
    t_score = TrustScoreResult(
        score=0.8,
        weight_breakdown={"frequency": 0.5},
        critical_override_active=False
    )
    return PersonalizedResult(
        ai_analysis=mock_analysis,
        trust_score=t_score,
        profile_id=1,
        friction_adjustment="reduce"
    )


@pytest.fixture
def mock_decision() -> PolicyDecision:
    return PolicyDecision(
        action="ALLOW",
        reason="high trust",
        override_possible=True,
        risk_level="medium",
        trust_score=0.8,
        timestamp="2026-08-17T12:00:00Z"
    )


@pytest.fixture
def mock_advisory() -> Advisory:
    return Advisory(explanation="Test exp", alternative="Test alt")


def test_audit_logger(db_session, mock_analysis, mock_personalized, mock_decision, mock_advisory):
    logger = AuditLogger(db_session)
    
    execution_result = {"exit_code": 0, "stdout": "ok"}
    user_response = "accepted"
    
    record = logger.log_event(
        user_id="alice",
        analysis=mock_analysis,
        personalized=mock_personalized,
        decision=mock_decision,
        advisory=mock_advisory,
        execution_result=execution_result,
        user_response=user_response
    )
    
    assert record.id is not None
    assert record.user_id == "alice"
    assert record.raw_command == "rm -rf /tmp/data"
    assert json.loads(record.execution_result_json) == execution_result
    
    history = logger.get_history("alice", limit=10)
    assert len(history) == 1
    assert history[0].id == record.id


def test_trust_updater_closes_loop(db_session, mock_analysis, mock_personalized, mock_decision, mock_advisory):
    logger = AuditLogger(db_session)
    updater = TrustProfileUpdater(db_session)
    
    record = logger.log_event(
        user_id="bob",
        analysis=mock_analysis,
        personalized=mock_personalized,
        decision=mock_decision,
        advisory=None,
        execution_result=None,
        user_response="accepted"
    )
    
    # Run the updater
    updater.update_trust(record)
    
    # Check that a trust profile was created for 'bob' and 'rm -rf <path>'
    from safeshell_member5.personalization.trust_profile import UserTrustProfile
    profile_manager = UserTrustProfile(db_session)
    profile = profile_manager.get_profile("bob", "rm -rf <path>")
    
    assert profile is not None
    assert profile.usage_frequency == 1
    assert profile.accept_count == 1
