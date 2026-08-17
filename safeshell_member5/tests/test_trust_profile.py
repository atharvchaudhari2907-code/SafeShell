"""
tests/test_trust_profile.py

Unit tests for personalization.trust_profile.UserTrustProfile.

All tests use an in-memory SQLite database for isolation.

Test scenarios
--------------
1. New user, first command → cold start (trust = 0.0, get_profile → None)
2. Repeated safe command  → trust score increases monotonically
3. Growing usage gap      → inactivity decay reduces trust toward zero
4. Reject decisions       → lower approval ratio, lower trust
5. Mixed context hashes   → consistency score drops
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from safeshell_member5.db.models import Base, TrustProfileRecord
from safeshell_member5.personalization.trust_profile import UserTrustProfile


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def engine():
    """Create a fresh in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def db_session(engine):
    """Yield a session bound to the in-memory engine; rolled back after use."""
    factory = sessionmaker(bind=engine)
    session: Session = factory()
    yield session
    session.close()


@pytest.fixture()
def manager(db_session) -> UserTrustProfile:
    """Return a UserTrustProfile manager bound to the test session."""
    return UserTrustProfile(db_session)


# ── 1. Cold start ────────────────────────────────────────────────────


class TestColdStart:
    """A previously-unseen (user, pattern) pair starts with zero trust."""

    def test_get_profile_returns_none_for_unknown(self, manager):
        """get_profile() must return None when no profile exists."""
        assert manager.get_profile("alice", "git status") is None

    def test_get_or_create_returns_zero_trust(self, manager):
        """get_or_create() must create a record with trust_score = 0.0."""
        record = manager.get_or_create("alice", "git status")
        assert record.current_trust_score == 0.0
        assert record.usage_frequency == 0
        assert record.accept_count == 0
        assert record.reject_count == 0
        assert record.historical_risk_avg == 0.0
        assert record.context_hashes == []

    def test_get_or_create_is_idempotent(self, manager, db_session):
        """Calling get_or_create() twice returns the same row."""
        r1 = manager.get_or_create("alice", "git status")
        r2 = manager.get_or_create("alice", "git status")
        assert r1.id == r2.id

    def test_first_record_usage_from_cold(self, manager):
        """First record_usage on an unknown pair bootstraps the profile."""
        record = manager.record_usage(
            "bob", "ls -la", risk_score=0.05, user_decision="accept",
            context_hash="abc123",
        )
        assert record.usage_frequency == 1
        assert record.accept_count == 1
        assert record.current_trust_score > 0.0


# ── 2. Repeated safe command ─────────────────────────────────────────


class TestRepeatedSafeCommand:
    """Trust should grow when the same low-risk command is accepted."""

    def test_trust_increases_with_repeated_safe_usage(self, manager):
        """Ten consecutive safe accepts should steadily raise trust."""
        scores: list[float] = []
        for _ in range(10):
            record = manager.record_usage(
                "carol", "git status",
                risk_score=0.05, user_decision="accept",
                context_hash="ctx_a",
            )
            scores.append(record.current_trust_score)

        # Every subsequent score must be ≥ the previous one.
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], (
                f"Trust must not decrease: step {i} "
                f"({scores[i]}) < step {i-1} ({scores[i-1]})"
            )

        # After 10 safe uses the score should be meaningfully above zero.
        assert scores[-1] > 0.3

    def test_risk_avg_stays_low_for_safe_commands(self, manager):
        """historical_risk_avg should converge to the per-use risk_score."""
        for _ in range(20):
            record = manager.record_usage(
                "carol", "echo hello",
                risk_score=0.02, user_decision="accept",
                context_hash="ctx_b",
            )
        assert abs(record.historical_risk_avg - 0.02) < 0.001

    def test_context_consistency_perfect_when_same_hash(self, manager):
        """If every invocation uses the same context hash, consistency = 1.0."""
        for _ in range(5):
            record = manager.record_usage(
                "carol", "pwd",
                risk_score=0.01, user_decision="accept",
                context_hash="same",
            )
        assert record.context_consistency_score == 1.0


# ── 3. Inactivity decay ─────────────────────────────────────────────


class TestInactivityDecay:
    """Trust should decay toward zero as time without usage grows."""

    def _build_trust(self, manager, n: int = 15) -> TrustProfileRecord:
        """Helper: record *n* safe usages to build a baseline trust."""
        record = None
        for _ in range(n):
            record = manager.record_usage(
                "dave", "git pull",
                risk_score=0.05, user_decision="accept",
                context_hash="ctx_x",
            )
        assert record is not None
        return record

    def test_partial_decay_after_half_window(self, manager):
        """Trust at 45 days inactive should be roughly half the baseline."""
        self._build_trust(manager)
        baseline = manager.get_profile("dave", "git pull")
        assert baseline is not None
        score_before = baseline.current_trust_score

        # Simulate 45 days into the future (half of the 90-day window).
        future = datetime.now(timezone.utc) + timedelta(days=45)
        manager.refresh_trust_score("dave", "git pull", now=future)

        baseline = manager.get_profile("dave", "git pull")
        score_after = baseline.current_trust_score

        assert score_after < score_before
        assert score_after == pytest.approx(score_before * 0.5, abs=0.01)

    def test_full_decay_after_90_days(self, manager):
        """Trust must reach 0.0 after 90+ days of inactivity."""
        self._build_trust(manager)

        far_future = datetime.now(timezone.utc) + timedelta(days=91)
        manager.refresh_trust_score("dave", "git pull", now=far_future)

        record = manager.get_profile("dave", "git pull")
        assert record is not None
        assert record.current_trust_score == 0.0

    def test_usage_resets_decay(self, manager):
        """A new record_usage after a gap should restore trust (re-anchor)."""
        self._build_trust(manager, n=10)
        baseline = manager.get_profile("dave", "git pull")
        score_fresh = baseline.current_trust_score

        # Simulate 60 days of inactivity.
        future = datetime.now(timezone.utc) + timedelta(days=60)
        manager.refresh_trust_score("dave", "git pull", now=future)
        record_decayed = manager.get_profile("dave", "git pull")
        score_decayed = record_decayed.current_trust_score
        assert score_decayed < score_fresh

        # A new usage re-anchors last_used_at to "now", resetting decay.
        record_restored = manager.record_usage(
            "dave", "git pull",
            risk_score=0.05, user_decision="accept",
            context_hash="ctx_x",
        )
        assert record_restored.current_trust_score > score_decayed


# ── 4. Reject decisions ──────────────────────────────────────────────


class TestRejectDecisions:
    """Rejecting commands should depress the approval-ratio component."""

    def test_reject_lowers_trust_vs_all_accept(self, manager):
        """A profile with 50 % rejects should have lower trust than 100 % accepts."""
        # All-accept user
        for _ in range(10):
            manager.record_usage(
                "eve", "make deploy",
                risk_score=0.3, user_decision="accept",
                context_hash="ctx_1",
            )
        accept_score = manager.get_profile("eve", "make deploy").current_trust_score

        # 50 / 50 user
        for i in range(10):
            decision = "accept" if i % 2 == 0 else "reject"
            manager.record_usage(
                "frank", "make deploy",
                risk_score=0.3, user_decision=decision,
                context_hash="ctx_1",
            )
        mixed_score = manager.get_profile("frank", "make deploy").current_trust_score

        assert mixed_score < accept_score


# ── 5. Context consistency ───────────────────────────────────────────


class TestContextConsistency:
    """Varying execution contexts should reduce the consistency component."""

    def test_mixed_contexts_lower_consistency(self, manager):
        """Using many different context hashes should drop consistency."""
        for i in range(10):
            manager.record_usage(
                "grace", "docker build",
                risk_score=0.2, user_decision="accept",
                context_hash=f"unique_{i}",
            )
        record = manager.get_profile("grace", "docker build")
        # Each hash is unique → consistency = 1/10 = 0.1
        assert record.context_consistency_score == pytest.approx(0.1)

    def test_dominant_context_raises_consistency(self, manager):
        """When most hashes are the same, consistency should be high."""
        for _ in range(8):
            manager.record_usage(
                "heidi", "npm test",
                risk_score=0.1, user_decision="accept",
                context_hash="dominant",
            )
        for _ in range(2):
            manager.record_usage(
                "heidi", "npm test",
                risk_score=0.1, user_decision="accept",
                context_hash="outlier",
            )
        record = manager.get_profile("heidi", "npm test")
        # 8/10 = 0.8
        assert record.context_consistency_score == pytest.approx(0.8)
