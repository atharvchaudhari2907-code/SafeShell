"""
db/models.py

SQLAlchemy ORM models for SafeShell Member 5.

Tables
------
trust_profiles
    Per-(user, command_pattern) trust state including usage frequency,
    risk history, user decisions, context consistency, and cached trust score.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import JSON

Base = declarative_base()


class TrustProfileRecord(Base):  # type: ignore[misc]
    """Persistent per-user, per-command-pattern trust profile."""

    __tablename__ = "trust_profiles"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "command_pattern", name="uq_user_command_pattern"
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ──────────────────────────────────────────────────────
    user_id = Column(String, nullable=False, index=True)
    command_pattern = Column(
        String, nullable=False, index=True,
        doc="Normalised command signature, e.g. 'git status', 'rm -rf <path>'",
    )

    # ── Usage statistics ──────────────────────────────────────────────
    usage_frequency = Column(Integer, nullable=False, default=0)
    last_used_at = Column(DateTime, nullable=True)

    # ── Risk history ──────────────────────────────────────────────────
    historical_risk_avg = Column(
        Float, nullable=False, default=0.0,
        doc="Running average of risk_score across all recorded usages.",
    )

    # ── User decision history ─────────────────────────────────────────
    accept_count = Column(Integer, nullable=False, default=0)
    reject_count = Column(Integer, nullable=False, default=0)

    # ── Context consistency ───────────────────────────────────────────
    context_consistency_score = Column(
        Float, nullable=False, default=0.0,
        doc="Dominance ratio of the most-frequent context hash (0–1).",
    )
    context_hashes = Column(
        JSON, nullable=False, default=list,
        doc="Sliding window of recent context hashes (max CONTEXT_HASH_WINDOW).",
    )

    # ── Cached trust score ────────────────────────────────────────────
    current_trust_score = Column(
        Float, nullable=False, default=0.0,
        doc="Materialised trust score; recomputed on every record_usage().",
    )

    # ── Timestamps ────────────────────────────────────────────────────
    created_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<TrustProfileRecord user={self.user_id!r} "
            f"pattern={self.command_pattern!r} "
            f"trust={self.current_trust_score:.4f}>"
        )


class AuditLogRecord(Base):  # type: ignore[misc]
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Input
    raw_command = Column(String, nullable=False)
    ast_json = Column(String, nullable=False)  # JSON
    
    # Analysis
    intent = Column(String, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    risk_signals = Column(String, nullable=False)  # JSON list
    normalized_context_json = Column(String, nullable=False)  # JSON
    
    # Trust
    trust_score = Column(Float, nullable=False)
    trust_breakdown_json = Column(String, nullable=False)  # JSON
    
    # Decision
    decision_action = Column(String, nullable=False)
    decision_reason = Column(String, nullable=False)
    
    # Advisory (Nullable)
    advisory_explanation = Column(String, nullable=True)
    advisory_alternative = Column(String, nullable=True)
    
    # User Response
    user_response = Column(String, nullable=False)
    
    # Execution
    execution_result_json = Column(String, nullable=True)  # JSON
    rollback_triggered = Column(Boolean, default=False, nullable=False)
