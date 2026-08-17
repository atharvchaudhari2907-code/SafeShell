"""
audit/logger.py

Audit Logger for SafeShell.
Writes the full lifecycle record of a command execution to the SQLite audit_log table.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from safeshell_member5.db.models import AuditLogRecord
from safeshell_member5.models.schemas import Advisory, AIAnalysisResult, PersonalizedResult, PolicyDecision


class AuditLogger:
    def __init__(self, session: Session) -> None:
        self.session = session

    def log_event(
        self,
        user_id: str,
        analysis: AIAnalysisResult,
        personalized: PersonalizedResult,
        decision: PolicyDecision,
        advisory: Advisory | None,
        execution_result: dict[str, Any] | None,
        user_response: str
    ) -> AuditLogRecord:
        """Single write, append-only logging of a command lifecycle."""
        
        record = AuditLogRecord(
            user_id=user_id,
            timestamp=datetime.now(timezone.utc),
            raw_command=analysis.command,
            ast_json=json.dumps({}), # Placeholder since ast isn't directly in AIAnalysisResult yet
            intent=analysis.intent,
            risk_score=analysis.risk_score,
            risk_level=analysis.risk_level,
            risk_signals=json.dumps(analysis.risk_signals),
            normalized_context_json=json.dumps(analysis.normalized_context),
            trust_score=personalized.trust_score.score,
            trust_breakdown_json=json.dumps(personalized.trust_score.weight_breakdown),
            decision_action=decision.action,
            decision_reason=decision.reason,
            advisory_explanation=advisory.explanation if advisory else None,
            advisory_alternative=advisory.alternative if advisory else None,
            user_response=user_response,
            execution_result_json=json.dumps(execution_result) if execution_result else None,
            rollback_triggered=False # Updated later if rollback occurs
        )
        
        self.session.add(record)
        self.session.commit()
        return record

    def get_history(self, user_id: str, limit: int = 50) -> list[AuditLogRecord]:
        """Fetch the recent command history for a user."""
        return (
            self.session.query(AuditLogRecord)
            .filter(AuditLogRecord.user_id == user_id)
            .order_by(AuditLogRecord.timestamp.desc())
            .limit(limit)
            .all()
        )

    def get_by_command_pattern(self, pattern: str) -> list[AuditLogRecord]:
        """Fetch all audit records matching a normalized command pattern."""
        # Using simple LIKE for pattern matching
        return (
            self.session.query(AuditLogRecord)
            .filter(AuditLogRecord.raw_command.like(f"{pattern}%"))
            .order_by(AuditLogRecord.timestamp.desc())
            .all()
        )
