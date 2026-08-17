"""
audit/trust_updater.py

Closes the loop between the Audit Log and the Trust Profile.
Runs after each audit log write to update the user's trust profile based on their behavior.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from safeshell_member5.db.models import AuditLogRecord
from safeshell_member5.personalization.engine import normalize_pattern
from safeshell_member5.personalization.trust_profile import UserTrustProfile
from safeshell_member5.personalization.trust_score import hash_context


class TrustProfileUpdater:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.trust_profile = UserTrustProfile(session)

    def update_trust(self, audit_record: AuditLogRecord) -> None:
        """Update the user's trust profile based on the audit record."""
        pattern = normalize_pattern(audit_record.raw_command)
        
        context: dict[str, Any] = {}
        try:
            if audit_record.normalized_context_json:
                context = json.loads(audit_record.normalized_context_json)
        except json.JSONDecodeError:
            pass
            
        context_hash = hash_context(context)
        
        self.trust_profile.record_usage(
            user_id=audit_record.user_id,
            command_pattern=pattern,
            risk_score=audit_record.risk_score,
            user_decision=audit_record.user_response,
            context_hash=context_hash
        )
