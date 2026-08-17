"""
personalization/pipeline.py

Single entrypoint integrating Personalization, Decision, Advisory, Audit, and Execution.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from safeshell_member5.audit.logger import AuditLogger
from safeshell_member5.audit.trust_updater import TrustProfileUpdater
from safeshell_member5.execution.engine import ExecutionEngine
from safeshell_member5.models.schemas import (
    AIAnalysisResult,
    ExecutionResult,
    PipelineResult,
)
from safeshell_member5.personalization.engine import PersonalizationEngine
from safeshell_member5.policy.advisory import AdvisoryGenerator, generate_advisory
from safeshell_member5.policy.decision_engine import decide


class Member5Pipeline:
    def __init__(self, session: Session, snapshot_dir: str = "/tmp/safeshell_snapshots"):
        self.session = session
        self.personalization_engine = PersonalizationEngine(session)
        self.execution_engine = ExecutionEngine(snapshot_dir=snapshot_dir)
        self.audit_logger = AuditLogger(session)
        self.trust_updater = TrustProfileUpdater(session)

    def process(self, analysis: AIAnalysisResult, user_id: str, mode: str) -> PipelineResult:
        """Process the analysis through personalization, decision, and advisory layers."""
        
        personalized = self.personalization_engine.personalize(analysis, user_id)
        decision = decide(personalized)
        
        advisory = None
        if decision.action != "ALLOW":
            advisory = generate_advisory(decision, analysis)
            
        result = PipelineResult(
            personalized=personalized, 
            decision=decision, 
            advisory=advisory
        )
        
        # Log event with execution_result=None and user_response="pending"
        self.audit_logger.log_event(
            user_id=user_id,
            analysis=analysis,
            personalized=personalized,
            decision=decision,
            advisory=advisory,
            execution_result=None,
            user_response="pending"
        )
        
        return result

    def confirm_and_execute(
        self, 
        pipeline_result: PipelineResult, 
        user_response: str, 
        mode: str,
        user_id: str
    ) -> ExecutionResult | None:
        """Execute the command if accepted and not blocked, then update audit and trust."""
        exec_result = None
        analysis = pipeline_result.personalized.ai_analysis
        decision = pipeline_result.decision
        
        if user_response == "accepted" and decision.action != "BLOCK":
            exec_result = self.execution_engine.execute(analysis.command, decision, mode)
            
        # Write the final audit log entry with execution results
        exec_dict = exec_result.model_dump() if exec_result else None
        
        audit_record = self.audit_logger.log_event(
            user_id=user_id,
            analysis=analysis,
            personalized=pipeline_result.personalized,
            decision=decision,
            advisory=pipeline_result.advisory,
            execution_result=exec_dict,
            user_response=user_response
        )
        
        self.trust_updater.update_trust(audit_record)
        
        return exec_result
