"""
models/schemas.py

Pydantic data contracts for SafeShell Member 5.

``AIAnalysisResult`` is the *input contract* — the payload this module
receives from Member 4's Confidence Fusion / Intent+Risk model output.
All downstream Member 5 modules consume it.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AIAnalysisResult(BaseModel):
    """Output of the upstream AI analysis pipeline (Member 4).

    Every field is mandatory.  ``risk_score`` and ``confidence`` are
    floating-point values clamped to the [0.0, 1.0] range.
    """

    command: str = Field(
        ..., description="Raw command string entered by the user."
    )
    normalized_context: dict[str, Any] = Field(
        ..., description="Structured context from the Context Builder."
    )
    intent: str = Field(
        ..., description="Inferred intent label (e.g. 'delete_files')."
    )
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Continuous risk score."
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Categorical risk classification."
    )
    risk_signals: list[str] = Field(
        ..., description="Human-readable risk signal descriptions."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence in the analysis."
    )


class TrustScoreResult(BaseModel):
    """Output of the adaptive trust-score calculator.

    ``weight_breakdown`` always contains the *true* computed component
    values (even when ``critical_override_active`` forces ``score`` to
    0.0) so that the audit log can inspect the underlying computation.
    """

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Effective trust score after time decay and critical override. "
            "0.0 when critical_override_active is True."
        ),
    )
    weight_breakdown: dict[str, float] = Field(
        ...,
        description=(
            "Per-component scores: frequency, recency, decision, "
            "risk_inverse, consistency, raw_weighted, time_decayed."
        ),
    )
    critical_override_active: bool = Field(
        default=False,
        description=(
            "True when the current risk_level is 'critical', meaning "
            "score was forced to 0.0 regardless of trust history."
        ),
    )


class PersonalizedResult(BaseModel):
    """Output of the Personalization Engine."""

    ai_analysis: AIAnalysisResult
    trust_score: TrustScoreResult
    profile_id: int
    friction_adjustment: str = Field(
        ...,
        description="Suggested friction adjustment: 'reduce', 'none', or 'increase'."
    )


class Advisory(BaseModel):
    """Human-readable explanation and alternative suggestion for a decision."""

    explanation: str
    alternative: str



class PolicyDecision(BaseModel):
    """Output of the Policy/Decision Engine."""

    action: Literal["ALLOW", "WARN", "BLOCK"]
    reason: str
    override_possible: bool
    risk_level: str
    trust_score: float
    timestamp: str = Field(
        ...,
        description="ISO format timestamp of the decision."
    )
    advisory: Optional[Advisory] = None


class DryRunResult(BaseModel):
    affected_paths: list[str]
    predicted_changes: list[str]
    risk_level: str


class RollbackResult(BaseModel):
    status: Literal["success", "failure", "not_reversible"]
    detail: str


class ExecutionResult(BaseModel):
    status: Literal["success", "failure", "blocked", "dry_run"]
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    transaction_id: Optional[str] = None
    dry_run_result: Optional[DryRunResult] = None


class PipelineResult(BaseModel):
    personalized: PersonalizedResult
    decision: PolicyDecision
    advisory: Optional[Advisory] = None

