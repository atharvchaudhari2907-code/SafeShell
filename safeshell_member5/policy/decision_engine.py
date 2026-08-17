"""
policy/decision_engine.py

Decision Engine for SafeShell Member 5.

Implements the ALLOW/WARN/BLOCK decision per the architecture flowchart.
This is the safety floor — no AI call inside this function, pure rule logic 
on already-computed scores.
"""

from __future__ import annotations

from datetime import datetime, timezone

from safeshell_member5.models.schemas import PersonalizedResult, PolicyDecision


def decide(personalized: PersonalizedResult) -> PolicyDecision:
    """Make the final ALLOW/WARN/BLOCK policy decision.
    
    Deterministic rules override everything (from Rules layer, already 
    baked into risk_level) — critical always wins. Trust can only reduce 
    friction on medium risk.
    """
    risk_level = personalized.ai_analysis.risk_level
    trust_score = personalized.trust_score.score
    timestamp = datetime.now(timezone.utc).isoformat()

    # deterministic rules override everything
    if risk_level == "critical":
        return PolicyDecision(
            action="BLOCK",
            reason="critical risk operation blocked deterministically",
            override_possible=False,
            risk_level=risk_level,
            trust_score=trust_score,
            timestamp=timestamp
        )

    if risk_level == "high":
        return PolicyDecision(
            action="WARN",
            reason="high risk operation requires user confirmation",
            override_possible=True,
            risk_level=risk_level,
            trust_score=trust_score,
            timestamp=timestamp
        )

    if risk_level == "medium":
        # trust can reduce friction here only
        if trust_score > 0.7:
            return PolicyDecision(
                action="ALLOW",
                reason="high trust, medium risk (friction reduced)",
                override_possible=True, # ALLOW inherently implies it goes through, but override_possible applies to warnings. False for ALLOW doesn't make much sense, let's keep True
                risk_level=risk_level,
                trust_score=trust_score,
                timestamp=timestamp
            )
        return PolicyDecision(
            action="WARN",
            reason="medium risk operation requires user confirmation",
            override_possible=True,
            risk_level=risk_level,
            trust_score=trust_score,
            timestamp=timestamp
        )

    if risk_level == "low":
        return PolicyDecision(
            action="ALLOW",
            reason="low risk operation allowed",
            override_possible=True,
            risk_level=risk_level,
            trust_score=trust_score,
            timestamp=timestamp
        )
        
    # Fallback safety (should never reach here given enum types)
    return PolicyDecision(
        action="BLOCK",
        reason="unknown risk level encountered",
        override_possible=False,
        risk_level=risk_level,
        trust_score=trust_score,
        timestamp=timestamp
    )
