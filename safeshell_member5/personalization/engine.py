"""
personalization/engine.py

Personalization Engine that ties trust profile and trust score together.
"""

from __future__ import annotations

import shlex

from sqlalchemy.orm import Session

from safeshell_member5.models.schemas import AIAnalysisResult, PersonalizedResult, TrustScoreResult
from safeshell_member5.personalization.trust_profile import UserTrustProfile
from safeshell_member5.personalization.trust_score import compute_trust_score


def normalize_pattern(command: str) -> str:
    """Extract command and flags, replacing arguments with a placeholder.
    
    Example: 'rm -rf /tmp/data' -> 'rm -rf <path>'
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
        
    if not tokens:
        return ""
        
    pattern = []
    # Capture command name and sudo if present
    idx = 0
    if tokens[idx] == "sudo":
        pattern.append("sudo")
        idx += 1
        
    if idx < len(tokens):
        pattern.append(tokens[idx])
        idx += 1
        
    for tok in tokens[idx:]:
        if tok.startswith("-"):
            pattern.append(tok)
        elif any(c in tok for c in ("/", "\\", ".", "~")) or "=" in tok:
            if not pattern or pattern[-1] != "<path>":
                pattern.append("<path>")
        else:
            pattern.append(tok)
            
    return " ".join(pattern)


def derive_friction_adjustment(trust_score: TrustScoreResult, risk_level: str) -> str:
    """Map (trust_score, risk_level) -> suggested friction delta.
    
    CRITICAL RULE: NEVER touch critical risk path.
    """
    if risk_level == "critical":
        return "none"
        
    score = trust_score.score
    
    if score >= 0.75 and risk_level in ("low", "medium"):
        return "reduce"
    elif score <= 0.25:
        return "increase"
        
    return "none"


class PersonalizationEngine:
    """Ties together trust profile and trust score computation."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._trust_profile = UserTrustProfile(session)

    def personalize(self, analysis: AIAnalysisResult, user_id: str) -> PersonalizedResult:
        """Personalize the AI analysis for the given user."""
        pattern = normalize_pattern(analysis.command)
        profile = self._trust_profile.get_or_create(user_id, pattern)
        
        trust_score = compute_trust_score(
            profile, 
            analysis.normalized_context, 
            current_risk_level=analysis.risk_level
        )
        
        friction = derive_friction_adjustment(trust_score, analysis.risk_level)
        
        return PersonalizedResult(
            ai_analysis=analysis,
            trust_score=trust_score,
            profile_id=profile.id,
            friction_adjustment=friction
        )
