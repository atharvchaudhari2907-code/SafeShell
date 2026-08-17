"""
policy/advisory.py

Advisory Engine for SafeShell.
Generates human-readable explanations and safer alternatives for WARN/BLOCK decisions.
Pluggable architecture allows swapping the default RuleBasedAdvisoryGenerator
with an NLG/LLM based one in the future.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import yaml

from safeshell_member5.models.schemas import Advisory, AIAnalysisResult, PolicyDecision
from safeshell_member5.personalization.engine import normalize_pattern


class AdvisoryGenerator(ABC):
    """Base interface for generating advisories."""

    @abstractmethod
    def generate(self, decision: PolicyDecision, analysis: AIAnalysisResult) -> Advisory:
        """Generate an Advisory for the given decision and analysis."""
        pass


class RuleBasedAdvisoryGenerator(AdvisoryGenerator):
    """Default advisory generator using template strings and a YAML lookup table."""

    def __init__(self, rules_path: str | None = None) -> None:
        if rules_path is None:
            rules_path = os.path.join(os.path.dirname(__file__), "advisory_rules.yaml")
            
        self.rules: dict[str, str] = {}
        if os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    self.rules = loaded

    def generate(self, decision: PolicyDecision, analysis: AIAnalysisResult) -> Advisory:
        explanation = self._build_explanation(decision, analysis)
        alternative = self._suggest_alternative(analysis)
        return Advisory(explanation=explanation, alternative=alternative)

    def _build_explanation(self, decision: PolicyDecision, analysis: AIAnalysisResult) -> str:
        """Template-based explanation builder."""
        signals = ", ".join(analysis.risk_signals) if analysis.risk_signals else "none detected"
        
        # Simple templating based on intent and signals
        action_text = "blocked" if decision.action == "BLOCK" else "flagged for review"
        
        explanation = (
            f"This command is {action_text} because it involves '{analysis.intent}' "
            f"operations with {decision.risk_level} risk. "
            f"Risk signals: {signals}."
        )
        return explanation

    def _suggest_alternative(self, analysis: AIAnalysisResult) -> str:
        """Rule-based lookup for safer alternatives based on normalized pattern."""
        pattern = normalize_pattern(analysis.command)
        
        # Try to find exact pattern match
        if pattern in self.rules:
            return self.rules[pattern]
            
        # Fallback generic advice
        if analysis.risk_level in ("high", "critical"):
            return "Verify parameters and consider running in a sandboxed or dry-run environment."
            
        return "Command parameters appear non-standard; verify target path before execution."


# Global default instance (lazy loaded)
_DEFAULT_GENERATOR: AdvisoryGenerator | None = None

def generate_advisory(
    decision: PolicyDecision, 
    analysis: AIAnalysisResult, 
    generator: AdvisoryGenerator | None = None
) -> Advisory:
    """Generate an advisory using the specified generator (or the default rule-based one)."""
    global _DEFAULT_GENERATOR
    
    if generator is None:
        if _DEFAULT_GENERATOR is None:
            _DEFAULT_GENERATOR = RuleBasedAdvisoryGenerator()
        generator = _DEFAULT_GENERATOR
        
    return generator.generate(decision, analysis)
