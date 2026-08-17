"""
tests/test_advisory.py

Tests for the Advisory Engine.
"""

from __future__ import annotations

import os

import pytest

from safeshell_member5.models.schemas import Advisory, AIAnalysisResult, PolicyDecision
from safeshell_member5.policy.advisory import (
    RuleBasedAdvisoryGenerator,
    generate_advisory,
)


@pytest.fixture
def mock_analysis() -> AIAnalysisResult:
    return AIAnalysisResult(
        command="rm -rf /usr/bin",
        normalized_context={},
        intent="recursive_delete",
        risk_score=0.9,
        risk_level="critical",
        risk_signals=["system_path_target", "force_flag_present"],
        confidence=0.95
    )


@pytest.fixture
def mock_decision() -> PolicyDecision:
    return PolicyDecision(
        action="BLOCK",
        reason="critical risk operation blocked deterministically",
        override_possible=False,
        risk_level="critical",
        trust_score=0.0,
        timestamp="2026-08-17T12:00:00Z"
    )


def test_rule_based_advisory_generator_loads_yaml():
    """Ensure the generator properly loads the default YAML rules."""
    generator = RuleBasedAdvisoryGenerator()
    assert "rm -rf <path>" in generator.rules
    assert "git push --force" in generator.rules


def test_generate_advisory_explanation(mock_decision, mock_analysis):
    """Test template explanation generation."""
    advisory = generate_advisory(mock_decision, mock_analysis)
    
    assert "blocked" in advisory.explanation
    assert "recursive_delete" in advisory.explanation
    assert "system_path_target, force_flag_present" in advisory.explanation


def test_generate_advisory_alternative_match(mock_decision, mock_analysis):
    """Test alternative lookup for a known pattern."""
    advisory = generate_advisory(mock_decision, mock_analysis)
    
    # "rm -rf /usr/bin" normalizes to "rm -rf <path>"
    assert "mv <path> /tmp/trash/" in advisory.alternative


def test_generate_advisory_alternative_fallback(mock_decision):
    """Test fallback alternative when pattern is not in rules."""
    analysis = AIAnalysisResult(
        command="unknown_cmd --dangerous-flag",
        normalized_context={},
        intent="unknown",
        risk_score=0.8,
        risk_level="high",
        risk_signals=["unknown_binary"],
        confidence=0.5
    )
    
    advisory = generate_advisory(mock_decision, analysis)
    assert "Verify parameters and consider running in a sandboxed" in advisory.alternative


def test_pluggable_generator(mock_decision, mock_analysis):
    """Test that a custom generator can be plugged in."""
    class CustomGenerator:
        def generate(self, decision, analysis):
            return Advisory(explanation="custom expl", alternative="custom alt")
            
    # type ignore because CustomGenerator doesn't formally inherit AdvisoryGenerator in this quick mock
    advisory = generate_advisory(mock_decision, mock_analysis, generator=CustomGenerator()) # type: ignore
    
    assert advisory.explanation == "custom expl"
    assert advisory.alternative == "custom alt"
