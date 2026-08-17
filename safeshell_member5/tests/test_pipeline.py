"""
tests/test_pipeline.py

Integration tests covering the entire Member 5 pipeline.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from safeshell_member5.db.models import Base
from safeshell_member5.models.schemas import AIAnalysisResult
from safeshell_member5.personalization.pipeline import Member5Pipeline


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    factory = sessionmaker(bind=eng)
    session = factory()
    yield session
    session.close()


@pytest.fixture()
def pipeline(db_session) -> Generator[Member5Pipeline, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Member5Pipeline(db_session, snapshot_dir=temp_dir)


def test_pipeline_low_risk_auto_allow(pipeline):
    """Low risk should be auto allowed and execute directly if user accepts."""
    analysis = AIAnalysisResult(
        command="echo hello",
        normalized_context={"cwd": "/tmp"},
        intent="print",
        risk_score=0.1,
        risk_level="low",
        risk_signals=[],
        confidence=0.9
    )
    
    # Process
    result = pipeline.process(analysis, "alice", "direct")
    assert result.decision.action == "ALLOW"
    assert result.advisory is None
    
    # Execute
    exec_res = pipeline.confirm_and_execute(result, "accepted", "direct", "alice")
    assert exec_res is not None
    assert exec_res.status == "success"
    assert exec_res.stdout.strip() == "hello"


def test_pipeline_critical_risk_blocked(pipeline):
    """Critical risk must be blocked and execution skipped regardless of user acceptance."""
    analysis = AIAnalysisResult(
        command="rm -rf /",
        normalized_context={"cwd": "/"},
        intent="delete",
        risk_score=1.0,
        risk_level="critical",
        risk_signals=["root_delete"],
        confidence=0.99
    )
    
    result = pipeline.process(analysis, "alice", "direct")
    assert result.decision.action == "BLOCK"
    assert result.advisory is not None
    
    # Attempt to execute even if user 'accepted' (bypassing TUI safety)
    exec_res = pipeline.confirm_and_execute(result, "accepted", "direct", "alice")
    assert exec_res is None # Execution skipped because decision is BLOCK


def test_pipeline_dry_run_does_not_mutate(pipeline):
    """Dry run should simulate without side effects."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"data")
        test_file = f.name
        
    analysis = AIAnalysisResult(
        command=f"rm {test_file}",
        normalized_context={"cwd": "/tmp"},
        intent="delete",
        risk_score=0.5,
        risk_level="medium",
        risk_signals=[],
        confidence=0.8
    )
    
    result = pipeline.process(analysis, "bob", "dry_run")
    exec_res = pipeline.confirm_and_execute(result, "accepted", "dry_run", "bob")
    
    assert exec_res is not None
    assert exec_res.status == "dry_run"
    assert test_file in exec_res.dry_run_result.affected_paths
    
    # File should still exist
    assert os.path.exists(test_file)
    os.remove(test_file)


def test_pipeline_transactional_rollback(pipeline):
    """Failed transactional operation should auto-rollback."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"precious data")
        test_file = f.name
        
    # We will simulate a failure by using a command that deletes the file 
    # but then fails (e.g. rm test_file && false)
    analysis = AIAnalysisResult(
        command=f"rm {test_file} && false",
        normalized_context={"cwd": "/tmp"},
        intent="delete",
        risk_score=0.5,
        risk_level="medium",
        risk_signals=[],
        confidence=0.8
    )
    
    # Process
    result = pipeline.process(analysis, "bob", "direct")
    
    # Manually adjust trust to bypass warning for test
    result.decision.action = "ALLOW" 
    
    # Execute - it should delete the file, then fail, then auto-rollback
    exec_res = pipeline.confirm_and_execute(result, "accepted", "direct", "bob")
    
    assert exec_res is not None
    assert exec_res.status == "failure"
    assert exec_res.exit_code != 0
    
    # File should have been restored by rollback
    assert os.path.exists(test_file)
    
    with open(test_file, "rb") as f:
        assert f.read() == b"precious data"
        
    os.remove(test_file)
