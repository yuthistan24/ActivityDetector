"""Unit tests for SessionManager and structured JSONL logging."""

import json
from pathlib import Path
import pytest

from activity_detector.core.engine import StepRecord, StepStatus, WarningEvent, WarningType
from activity_detector.core.procedure import Procedure, StepDefinition
from activity_detector.core.session import SessionManager


@pytest.fixture
def dummy_procedure() -> Procedure:
    return Procedure(
        id="audit_test",
        title="Audit Test Protocol",
        steps=[
            StepDefinition(
                id="s1",
                order=1,
                name="First Step",
                instruction="Do task 1",
            ),
            StepDefinition(
                id="s2",
                order=2,
                name="Second Step",
                instruction="Do task 2",
            ),
        ]
    )


def test_session_lifecycle(tmp_path: Path, dummy_procedure: Procedure):
    mgr = SessionManager(output_root=str(tmp_path))
    assert not mgr.is_active

    # Start session
    session_dir = mgr.start_session(dummy_procedure)
    assert mgr.is_active
    assert session_dir.exists()
    assert (session_dir / "session_log.jsonl").exists()

    # Log custom event
    mgr.log_event(
        event_type="TEST_EVENT",
        message="Running automated audit test",
        confidence=0.95
    )

    # Close session
    summary_path = mgr.close_session(
        final_records=[
            StepRecord(step_id="s1", order=1, name="First Step", instruction="Do task 1", status=StepStatus.COMPLETED, duration_seconds=5.2),
            StepRecord(step_id="s2", order=2, name="Second Step", instruction="Do task 2", status=StepStatus.COMPLETED, duration_seconds=6.1),
        ],
        all_warnings=[
            WarningEvent(warning_type=WarningType.STEP_TIMEOUT, message="Timeout warning test", step_id="s1")
        ]
    )

    assert not mgr.is_active
    assert summary_path is not None
    assert summary_path.exists()

    # Verify JSONL lines
    log_file = session_dir / "session_log.jsonl"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) >= 3  # SESSION_STARTED, TEST_EVENT, SESSION_CLOSED
    assert lines[0]["event_type"] == "SESSION_STARTED"
    assert lines[1]["event_type"] == "TEST_EVENT"

    # Verify summary JSON
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert summary_data["procedure_id"] == "audit_test"
    assert summary_data["completed_steps"] == 2
    assert summary_data["total_steps"] == 2
    assert summary_data["compliance_status"] == "COMPLETED_WITH_WARNINGS"
    assert summary_data["warning_count"] == 1
