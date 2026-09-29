"""Unit tests for ProcedureEngine state machine and error validation."""

import time
import pytest

from activity_detector.core.engine import (
    DetectionItem,
    EngineState,
    FrameEvidence,
    ProcedureEngine,
    StepStatus,
    WarningType,
)
from activity_detector.core.procedure import (
    CompletionRule,
    ExpectedEvidence,
    Procedure,
    StepDefinition,
)


@pytest.fixture
def sample_procedure() -> Procedure:
    return Procedure(
        id="test_seq",
        title="Sequence Test",
        steps=[
            StepDefinition(
                id="step_1",
                order=1,
                name="Step 1 Safety",
                instruction="Inspect safety",
                expected_evidence=ExpectedEvidence(required_colors=["yellow_flask"], roi="staging_left"),
                completion_rule=CompletionRule(stable_frames=3, hold_seconds=0.1, min_confidence=0.6),
                timeout_seconds=2.0,
            ),
            StepDefinition(
                id="step_2",
                order=2,
                name="Step 2 Reaction",
                instruction="Move flask to center",
                expected_evidence=ExpectedEvidence(required_colors=["yellow_flask"], roi="workbench_center"),
                completion_rule=CompletionRule(stable_frames=3, hold_seconds=0.1, min_confidence=0.6),
            ),
            StepDefinition(
                id="step_3",
                order=3,
                name="Step 3 Titrant",
                instruction="Add blue titrant",
                expected_evidence=ExpectedEvidence(required_colors=["blue_reagent"], roi="workbench_center"),
                completion_rule=CompletionRule(stable_frames=3, hold_seconds=0.1, min_confidence=0.6),
            ),
        ]
    )


def test_initial_engine_state(sample_procedure: Procedure):
    engine = ProcedureEngine(sample_procedure)
    assert engine.state == EngineState.IDLE
    assert engine.current_step_index == 0


def test_start_session(sample_procedure: Procedure):
    engine = ProcedureEngine(sample_procedure)
    engine.start_session()
    assert engine.state == EngineState.IN_PROGRESS
    assert engine.step_records[0].status == StepStatus.ACTIVE
    assert engine.step_records[1].status == StepStatus.PENDING


def test_step_completion_progression(sample_procedure: Procedure):
    engine = ProcedureEngine(sample_procedure)
    transitions = []
    engine.register_transition_listener(lambda r: transitions.append(r.step_id))
    engine.start_session()

    def make_evidence(f_num: int) -> FrameEvidence:
        return FrameEvidence(
            frame_number=f_num,
            timestamp=time.time(),
            detections=[
                DetectionItem(name="yellow_flask", source="color_roi", roi="staging_left", confidence=0.9)
            ]
        )

    # Frame 1: stability increases
    u1 = engine.process_frame(make_evidence(1))
    assert u1.current_step_index == 0
    assert not u1.transition_occurred

    # Frame 2
    u2 = engine.process_frame(make_evidence(2))
    assert u2.current_step_index == 0

    # Wait for hold_seconds (0.1s)
    time.sleep(0.15)
    # Frame 3: satisfies stable_frames=3 and hold_seconds=0.1s -> triggers transition!
    u3 = engine.process_frame(make_evidence(3))
    assert u3.transition_occurred
    assert u3.completed_step_id == "step_1"
    assert u3.current_step_index == 1
    assert engine.step_records[0].status == StepStatus.COMPLETED
    assert engine.step_records[1].status == StepStatus.ACTIVE
    assert transitions == ["step_1"]


def test_skipped_step_warning(sample_procedure: Procedure):
    engine = ProcedureEngine(sample_procedure)
    warnings = []
    engine.register_warning_listener(lambda w: warnings.append(w.warning_type))
    engine.start_session()

    # While on step_1 (yellow_flask in staging_left), operator skips ahead to step_3 (blue_reagent in workbench_center)
    future_evidence = FrameEvidence(
        frame_number=1,
        timestamp=time.time(),
        detections=[
            DetectionItem(name="blue_reagent", source="color_roi", roi="workbench_center", confidence=0.85)
        ]
    )

    # Send 7 frames of future step evidence
    for i in range(7):
        engine.process_frame(future_evidence)

    # Skipped step warning should be emitted
    assert WarningType.SKIPPED_STEP in warnings
    assert engine.state == EngineState.NEEDS_ATTENTION


def test_timeout_warning(sample_procedure: Procedure):
    engine = ProcedureEngine(sample_procedure)
    warnings = []
    engine.register_warning_listener(lambda w: warnings.append(w.warning_type))
    engine.start_session()

    # Step 1 timeout is 2.0s
    empty_evidence_1 = FrameEvidence(frame_number=1, timestamp=time.time(), detections=[])
    engine.process_frame(empty_evidence_1)

    time.sleep(2.1)
    empty_evidence_2 = FrameEvidence(frame_number=2, timestamp=time.time(), detections=[])
    u = engine.process_frame(empty_evidence_2)

    assert WarningType.STEP_TIMEOUT in warnings
    assert u.state == EngineState.NEEDS_ATTENTION


def test_manual_operator_advance_and_revert(sample_procedure: Procedure):
    engine = ProcedureEngine(sample_procedure)
    engine.start_session()
    assert engine.current_step_index == 0

    # Operator advances
    ok = engine.advance_step(reason="operator_test")
    assert ok
    assert engine.current_step_index == 1
    assert engine.step_records[0].status == StepStatus.COMPLETED

    # Operator reverts
    ok_rev = engine.revert_step(reason="operator_revert")
    assert ok_rev
    assert engine.current_step_index == 0
    assert engine.step_records[0].status == StepStatus.ACTIVE


def test_full_procedure_completion(sample_procedure: Procedure):
    engine = ProcedureEngine(sample_procedure)
    engine.start_session()

    # Complete step 1
    engine.advance_step()
    # Complete step 2
    engine.advance_step()
    # Complete step 3
    engine.advance_step()

    assert engine.state == EngineState.COMPLETED
    assert all(r.status == StepStatus.COMPLETED for r in engine.step_records)


def test_notebook_vlm_multi_sample_requirement():
    """Verify that a single VLM observation cannot advance a step, requiring min_vlm_samples."""
    notebook_proc = Procedure(
        id="notebook_test",
        title="Notebook Handling Test",
        target_object="notebook",
        steps=[
            StepDefinition(
                id="step_locate",
                order=1,
                name="Locate Notebook",
                instruction="Place notebook in workspace center",
                expected_evidence=ExpectedEvidence(
                    target_object="notebook",
                    expected_state={"object_visible": True, "location": "workspace_center"},
                ),
                completion_rule=CompletionRule(
                    rule_type="vlm_state_tracking",
                    min_vlm_samples=2,
                    stable_frames=3,
                    hold_seconds=0.05,
                    min_confidence=0.70,
                ),
            )
        ]
    )

    engine = ProcedureEngine(notebook_proc)
    engine.start_session()

    # Observation 1: sample_001
    ev_sample1 = FrameEvidence(
        frame_number=1,
        timestamp=time.time(),
        target_object="notebook",
        vlm_sample_id="sample_001",
        vlm_confidence=0.85,
        vlm_state={"object_visible": True, "location": "workspace_center", "open_or_closed": "closed"},
        vlm_uncertain=False,
    )

    # Feed 10 frames with only 1 VLM sample - should NOT advance because min_vlm_samples is 2!
    for f in range(1, 10):
        ev = ev_sample1.model_copy()
        ev.frame_number = f
        ev.timestamp = time.time()
        res = engine.process_frame(ev)
        assert not res.transition_occurred
        assert engine.current_step_index == 0

    # Wait for hold duration
    time.sleep(0.08)

    # Observation 2: distinct sample_002 across time
    ev_sample2 = FrameEvidence(
        frame_number=11,
        timestamp=time.time(),
        target_object="notebook",
        vlm_sample_id="sample_002",
        vlm_confidence=0.88,
        vlm_state={"object_visible": True, "location": "workspace_center", "open_or_closed": "closed"},
        vlm_uncertain=False,
    )

    # Feed frames with second observation
    transitioned = False
    for f in range(11, 15):
        ev = ev_sample2.model_copy()
        ev.frame_number = f
        ev.timestamp = time.time()
        res = engine.process_frame(ev)
        if res.transition_occurred:
            transitioned = True
            break

    assert transitioned is True
    assert engine.state == EngineState.COMPLETED
    assert engine.step_records[0].status == StepStatus.COMPLETED


def test_early_stowage_out_of_order_detection():
    """Verify that moving the notebook to stowed area early triggers an OUT_OF_ORDER anomaly."""
    notebook_proc = Procedure(
        id="notebook_test_seq",
        title="Notebook Test",
        target_object="notebook",
        steps=[
            StepDefinition(
                id="step_locate",
                order=1,
                name="Locate Notebook",
                instruction="Locate notebook",
                expected_evidence=ExpectedEvidence(
                    target_object="notebook",
                    expected_state={"object_visible": True, "location": "workspace_center"},
                ),
                completion_rule=CompletionRule(rule_type="vlm_state_tracking", min_vlm_samples=2),
            ),
            StepDefinition(
                id="step_open",
                order=2,
                name="Open Notebook",
                instruction="Open notebook",
                expected_evidence=ExpectedEvidence(
                    target_object="notebook",
                    expected_state={"object_visible": True, "open_or_closed": "open"},
                ),
                completion_rule=CompletionRule(rule_type="vlm_state_tracking", min_vlm_samples=2),
            ),
            StepDefinition(
                id="step_stow",
                order=3,
                name="Stow Notebook",
                instruction="Stow notebook",
                expected_evidence=ExpectedEvidence(
                    target_object="notebook",
                    expected_state={"object_visible": True, "location": "stowed_area", "open_or_closed": "closed"},
                ),
                completion_rule=CompletionRule(rule_type="vlm_state_tracking", min_vlm_samples=2),
            ),
        ]
    )

    engine = ProcedureEngine(notebook_proc)
    warnings = []
    engine.register_warning_listener(lambda w: warnings.append(w.warning_type))
    engine.start_session()

    # While on step 1 (Locate Notebook), notebook is observed already closed in stowed_area
    early_stow_ev = FrameEvidence(
        frame_number=1,
        timestamp=time.time(),
        target_object="notebook",
        vlm_sample_id="sample_stow",
        vlm_confidence=0.90,
        vlm_state={"object_visible": True, "location": "stowed_area", "open_or_closed": "closed"},
    )

    for i in range(5):
        engine.process_frame(early_stow_ev)

    assert WarningType.OUT_OF_ORDER in warnings
    assert engine.state == EngineState.NEEDS_ATTENTION


def test_operator_confirm_and_flag_uncertain():
    """Verify manual operator confirm and inconclusive flagging controls."""
    notebook_proc = Procedure(
        id="notebook_test_manual",
        title="Notebook Test",
        target_object="notebook",
        steps=[
            StepDefinition(
                id="step_locate",
                order=1,
                name="Locate Notebook",
                instruction="Locate notebook",
                expected_evidence=ExpectedEvidence(target_object="notebook"),
                completion_rule=CompletionRule(rule_type="vlm_state_tracking", min_vlm_samples=2),
            )
        ]
    )

    engine = ProcedureEngine(notebook_proc)
    engine.start_session()

    # Operator flags uncertain
    assert engine.flag_uncertain_step(reason="lighting_too_dark")
    assert engine.state == EngineState.UNCERTAIN

    # Operator manually confirms step
    assert engine.confirm_current_step(reason="operator_verified_visually")
    assert engine.state == EngineState.COMPLETED
    assert engine.step_records[0].status == StepStatus.COMPLETED
    assert engine.step_records[0].evidence_source == "operator_verified_visually"
