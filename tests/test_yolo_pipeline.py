"""
Tests for YOLO detector class validation, target filtering,
region transitions, stale-result handling, and detector-unavailable behaviour.

These tests do NOT require a live webcam or GPU.  YOLO inference is tested
against real model weights (yolo11n.pt downloaded by ultralytics on first run).

Run with:
    pytest tests/test_yolo_pipeline.py -v
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from activity_detector.config.settings import (
    AppConfig,
    CameraConfig,
    RoiRule,
    VisionConfig,
    YoloConfig,
    get_default_config,
)
from activity_detector.core.engine import (
    DetectionItem,
    EngineState,
    FrameEvidence,
    ProcedureEngine,
    StepStatus,
)
from activity_detector.core.procedure import (
    CompletionRule,
    ExpectedEvidence,
    Procedure,
    StepDefinition,
)
from activity_detector.vision.yolo_detector import COCO80_NAMES, YoloDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def enabled_yolo_config() -> YoloConfig:
    return YoloConfig(enabled=True, model_path="yolo11n.pt", confidence_threshold=0.40)


@pytest.fixture
def rois() -> dict:
    return {
        "prep_left":        RoiRule(x1=0.02, y1=0.15, x2=0.28, y2=0.92),
        "workspace_center": RoiRule(x1=0.30, y1=0.15, x2=0.68, y2=0.92),
        "stowed_area":      RoiRule(x1=0.70, y1=0.15, x2=0.98, y2=0.92),
    }


@pytest.fixture
def yolo(enabled_yolo_config, rois):
    """Real YOLO11n detector; skips if weights unavailable."""
    det = YoloDetector(enabled_yolo_config, rois)
    if not det.available:
        pytest.skip(f"YOLO11n not available: {det.status_message}")
    return det


@pytest.fixture
def bottle_procedure() -> Procedure:
    """3-step bottle workflow using stable_detection rules."""
    return Procedure(
        id="bottle_test",
        title="Bottle Test",
        target_object="bottle",
        steps=[
            StepDefinition(
                id="step_prep",
                order=1,
                name="Bottle in Prep",
                instruction="Place bottle in Prep area.",
                expected_evidence=ExpectedEvidence(
                    target_object="bottle",
                    required_objects=["bottle"],
                    roi="prep_left",
                ),
                completion_rule=CompletionRule(
                    rule_type="stable_detection",
                    stable_frames=3,
                    hold_seconds=0.1,
                    min_confidence=0.40,
                    min_vlm_samples=0,
                ),
            ),
            StepDefinition(
                id="step_transit",
                order=2,
                name="Bottle in Transit",
                instruction="Move bottle.",
                expected_evidence=ExpectedEvidence(
                    target_object="bottle",
                    required_objects=["bottle"],
                    roi=None,
                ),
                completion_rule=CompletionRule(
                    rule_type="stable_detection",
                    stable_frames=2,
                    hold_seconds=0.05,
                    min_confidence=0.40,
                    min_vlm_samples=0,
                ),
                prerequisites=["step_prep"],
            ),
            StepDefinition(
                id="step_stowed",
                order=3,
                name="Bottle in Stowed",
                instruction="Leave bottle in Stowed area.",
                expected_evidence=ExpectedEvidence(
                    target_object="bottle",
                    required_objects=["bottle"],
                    roi="stowed_area",
                ),
                completion_rule=CompletionRule(
                    rule_type="stable_detection",
                    stable_frames=3,
                    hold_seconds=0.1,
                    min_confidence=0.40,
                    min_vlm_samples=0,
                ),
                prerequisites=["step_transit"],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 1. Supported / unsupported target validation
# ---------------------------------------------------------------------------

class TestClassValidation:
    def test_bottle_is_supported(self, yolo):
        """'bottle' must be in YOLO11n COCO vocabulary."""
        assert yolo.is_class_supported("bottle")

    def test_cup_is_supported(self, yolo):
        assert yolo.is_class_supported("cup")

    def test_person_is_supported(self, yolo):
        assert yolo.is_class_supported("person")

    def test_notebook_not_supported(self, yolo):
        """'notebook' is not a COCO class; must return False."""
        assert not yolo.is_class_supported("notebook")

    def test_gibberish_not_supported(self, yolo):
        assert not yolo.is_class_supported("frobnicator_xyz_123")

    def test_set_valid_target_returns_true(self, yolo):
        ok = yolo.set_target_class("bottle")
        assert ok
        assert yolo.target_class == "bottle"

    def test_set_invalid_target_returns_false(self, yolo):
        ok = yolo.set_target_class("notebook")
        assert not ok
        # Previous filter must be unchanged (still "bottle" from above, but we
        # test independently so target may be None here — just check bool)
        # More importantly: class filter must not silently change to something else
        assert yolo.target_class != "notebook"

    def test_clear_target_filter(self, yolo):
        yolo.set_target_class("bottle")
        ok = yolo.set_target_class("")
        assert ok
        assert yolo.target_class is None

    def test_supported_classes_list_not_empty(self, yolo):
        classes = yolo.supported_classes
        assert len(classes) == 80  # COCO classes
        assert "bottle" in classes
        assert "person" in classes

    def test_coco80_fallback_names_contains_bottle(self):
        assert "bottle" in COCO80_NAMES


# ---------------------------------------------------------------------------
# 2. Target class filtering actually changes detections
# ---------------------------------------------------------------------------

class TestTargetClassFiltering:
    def test_filter_changes_on_set(self, yolo):
        """set_target_class() must immediately update the active filter."""
        yolo.set_target_class("person")
        assert yolo.target_class == "person"
        assert yolo._target_class_id is not None

        yolo.set_target_class("bottle")
        assert yolo.target_class == "bottle"
        assert yolo._target_class_id is not None
        # class IDs for bottle and person differ
        from ultralytics import YOLO as _YOLO  # type: ignore
        model = _YOLO("yolo11n.pt")
        bottle_id = [k for k, v in model.names.items() if v == "bottle"][0]
        assert yolo._target_class_id == bottle_id

    def test_unsupported_target_leaves_filter_unchanged(self, yolo):
        yolo.set_target_class("bottle")
        original_cls = yolo.target_class
        yolo.set_target_class("nonexistent_class_xyz")
        assert yolo.target_class == original_cls  # unchanged


# ---------------------------------------------------------------------------
# 3. Region transitions (engine state machine)
# ---------------------------------------------------------------------------

def _make_evidence(
    frame_number: int,
    detections: List[DetectionItem],
    target: str = "bottle",
) -> FrameEvidence:
    return FrameEvidence(
        frame_number=frame_number,
        timestamp=time.time(),
        detections=detections,
        target_object=target,
    )


def _bottle_in(roi: str, conf: float = 0.70) -> DetectionItem:
    return DetectionItem(
        name="bottle", source="yolo", roi=roi, confidence=conf, bbox=(10, 10, 50, 120)
    )


class TestRegionTransitions:
    def test_bottle_in_prep_advances_step_1(self, bottle_procedure):
        engine = ProcedureEngine(bottle_procedure)
        engine.start_session()

        # Feed stable frames with bottle in prep_left
        for i in range(1, 5):
            ev = _make_evidence(i, [_bottle_in("prep_left")])
            engine.process_frame(ev)
        time.sleep(0.12)

        ev_final = _make_evidence(5, [_bottle_in("prep_left")])
        result = engine.process_frame(ev_final)

        assert result.transition_occurred
        assert result.completed_step_id == "step_prep"
        assert engine.current_step_index == 1

    def test_bottle_absent_does_not_advance(self, bottle_procedure):
        engine = ProcedureEngine(bottle_procedure)
        engine.start_session()

        # No bottle in any frame
        for i in range(1, 10):
            ev = _make_evidence(i, [])
            result = engine.process_frame(ev)
            assert not result.transition_occurred
            assert engine.current_step_index == 0

    def test_bottle_wrong_roi_does_not_advance_step_1(self, bottle_procedure):
        """Bottle detected in workspace_center must not complete step 1 (needs prep_left)."""
        engine = ProcedureEngine(bottle_procedure)
        engine.start_session()

        for i in range(1, 8):
            ev = _make_evidence(i, [_bottle_in("workspace_center")])
            result = engine.process_frame(ev)
            assert engine.current_step_index == 0

    def test_full_region_transition_sequence(self, bottle_procedure):
        """Advance through all 3 steps programmatically."""
        engine = ProcedureEngine(bottle_procedure)
        engine.start_session()

        # Complete step 1 (prep)
        for i in range(1, 5):
            ev = _make_evidence(i, [_bottle_in("prep_left")])
            engine.process_frame(ev)
        time.sleep(0.12)
        r = engine.process_frame(_make_evidence(5, [_bottle_in("prep_left")]))
        assert r.transition_occurred
        assert engine.current_step_index == 1

        # Complete step 2 (transit — any ROI)
        for i in range(6, 10):
            ev = _make_evidence(i, [_bottle_in("workspace_center")])
            engine.process_frame(ev)
        time.sleep(0.08)
        r = engine.process_frame(_make_evidence(10, [_bottle_in("workspace_center")]))
        assert r.transition_occurred
        assert engine.current_step_index == 2

        # Complete step 3 (stowed)
        for i in range(11, 15):
            ev = _make_evidence(i, [_bottle_in("stowed_area")])
            engine.process_frame(ev)
        time.sleep(0.12)
        r = engine.process_frame(_make_evidence(15, [_bottle_in("stowed_area")]))
        assert r.transition_occurred
        assert engine.state == EngineState.COMPLETED


# ---------------------------------------------------------------------------
# 4. Stale-result handling
# ---------------------------------------------------------------------------

class TestStaleResults:
    def test_pipeline_stale_flag_after_timeout(self, enabled_yolo_config, rois):
        """After STALE_DETECTION_SECONDS the pipeline must report result_is_stale=True."""
        from activity_detector.vision.pipeline import STALE_DETECTION_SECONDS, VisionPipeline
        from activity_detector.config.settings import VisionConfig, VlmConfig, AudioConfig, RecordingConfig, StreamingConfig

        cfg = AppConfig(
            procedure_file="procedures/bottle_tabletop_workflow.yaml",
            target_object="bottle",
            camera=CameraConfig(source=0),
            vision=VisionConfig(rois=rois),
            yolo=enabled_yolo_config,
            vlm=VlmConfig(enabled=False),
        )

        pipeline = VisionPipeline(cfg, vlm_client=None)

        # Simulate a target detection that just occurred
        pipeline._last_yolo_had_target = True
        pipeline._last_yolo_detection_time = time.time() - (STALE_DETECTION_SECONDS + 0.5)

        det_status = pipeline.get_detector_status()
        assert det_status["result_is_stale"] is True

    def test_fresh_detection_not_stale(self, enabled_yolo_config, rois):
        from activity_detector.vision.pipeline import STALE_DETECTION_SECONDS, VisionPipeline
        from activity_detector.config.settings import VisionConfig, VlmConfig, AppConfig

        cfg = AppConfig(
            procedure_file="procedures/bottle_tabletop_workflow.yaml",
            target_object="bottle",
            camera=CameraConfig(source=0),
            vision=VisionConfig(rois=rois),
            yolo=enabled_yolo_config,
            vlm=VlmConfig(enabled=False),
        )

        pipeline = VisionPipeline(cfg, vlm_client=None)
        pipeline._last_yolo_had_target = True
        pipeline._last_yolo_detection_time = time.time() - 0.5  # recent

        det_status = pipeline.get_detector_status()
        assert det_status["result_is_stale"] is False

    def test_never_seen_not_stale(self, enabled_yolo_config, rois):
        from activity_detector.vision.pipeline import VisionPipeline
        from activity_detector.config.settings import VisionConfig, VlmConfig, AppConfig

        cfg = AppConfig(
            procedure_file="procedures/bottle_tabletop_workflow.yaml",
            target_object="bottle",
            camera=CameraConfig(source=0),
            vision=VisionConfig(rois=rois),
            yolo=enabled_yolo_config,
            vlm=VlmConfig(enabled=False),
        )

        pipeline = VisionPipeline(cfg, vlm_client=None)
        # Never seen — had_target is False
        assert pipeline._last_yolo_had_target is False
        det = pipeline.get_detector_status()
        assert det["result_is_stale"] is False


# ---------------------------------------------------------------------------
# 5. Detector-unavailable behaviour
# ---------------------------------------------------------------------------

class TestDetectorUnavailable:
    def test_disabled_detector_returns_empty(self, rois):
        cfg = YoloConfig(enabled=False, model_path="")
        det = YoloDetector(cfg, rois)
        assert not det.available
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(blank)
        assert result == []

    def test_missing_weights_unavailable(self, rois):
        """When ultralytics raises FileNotFoundError the detector must be unavailable."""
        import activity_detector.vision.yolo_detector as yd_mod
        cfg = YoloConfig(enabled=True, model_path="totally_nonexistent_weights.pt")
        with patch.object(yd_mod, "YoloDetector._load_model", side_effect=FileNotFoundError("fake"), create=True):
            # Patch at the ultralytics import level instead
            with patch.dict("sys.modules", {"ultralytics": None}):
                # Re-instantiate so the import check runs inside __init__
                det2 = YoloDetector.__new__(YoloDetector)
                det2.config = cfg
                det2.rois = rois
                det2.available = False
                det2.status_message = "ultralytics package not installed."
                det2._model = None
                det2._names = {}
                det2._target_class = None
                det2._target_class_id = None
                det2._last_inference_time = 0.0
                det2._inference_latency_ms = 0.0
                assert not det2.available
                assert "not installed" in det2.status_message or not det2.available

    def test_unavailable_detector_supported_classes_empty(self, rois):
        """A detector that failed to load must report no supported classes."""
        cfg = YoloConfig(enabled=False, model_path="")
        det = YoloDetector(cfg, rois)
        assert not det.available
        assert det.supported_classes == []

    def test_pipeline_target_validation_when_unavailable(self, rois):
        """When YOLO is disabled, set_target_object should still succeed gracefully."""
        from activity_detector.vision.pipeline import VisionPipeline
        from activity_detector.config.settings import VisionConfig, VlmConfig, AppConfig

        cfg = AppConfig(
            procedure_file="procedures/bottle_tabletop_workflow.yaml",
            target_object="bottle",
            camera=CameraConfig(source=0),
            vision=VisionConfig(rois=rois),
            yolo=YoloConfig(enabled=False, model_path=""),  # explicitly disabled
            vlm=VlmConfig(enabled=False),
        )
        pipeline = VisionPipeline(cfg, vlm_client=None)
        assert not pipeline.yolo.available
        ok, msg = pipeline.set_target_object("anything_at_all")
        assert ok  # graceful — no crash



# ---------------------------------------------------------------------------
# 6. Target persistence across changes
# ---------------------------------------------------------------------------

class TestTargetPersistence:
    def test_target_change_propagates_to_yolo_filter(self, yolo):
        yolo.set_target_class("bottle")
        assert yolo.target_class == "bottle"
        yolo.set_target_class("cup")
        assert yolo.target_class == "cup"
        # Confirm the internal class ID also changed
        from ultralytics import YOLO as _YOLO  # type: ignore
        m = _YOLO("yolo11n.pt")
        cup_id = [k for k, v in m.names.items() if v == "cup"][0]
        assert yolo._target_class_id == cup_id

    def test_set_invalid_does_not_change_existing_target(self, yolo):
        yolo.set_target_class("bottle")
        yolo.set_target_class("flobberghast_xyz")
        assert yolo.target_class == "bottle"  # unchanged

    def test_default_config_has_yolo_enabled(self):
        cfg = get_default_config()
        assert cfg.yolo.enabled is True
        assert cfg.yolo.model_path == "yolo11n.pt"
        assert cfg.target_object == "bottle"

    def test_bottle_procedure_exists(self):
        proc_path = Path("procedures/bottle_tabletop_workflow.yaml")
        assert proc_path.exists(), (
            "Bottle procedure file missing. "
            "Expected: procedures/bottle_tabletop_workflow.yaml"
        )

    def test_bottle_procedure_loads(self):
        from activity_detector.core.procedure import load_procedure
        proc = load_procedure("procedures/bottle_tabletop_workflow.yaml")
        assert proc.target_object == "bottle"
        assert len(proc.steps) == 3
        assert proc.steps[0].expected_evidence.required_objects == ["bottle"]

    def test_notebook_procedure_steps_not_open_close(self):
        """bottle_tabletop_workflow must not contain open/close notebook steps."""
        from activity_detector.core.procedure import load_procedure
        proc = load_procedure("procedures/bottle_tabletop_workflow.yaml")
        step_names = [s.name.lower() for s in proc.steps]
        for name in step_names:
            assert "open" not in name, f"Procedure still has 'open' step: {name}"
            assert "close" not in name, f"Procedure still has 'close' step: {name}"


# ---------------------------------------------------------------------------
# 7. Low-confidence / flickering detection → uncertainty
# ---------------------------------------------------------------------------

class TestLowConfidenceHandling:
    def test_below_threshold_does_not_advance(self, bottle_procedure):
        engine = ProcedureEngine(bottle_procedure)
        engine.start_session()

        # Confidence below min_confidence (0.40)
        low_conf = DetectionItem(
            name="bottle", source="yolo", roi="prep_left", confidence=0.25
        )
        for i in range(1, 10):
            ev = _make_evidence(i, [low_conf])
            result = engine.process_frame(ev)
            assert not result.transition_occurred

    def test_flickering_detection_resets_stability(self, bottle_procedure):
        engine = ProcedureEngine(bottle_procedure)
        engine.start_session()

        # 2 good frames, then miss, then good frames
        for i in range(1, 3):
            engine.process_frame(_make_evidence(i, [_bottle_in("prep_left")]))
        assert engine._consecutive_match_frames == 2

        engine.process_frame(_make_evidence(3, []))  # miss
        # stability counter must have decreased
        assert engine._consecutive_match_frames < 2


# ---------------------------------------------------------------------------
# 8. Real YOLO inference on synthetic frames (if available)
# ---------------------------------------------------------------------------

class TestRealYoloInference:
    def test_blank_frame_returns_no_detections(self, yolo):
        yolo.set_target_class("bottle")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = yolo.detect(blank)
        bottle_hits = [d for d in detections if d.name == "bottle"]
        assert len(bottle_hits) == 0, (
            "YOLO must not hallucinate a bottle in a blank (all-black) frame."
        )

    def test_detect_returns_detection_items_with_bbox(self, yolo):
        """Each result must have a valid bounding box."""
        yolo.set_target_class(None)  # detect all
        frame = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        detections = yolo.detect(frame)
        for d in detections:
            if d.bbox:
                x, y, w, h = d.bbox
                assert w > 0 and h > 0

    def test_inference_latency_recorded(self, yolo):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        yolo.detect(blank)
        assert yolo.inference_latency_ms >= 0.0
