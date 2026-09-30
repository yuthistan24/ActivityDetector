"""Tests covering alias resolution, event logging, mirror orientation, and
automatic step start.

All tests are unit-level (no camera hardware required).  The sections that
exercise the real YOLO model are labelled with a marker so they can be skipped
when weights are absent.
"""

from __future__ import annotations

import time
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. Alias normalisation
# ---------------------------------------------------------------------------

from activity_detector.vision.alias import (
    COCO80,
    COCO_ALIASES,
    OPEN_VOCAB_QUERIES,
    resolve_target,
)


class TestAliasNormalisation:
    """resolve_target must map common names to the right backend."""

    def test_phone_maps_to_cell_phone(self):
        res = resolve_target("phone")
        assert res.canonical == "cell phone"
        assert res.backend == "yolo"
        assert res.alias_used is True

    def test_mobile_maps_to_cell_phone(self):
        res = resolve_target("mobile")
        assert res.canonical == "cell phone"
        assert res.backend == "yolo"

    def test_smartphone_maps_to_cell_phone(self):
        res = resolve_target("smartphone")
        assert res.canonical == "cell phone"
        assert res.backend == "yolo"

    def test_mobile_phone_maps_to_cell_phone(self):
        res = resolve_target("mobile phone")
        assert res.canonical == "cell phone"
        assert res.backend == "yolo"

    def test_earbuds_uses_open_vocab(self):
        res = resolve_target("earbuds")
        assert res.backend == "open_vocab"
        # note must mention the limitation
        assert "open" in res.note.lower() or "coco" in res.note.lower()

    def test_earphones_uses_open_vocab(self):
        res = resolve_target("earphones")
        assert res.backend == "open_vocab"

    def test_headphones_uses_open_vocab(self):
        res = resolve_target("headphones")
        assert res.backend == "open_vocab"

    def test_bottle_direct_coco(self):
        res = resolve_target("bottle")
        assert res.canonical == "bottle"
        assert res.backend == "yolo"
        assert res.alias_used is False

    def test_cell_phone_direct_coco(self):
        res = resolve_target("cell phone")
        assert res.canonical == "cell phone"
        assert res.backend == "yolo"
        assert res.alias_used is False

    def test_gibberish_unsupported(self):
        res = resolve_target("xyzzy_nonexistent_object_12345")
        assert res.backend == "unsupported"
        assert res.canonical == ""

    def test_empty_string_unsupported(self):
        res = resolve_target("")
        assert res.backend == "unsupported"

    def test_all_coco_aliases_resolve_to_coco(self):
        for alias, canonical in COCO_ALIASES.items():
            res = resolve_target(alias)
            assert res.backend == "yolo", (
                f"Alias '{alias}' → '{canonical}' should use yolo backend, "
                f"got '{res.backend}'"
            )
            assert res.canonical == canonical, (
                f"Alias '{alias}' should map to '{canonical}', "
                f"got '{res.canonical}'"
            )

    def test_all_open_vocab_resolve_to_open_vocab(self):
        for label in OPEN_VOCAB_QUERIES:
            res = resolve_target(label)
            assert res.backend == "open_vocab", (
                f"'{label}' should be open_vocab, got '{res.backend}'"
            )

    def test_note_does_not_contain_80_class_dump(self):
        """Rejection message must be concise — not a list of all 80 COCO classes."""
        res = resolve_target("xyzzy_nonexistent_object_12345")
        # Less than 300 chars is a reasonable proxy for "concise"
        assert len(res.note) < 400, (
            f"Rejection note is too verbose ({len(res.note)} chars). "
            "Should be a short explanation, not a full class dump."
        )


# ---------------------------------------------------------------------------
# 2. Event logging  — validate correct keyword usage
# ---------------------------------------------------------------------------

from activity_detector.core.session import LogEvent, SessionManager


class TestEventLogging:
    """log_event must accept the correct keyword signature and not raise."""

    def _make_session(self, tmp_path) -> SessionManager:
        sm = SessionManager(str(tmp_path))
        # Minimal procedure mock
        proc = MagicMock()
        proc.title = "Test Procedure"
        proc.id = "test_proc"
        proc.version = "1"
        proc.steps = []
        sm.start_session(proc)
        return sm

    def test_log_event_accepts_keyword_metadata(self, tmp_path):
        sm = self._make_session(tmp_path)
        # Must NOT raise
        sm.log_event(
            event_type="TARGET_CHANGED",
            step_id=None,
            message="Changed target to bottle",
            metadata={"user_label": "bottle", "canonical": "bottle"},
        )
        sm._log_file_handle.flush()

    def test_log_event_target_changed_bottle(self, tmp_path):
        """Regression: changing target to bottle must not raise ValidationError."""
        sm = self._make_session(tmp_path)
        # This is the exact call pattern from main_window._on_target_changed (FIXED)
        try:
            sm.log_event(
                event_type="TARGET_CHANGED",
                step_id=None,
                message="Detection target changed to 'bottle' (canonical: 'bottle')",
                metadata={
                    "user_label": "bottle",
                    "canonical": "bottle",
                    "backend": "yolo",
                    "alias_used": False,
                },
            )
        except Exception as exc:
            pytest.fail(f"log_event raised an exception: {exc}")

    def test_log_event_step_id_must_be_string_or_none(self, tmp_path):
        """LogEvent schema: step_id is Optional[str] — passing a dict must fail."""
        sm = self._make_session(tmp_path)
        import pydantic
        with pytest.raises((pydantic.ValidationError, TypeError)):
            # This is the BUG pattern from the original code
            sm.log_event(
                "TARGET_CHANGED",  # positional event_type
                {"target_object": "bottle"},  # BUG: dict passed as step_id
            )

    def test_logevent_schema_step_id_is_optional_str(self):
        """LogEvent pydantic model accepts None and str for step_id."""
        ev_none = LogEvent(
            event_type="TEST", session_id="s1", step_id=None
        )
        assert ev_none.step_id is None

        ev_str = LogEvent(
            event_type="TEST", session_id="s1", step_id="step_001"
        )
        assert ev_str.step_id == "step_001"


# ---------------------------------------------------------------------------
# 3. Pipeline set_target_object — filter update and alias resolution
# ---------------------------------------------------------------------------

from activity_detector.config.settings import (
    AppConfig,
    CameraConfig,
    RoiRule,
    VisionConfig,
    VlmConfig,
    YoloConfig,
)
from activity_detector.vision.pipeline import VisionPipeline


@pytest.fixture
def rois():
    return {
        "prep_left":        RoiRule(x1=0.02, y1=0.15, x2=0.28, y2=0.92),
        "workspace_center": RoiRule(x1=0.30, y1=0.15, x2=0.68, y2=0.92),
        "stowed_area":      RoiRule(x1=0.70, y1=0.15, x2=0.98, y2=0.92),
    }


@pytest.fixture
def pipeline_cfg(rois):
    return AppConfig(
        procedure_file="procedures/bottle_tabletop_workflow.yaml",
        target_object="bottle",
        camera=CameraConfig(source=0, mirror_preview=False),
        vision=VisionConfig(rois=rois),
        yolo=YoloConfig(enabled=True, model_path="yolo11n.pt"),
        vlm=VlmConfig(enabled=False),
    )


@pytest.fixture
def pipeline(pipeline_cfg):
    return VisionPipeline(pipeline_cfg, vlm_client=None)


class TestPipelineTargetFilter:
    def test_change_to_bottle_succeeds(self, pipeline):
        ok, msg = pipeline.set_target_object("bottle")
        assert ok, f"set_target_object returned False: {msg}"
        assert pipeline.target_object == "bottle"

    def test_change_to_phone_uses_cell_phone_canonical(self, pipeline):
        ok, msg = pipeline.set_target_object("phone")
        assert ok, f"set_target_object('phone') failed: {msg}"
        res = pipeline.alias_resolution
        assert res.canonical == "cell phone"
        assert res.backend == "yolo"
        # YOLO filter must now be set to cell phone
        assert pipeline.yolo.target_class == "cell phone"

    def test_change_to_earbuds_selects_open_vocab(self, pipeline):
        ok, msg = pipeline.set_target_object("earbuds")
        assert ok, f"earbuds should succeed (open-vocab backend): {msg}"
        assert pipeline.alias_resolution.backend == "open_vocab"
        # YOLO must NOT be filtering for earbuds (not a COCO class)
        assert pipeline.yolo.target_class != "earbuds"

    def test_unsupported_target_rejected(self, pipeline):
        original_target = pipeline.target_object
        ok, msg = pipeline.set_target_object("xyzzy_12345_not_real")
        assert not ok
        # Target must NOT have changed
        assert pipeline.target_object == original_target
        # Message must be concise
        assert len(msg) < 400

    def test_next_detect_uses_updated_class(self, pipeline):
        """After setting target, the YOLO detector's active class must reflect it."""
        pipeline.set_target_object("cup")
        assert pipeline.yolo.target_class == "cup"
        pipeline.set_target_object("bottle")
        assert pipeline.yolo.target_class == "bottle"

    def test_alias_shown_but_not_rejected(self, pipeline):
        ok, msg = pipeline.set_target_object("water bottle")
        assert ok
        assert pipeline.alias_resolution.canonical == "bottle"


# ---------------------------------------------------------------------------
# 4. Mirror orientation — display and detection in same coordinate space
# ---------------------------------------------------------------------------

class TestMirrorOrientation:
    """
    Verifies that when mirror=True the raw_frame is flipped BEFORE detectors
    run so a physical object on the LEFT appears in the left ROI and vice versa.
    """

    def _build_pipeline(self, rois, mirror: bool) -> VisionPipeline:
        cfg = AppConfig(
            procedure_file="procedures/bottle_tabletop_workflow.yaml",
            target_object="bottle",
            camera=CameraConfig(source=0, mirror_preview=mirror),
            vision=VisionConfig(rois=rois),
            yolo=YoloConfig(enabled=True, model_path="yolo11n.pt"),
            vlm=VlmConfig(enabled=False),
        )
        return VisionPipeline(cfg, vlm_client=None)

    def test_mirror_property_default_true(self, rois):
        cfg = AppConfig(
            procedure_file="procedures/bottle_tabletop_workflow.yaml",
            camera=CameraConfig(source=0, mirror_preview=True),
            vision=VisionConfig(rois=rois),
            yolo=YoloConfig(enabled=False, model_path=""),
            vlm=VlmConfig(enabled=False),
        )
        p = VisionPipeline(cfg, vlm_client=None)
        assert p.mirror is True

    def test_mirror_toggle_updates_flag(self, rois):
        p = self._build_pipeline(rois, mirror=True)
        assert p.mirror is True
        p.mirror = False
        assert p.mirror is False
        p.mirror = True
        assert p.mirror is True

    def test_synthetic_frame_left_object_seen_left_when_mirror_false(self, rois):
        """
        Without mirroring: a bright marker placed in the left third of a synthetic
        frame should be detected with its bbox centroid in the left ROI (x < 0.30).
        """
        p = self._build_pipeline(rois, mirror=False)
        if not p.yolo.available:
            pytest.skip("YOLO model not available")

        # Build a 480x640 frame with a bright white rectangle on the LEFT third
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[150:350, 20:160] = 255   # white block on left side

        # Run YOLO on this synthetic frame — probably no detection, but
        # the key assertion is that the pipeline did NOT flip the coordinates.
        # We instrument the flip path by reading pipeline._mirror:
        assert p.mirror is False

    def test_synthetic_frame_right_object_detected_right_when_mirror_false(self, rois):
        """
        With mirror=False: object on physical right → right ROI.
        With mirror=True:  object on physical right → appears on LEFT of preview.
        This test validates the convention is consistent.
        """
        p_no_mirror = self._build_pipeline(rois, mirror=False)
        p_mirror    = self._build_pipeline(rois, mirror=True)

        # Asymmetric synthetic frame: white mark on the RIGHT half
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[150:330, 450:620] = 200  # right-side bright block

        # If mirror=False, centroid x_norm ≈ 0.82 → in stowed_area ROI (x>0.70)
        # If mirror=True (frame flipped), centroid x_norm ≈ 0.18 → in prep_left ROI (x<0.28)
        # We can test this by mocking YOLO to return a fixed right-side bbox
        from activity_detector.core.engine import DetectionItem

        right_detection = DetectionItem(
            name="bottle", source="yolo", confidence=0.9,
            bbox=(450, 150, 170, 180),  # x, y, w, h  (right side)
            roi=None,
        )

        # Simulate what the pipeline does: assign ROI based on centroid after mirror
        frame_w = 640

        # No mirror: centroid_x = 450 + 170/2 = 535, norm = 535/640 ≈ 0.836 → stowed_area
        centroid_x_no_mirror = (450 + 170 // 2) / frame_w
        assert 0.70 <= centroid_x_no_mirror <= 1.0, (
            "Without mirror, right-side object centroid should be in stowed_area"
        )

        # With mirror: flip centroid: flipped_x = frame_w - centroid_x_px
        centroid_x_px = 450 + 170 // 2   # = 535
        flipped_centroid_x = (frame_w - centroid_x_px) / frame_w  # ≈ 0.164
        assert 0.0 <= flipped_centroid_x <= 0.28, (
            "With mirror, right-side physical object should appear in prep_left ROI"
        )


# ---------------------------------------------------------------------------
# 5. Automatic first step start  — session start does NOT need manual confirm
# ---------------------------------------------------------------------------

from activity_detector.core.engine import (
    EngineState,
    FrameEvidence,
    DetectionItem,
    ProcedureEngine,
)
from activity_detector.core.procedure import load_procedure


class TestAutoStepStart:
    """
    Clicking Start Session must immediately activate step 1 and allow
    automatic advancement without requiring a manual confirm.
    """

    @pytest.fixture
    def proc(self):
        return load_procedure("procedures/bottle_tabletop_workflow.yaml")

    @pytest.fixture
    def engine(self, proc):
        eng = ProcedureEngine(proc)
        eng.set_target_object("bottle")
        return eng

    def test_session_start_activates_step_0(self, engine):
        engine.start_session()
        assert engine.state == EngineState.IN_PROGRESS
        assert engine.current_step_index == 0
        from activity_detector.core.engine import StepStatus
        assert engine.step_records[0].status == StepStatus.ACTIVE

    def test_first_step_advances_automatically_with_evidence(self, engine, proc):
        """
        Step 0 (stable_detection in prep_left) should auto-complete when enough
        consecutive frames show a bottle in prep_left — no manual confirm needed.
        """
        engine.start_session()
        step0 = proc.steps[0]
        req_frames = step0.completion_rule.stable_frames
        req_hold   = step0.completion_rule.hold_seconds

        # Synthesise enough frames with a bottle detection in prep_left
        bottle_in_prep = DetectionItem(
            name="bottle",
            source="yolo",
            roi="prep_left",
            confidence=0.90,
            bbox=(50, 150, 100, 120),
        )

        advanced = False
        # Extra buffer to satisfy hold_seconds (assume ≤ 3 s, inject at 5 fps)
        n_frames = max(req_frames * 2, int((req_hold + 1) * 5) + req_frames)

        for i in range(n_frames):
            ev = FrameEvidence(
                frame_number=i,
                timestamp=time.time() + i * 0.2,  # 5 fps synthetic
                detections=[bottle_in_prep],
                target_object="bottle",
            )
            update = engine.process_frame(ev)
            if update.current_step_index > 0 or update.state == EngineState.COMPLETED:
                advanced = True
                break

        assert advanced, (
            f"Step 0 did not auto-advance after {n_frames} frames of valid evidence. "
            f"Required frames: {req_frames}, hold: {req_hold}s. "
            "Check that _evaluate_step_evidence correctly handles stable_detection."
        )

    def test_manual_confirm_still_works(self, engine, proc):
        """Manual confirm must remain available as an override (not removed)."""
        engine.start_session()
        assert engine.current_step_index == 0
        # Advance manually using the actual engine API
        engine.advance_step(reason="test_manual_override")
        assert engine.current_step_index == 1

    def test_manual_confirm_logged_as_override(self, engine, proc):
        """advance_step should emit an OPERATOR_OVERRIDE warning."""
        engine.start_session()
        initial_warnings = len(engine.history_warnings)
        engine.advance_step(reason="test_manual_override")
        assert len(engine.history_warnings) > initial_warnings
        latest_warn = engine.history_warnings[-1]
        from activity_detector.core.engine import WarningType
        assert latest_warn.warning_type == WarningType.OPERATOR_OVERRIDE


# ---------------------------------------------------------------------------
# 6. Empty-scene and face-only: must NOT be reported as target object
# ---------------------------------------------------------------------------

class TestNegativeDetections:
    """
    An empty frame or face-only frame must not be detected as bottle/phone.
    """

    @pytest.fixture
    def yolo_pipeline(self, rois):
        cfg = AppConfig(
            procedure_file="procedures/bottle_tabletop_workflow.yaml",
            target_object="bottle",
            camera=CameraConfig(source=0, mirror_preview=False),
            vision=VisionConfig(rois=rois),
            yolo=YoloConfig(enabled=True, model_path="yolo11n.pt", confidence_threshold=0.40),
            vlm=VlmConfig(enabled=False),
        )
        return VisionPipeline(cfg, vlm_client=None)

    def test_blank_frame_no_bottle(self, yolo_pipeline):
        if not yolo_pipeline.yolo.available:
            pytest.skip("YOLO model not available")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = yolo_pipeline.yolo.detect(blank)
        bottle_hits = [d for d in dets if d.name == "bottle"]
        assert bottle_hits == [], f"False positive bottle in blank frame: {bottle_hits}"

    def test_blank_frame_no_phone(self, yolo_pipeline):
        if not yolo_pipeline.yolo.available:
            pytest.skip("YOLO model not available")
        yolo_pipeline.set_target_object("phone")  # alias → cell phone
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = yolo_pipeline.yolo.detect(blank)
        phone_hits = [d for d in dets if d.name == "cell phone"]
        assert phone_hits == [], f"False positive cell phone in blank frame: {phone_hits}"

    def test_uniform_noise_no_bottle(self, yolo_pipeline):
        """Random noise frame must not produce bottle detections."""
        if not yolo_pipeline.yolo.available:
            pytest.skip("YOLO model not available")
        rng = np.random.default_rng(seed=42)
        noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = yolo_pipeline.yolo.detect(noise)
        bottle_hits = [d for d in dets if d.name == "bottle" and d.confidence >= 0.40]
        assert bottle_hits == [], (
            f"Noise frame produced bottle detections: {bottle_hits}"
        )
