"""Vision pipeline: camera → YOLO detector (primary) → optional VLM → HUD.

Architecture
------------
* YOLO11n is the PRIMARY evidence source.  HSV colour detection is a
  supporting cue only and is never used to assert object identity.
* Target class changes propagate synchronously to the YOLO filter on the
  next frame; no stale filter persists across calls.
* Stale-detection tracking: if the last YOLO result is older than
  ``STALE_DETECTION_SECONDS`` the HUD shows "[STALE]" and the evidence
  timestamp is flagged so the engine does not count it as current.
* VLM (Ollama) remains optional.  When enabled it provides descriptive
  context but must NEVER override YOLO's class evidence.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from activity_detector.config.settings import AppConfig, RoiRule
from activity_detector.core.engine import (
    DetectionItem,
    EngineState,
    EngineUpdate,
    FrameEvidence,
)
from activity_detector.vision.alias import OpenVocabDetector, TargetResolution, resolve_target
from activity_detector.vision.camera import CameraManager
from activity_detector.vision.detector import DeterministicDetector
from activity_detector.vision.vlm import BaseVlmClient, OllamaVlmClient, VlmInterpretation
from activity_detector.vision.yolo_detector import YoloDetector

logger = logging.getLogger("activity_detector.pipeline")

# How many seconds before a YOLO result is considered stale
STALE_DETECTION_SECONDS: float = 3.0

# HUD colour constants (BGR)
COLOR_ROI: Dict[str, Tuple[int, int, int]] = {
    "prep_left":        (40, 200, 255),   # amber
    "stowed_area":      (80, 220, 80),    # green
    "workspace_center": (240, 160, 50),   # cyan-blue
}
DEFAULT_ROI_COLOR: Tuple[int, int, int] = (180, 180, 180)

STATE_COLORS: Dict[EngineState, Tuple[int, int, int]] = {
    EngineState.IDLE:             (140, 140, 140),
    EngineState.IN_PROGRESS:      (240, 160, 50),
    EngineState.NEEDS_ATTENTION:  (50,  50, 230),
    EngineState.UNCERTAIN:        (40, 160, 240),
    EngineState.COMPLETED:        (70, 210, 100),
}


class VisionPipeline:
    """Coordinates camera, YOLO detector, optional VLM, and HUD drawing."""

    def __init__(
        self,
        config: AppConfig,
        vlm_client: Optional[BaseVlmClient] = None,
    ) -> None:
        self.config = config
        self._target_object: str = getattr(config, "target_object", "bottle")
        self._mirror: bool = getattr(config.camera, "mirror_preview", True)

        self.camera = CameraManager(config.camera)
        self.detector = DeterministicDetector(config.vision)  # colour/ROI cues
        self.yolo = YoloDetector(config.yolo, config.vision.rois)
        self.open_vocab = OpenVocabDetector()  # optional; available=False until weights present

        # Resolve initial target through alias map
        self._resolution: TargetResolution = resolve_target(self._target_object)
        self._apply_resolution(self._resolution)

        # VLM (optional)
        if vlm_client is not None:
            self.vlm = vlm_client
        elif config.vlm.enabled:
            self.vlm: Optional[BaseVlmClient] = OllamaVlmClient(config.vlm)
        else:
            self.vlm = None

        self._frame_count: int = 0
        self._last_vlm_interpretation: Optional[VlmInterpretation] = None
        self._fresh_frame_times: collections.deque = collections.deque()
        self._last_seen_frame_id: int = -1
        self._ui_fresh_fps: float = 0.0
        self._last_fresh_ui_time: float = time.time()

        # Stale-detection tracking
        self._last_yolo_detection_time: float = 0.0  # 0 = never
        self._last_yolo_had_target: bool = False

        # Target-class validation state for UI reporting
        self._target_supported: bool = True
        self._target_validation_msg: str = ""
        # Rate-limit unsupported-target warnings (avoid per-frame spam)
        self._last_unsupported_warn_time: float = 0.0

    def _apply_resolution(self, res: TargetResolution) -> None:
        """Push resolved canonical name into the appropriate detector."""
        if res.backend == "yolo" and self.yolo.available:
            ok = self.yolo.set_target_class(res.canonical)
            if not ok:
                logger.warning(
                    f"Alias resolved to '{res.canonical}' but YOLO rejected it. "
                    "Detecting all classes until fixed."
                )
        elif res.backend == "open_vocab":
            # open_vocab detector will be queried with res.query per-frame
            # Nothing to pre-set; query is embedded in res
            pass
        elif res.backend == "unsupported":
            logger.warning(f"Target '{res.user_label}' is unsupported: {res.note}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def target_object(self) -> str:
        return self._target_object

    @property
    def vlm_client(self) -> Optional[BaseVlmClient]:
        return self.vlm

    @property
    def yolo_status(self) -> str:
        """Human-readable detector status for the UI."""
        res = self._resolution
        mirror_tag = "mirror" if self._mirror else "no-mirror"

        if res and res.backend == "open_vocab":
            if self.open_vocab.available:
                return (
                    f"Open-Vocab ({self.open_vocab._backend}) | "
                    f"query: '{res.query}' | {mirror_tag}"
                )
            else:
                return (
                    f"Open-Vocab: UNAVAILABLE — weights absent | "
                    f"target '{res.user_label}' requires setup | {mirror_tag}"
                )

        if not self.yolo.available:
            return f"YOLO11n UNAVAILABLE — {self.yolo.status_message} | {mirror_tag}"

        cls = self.yolo.target_class or "(all classes)"
        alias_tag = ""
        if res and res.alias_used:
            alias_tag = f" [{res.user_label}→{cls}]"
        lat = self.yolo.inference_latency_ms
        return (
            f"YOLO11n | target: {cls}{alias_tag} | "
            f"conf≥{self.config.yolo.confidence_threshold:.2f} | "
            f"latency: {lat:.0f} ms | {mirror_tag}"
        )

    @property
    def supported_classes(self) -> List[str]:
        return self.yolo.supported_classes

    @property
    def mirror(self) -> bool:
        return self._mirror

    @mirror.setter
    def mirror(self, value: bool) -> None:
        self._mirror = value
        self.config.camera.mirror_preview = value

    @property
    def alias_resolution(self) -> TargetResolution:
        """Current target alias resolution result (for UI display)."""
        return self._resolution

    def is_target_supported(self, name: str) -> bool:
        res = resolve_target(name)
        return res.backend != "unsupported"

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def set_target_object(self, object_name: str) -> Tuple[bool, str]:
        """
        Change the detection target.  Returns (success, message).

        Uses alias resolution so "phone" → "cell phone",
        "earbuds" → open_vocab backend, etc.
        Unsupported names are rejected with a concise message (no full 80-class dump).
        """
        name = object_name.strip()
        if not name:
            return False, "Target name must not be empty."

        res = resolve_target(name)

        if res.backend == "unsupported":
            now = time.time()
            # Rate-limit identical warnings to once per 5 s
            if (now - self._last_unsupported_warn_time) > 5.0:
                logger.warning(f"Unsupported target '{name}': {res.note}")
                self._last_unsupported_warn_time = now
            self._target_supported = False
            self._target_validation_msg = res.note
            return False, res.note

        # Apply to relevant detector
        self._apply_resolution(res)

        # Store user-visible label and canonical backend label
        self._target_object = name          # what the user typed (for display)
        self._resolution = res
        self._target_supported = True
        self._target_validation_msg = ""
        self._last_vlm_interpretation = None
        self._last_yolo_detection_time = 0.0
        self._last_yolo_had_target = False

        if self.vlm:
            self.vlm.set_target_object(res.canonical or name)

        info_msg = f"Target set to '{name}'"
        if res.alias_used:
            info_msg += f" (canonical: '{res.canonical}')"
        if res.backend == "open_vocab":
            info_msg += " [open-vocab backend]"
            if not self.open_vocab.available:
                info_msg += " — weights absent, see setup instructions"
        logger.info(info_msg)
        return True, info_msg

    def start(self) -> bool:
        return self.camera.start()

    def stop(self) -> None:
        self.camera.stop()
        if self.vlm:
            self.vlm.stop()

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def process_next_frame(
        self,
        current_engine_update: Optional[EngineUpdate] = None,
    ) -> Tuple[np.ndarray, FrameEvidence]:
        """Pull latest camera frame, run detectors, build evidence, draw HUD."""
        has_frame, raw_frame, cam_fps, frame_id, frame_time = (
            self.camera.get_frame_packet()
        )
        self._frame_count += 1
        now = time.time()

        if raw_frame is None:
            raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # ── Mirror / orientation ──────────────────────────────────────
        # Apply BEFORE detectors so bboxes, ROIs, and display are all
        # in the same coordinate space. When mirror=True the user sees
        # left/right as physically expected for a laptop webcam.
        if self._mirror:
            raw_frame = cv2.flip(raw_frame, 1)   # horizontal flip

        # Track UI fresh-frame rate
        if frame_id > 0 and frame_id != self._last_seen_frame_id:
            self._last_seen_frame_id = frame_id
            self._last_fresh_ui_time = now
            self._fresh_frame_times.append(now)

        while self._fresh_frame_times and (now - self._fresh_frame_times[0]) > 1.0:
            self._fresh_frame_times.popleft()

        if len(self._fresh_frame_times) >= 2:
            dt = self._fresh_frame_times[-1] - self._fresh_frame_times[0]
            self._ui_fresh_fps = (
                round((len(self._fresh_frame_times) - 1) / dt, 1)
                if dt > 0.05
                else float(len(self._fresh_frame_times))
            )
        elif len(self._fresh_frame_times) == 1:
            self._ui_fresh_fps = 1.0
        else:
            self._ui_fresh_fps = 0.0

        # ── 1. YOLO detection (PRIMARY) ──────────────────────────────
        yolo_detections: List[DetectionItem] = []
        canonical = self._resolution.canonical if self._resolution else self._target_object
        if self.yolo.available and self._resolution.backend == "yolo":
            yolo_detections = self.yolo.detect(raw_frame)
            target_hits = [
                d for d in yolo_detections
                if d.name.lower() == canonical.lower()
            ]
            if target_hits:
                self._last_yolo_detection_time = now
                self._last_yolo_had_target = True

        # ── 1b. Open-vocabulary detection (non-COCO targets) ─────────
        open_vocab_detections: List[DetectionItem] = []
        if self._resolution.backend == "open_vocab" and self.open_vocab.available:
            open_vocab_detections = self.open_vocab.detect(
                raw_frame,
                text_query=self._resolution.query,
                confidence_threshold=self.config.yolo.confidence_threshold,
            )
            if open_vocab_detections:
                self._last_yolo_detection_time = now
                self._last_yolo_had_target = True

        # ── 2. HSV colour cues (SUPPORTING ONLY) ─────────────────────
        colour_detections: List[DetectionItem] = self.detector.detect(raw_frame)

        all_detections = yolo_detections + open_vocab_detections + colour_detections

        # ── 3. Stale-result flag ──────────────────────────────────────
        yolo_result_is_stale = (
            self._last_yolo_had_target
            and (now - self._last_yolo_detection_time) > STALE_DETECTION_SECONDS
        )

        # ── 4. Optional VLM (descriptive context only) ────────────────
        vlm_summary: Optional[str] = None
        vlm_conf: Optional[float] = None
        vlm_uncertain = False
        vlm_state: Dict[str, Any] = {}
        vlm_sample_id: Optional[str] = None

        is_session_active = (
            current_engine_update is not None
            and current_engine_update.state
            in (EngineState.IN_PROGRESS, EngineState.NEEDS_ATTENTION, EngineState.UNCERTAIN)
            and current_engine_update.current_step is not None
        )

        if self.vlm and not getattr(self.vlm, "is_paused", False):
            step_name = "Preview"
            step_instruction = (
                f"Describe the scene, focusing on any '{self._target_object}' visible."
            )
            expected_state: Dict[str, Any] = {"object_visible": True}

            if is_session_active and current_engine_update.current_step:
                step = current_engine_update.current_step
                step_name = step.name
                step_instruction = step.instruction
                expected_state = step.expected_evidence.expected_state

            self.vlm.submit_sample(
                frame=raw_frame,
                target_object=self._target_object,
                step_name=step_name,
                step_instruction=step_instruction,
                expected_state=expected_state,
            )

            interp = self.vlm.get_latest_interpretation()
            if interp:
                self._last_vlm_interpretation = interp
                vlm_summary = (
                    interp.schema_data.object_description
                    or interp.schema_data.reasoning
                )
                vlm_conf = interp.schema_data.confidence
                vlm_uncertain = interp.schema_data.is_uncertain
                vlm_state = interp.schema_data.model_dump()
                vlm_sample_id = str(interp.timestamp)

        if self.vlm and getattr(self.vlm, "is_degraded", False):
            vlm_uncertain = True
            reason = getattr(self.vlm, "degraded_reason", "timeout")
            vlm_summary = f"[VLM DEGRADED: {reason}]"

        # ── 5. Build FrameEvidence ───────────────────────────────────
        evidence = FrameEvidence(
            frame_number=self._frame_count,
            timestamp=now,
            detections=all_detections,
            target_object=self._target_object,
            vlm_state=vlm_state,
            vlm_summary=vlm_summary,
            vlm_confidence=vlm_conf,
            vlm_uncertain=vlm_uncertain,
            vlm_sample_id=vlm_sample_id,
        )

        # ── 6. Draw HUD ──────────────────────────────────────────────
        annotated = self._draw_hud(
            raw_frame,
            evidence,
            current_engine_update,
            self._ui_fresh_fps,
            yolo_result_is_stale,
        )
        return annotated, evidence

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_camera_diagnostics(self) -> Dict[str, Any]:
        diag = self.camera.get_diagnostics()
        diag["ui_fresh_fps"] = self._ui_fresh_fps
        return diag

    def get_ui_fresh_fps(self) -> float:
        return self._ui_fresh_fps

    def get_detector_status(self) -> Dict[str, Any]:
        """Returns a summary dict suitable for the UI status panel."""
        target_hits_now = False
        last_det_age = (
            (time.time() - self._last_yolo_detection_time)
            if self._last_yolo_detection_time
            else None
        )
        is_stale = (
            self._last_yolo_had_target
            and last_det_age is not None
            and last_det_age > STALE_DETECTION_SECONDS
        )
        return {
            "available": self.yolo.available,
            "status_message": self.yolo.status_message,
            "target_class": self._target_object,
            "target_supported": self._target_supported,
            "target_validation_msg": self._target_validation_msg,
            "supported_class_count": len(self.yolo.supported_classes),
            "inference_latency_ms": self.yolo.inference_latency_ms,
            "confidence_threshold": self.config.yolo.confidence_threshold,
            "last_detection_age": last_det_age,
            "result_is_stale": is_stale,
        }

    # ------------------------------------------------------------------
    # HUD drawing
    # ------------------------------------------------------------------

    def _draw_hud(
        self,
        frame: np.ndarray,
        evidence: FrameEvidence,
        engine_update: Optional[EngineUpdate],
        fps: float,
        stale: bool,
    ) -> np.ndarray:
        try:
            canvas = frame.copy()
        except Exception:
            canvas = frame
        h, w = canvas.shape[:2]

        # ── Draw ROI boundaries ──────────────────────────────────────
        for roi_name, roi_rule in self.config.vision.rois.items():
            rx1 = int(roi_rule.x1 * w)
            ry1 = int(roi_rule.y1 * h)
            rx2 = int(roi_rule.x2 * w)
            ry2 = int(roi_rule.y2 * h)

            col = COLOR_ROI.get(roi_name, DEFAULT_ROI_COLOR)
            thickness = 2
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), col, thickness, cv2.LINE_AA)

            label = roi_name.replace("_", " ").upper()
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (rx1, ry1 - th - 8), (rx1 + tw + 8, ry1), col, -1)
            cv2.putText(
                canvas, label, (rx1 + 4, ry1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA,
            )

        # ── Draw YOLO bounding boxes ─────────────────────────────────
        for item in evidence.detections:
            if item.source != "yolo" or not item.bbox:
                continue
            bx, by, bw, bh = item.bbox
            is_target = item.name.lower() == self._target_object.lower()
            box_col = (0, 255, 80) if is_target else (180, 180, 60)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), box_col, 2, cv2.LINE_AA)

            roi_tag = f" [{item.roi}]" if item.roi else ""
            lbl = f"{item.name}: {int(item.confidence * 100)}%{roi_tag}"
            if stale and is_target:
                lbl = f"[STALE] {lbl}"

            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                canvas, (bx, max(0, by - lh - 6)), (bx + lw + 6, by), box_col, -1
            )
            cv2.putText(
                canvas, lbl, (bx + 3, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA,
            )

        # ── Top banner ───────────────────────────────────────────────
        self._render_top_banner(canvas, engine_update, fps, evidence, stale)
        return canvas

    def _render_top_banner(
        self,
        canvas: np.ndarray,
        update: Optional[EngineUpdate],
        fps: float,
        evidence: FrameEvidence,
        stale: bool,
    ) -> None:
        w = canvas.shape[1]
        state = update.state if update else EngineState.IDLE
        banner_color = STATE_COLORS.get(state, (100, 100, 100))

        banner_h = min(55, canvas.shape[0])
        banner_roi = canvas[0:banner_h, 0:w]
        overlay = banner_roi.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 22, 28), -1)
        cv2.rectangle(
            overlay, (0, max(0, banner_h - 3)), (w, banner_h), banner_color, -1
        )
        cv2.addWeighted(overlay, 0.85, banner_roi, 0.15, 0, banner_roi)

        # State badge chip
        state_str = state.value.replace("_", " ").upper()
        (sw, sh), _ = cv2.getTextSize(state_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        chip_x, chip_y = 15, 12
        cv2.rectangle(
            canvas, (chip_x, chip_y), (chip_x + sw + 16, chip_y + 26), banner_color, -1
        )
        cv2.putText(
            canvas, state_str, (chip_x + 8, chip_y + 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Target + step info
        step_text = (
            f"Target: [{self._target_object}] | PREVIEW — Start session to begin"
        )
        if update and update.state != EngineState.IDLE and update.current_step:
            step_text = (
                f"Target: [{self._target_object}] | "
                f"Step {update.current_step.order}: {update.current_step.name}"
            )
        elif update and state == EngineState.COMPLETED:
            step_text = f"Target: [{self._target_object}] | Procedure Completed ✓"

        cv2.putText(
            canvas, step_text, (chip_x + sw + 25, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA,
        )

        # Stability bar
        if update and update.state == EngineState.IN_PROGRESS and update.current_step:
            gx = chip_x + sw + 25
            gy = 38
            gauge_w = 140
            ratio = max(0.0, min(1.0, update.stability_ratio))
            fill_w = int(gauge_w * ratio)
            cv2.rectangle(canvas, (gx, gy), (gx + gauge_w, gy + 9), (50, 55, 65), -1)
            cv2.rectangle(canvas, (gx, gy), (gx + fill_w, gy + 9), (70, 210, 100), -1)
            cv2.rectangle(canvas, (gx, gy), (gx + gauge_w, gy + 9), (100, 110, 125), 1)
            stab_lbl = f"Stability {int(ratio * 100)}%"
            cv2.putText(
                canvas, stab_lbl, (gx + gauge_w + 6, gy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA,
            )

        # Camera health (right side)
        cam_diag = self.camera.get_diagnostics()
        cam_state = cam_diag["state"].upper()
        cam_age = cam_diag["last_frame_age_seconds"]
        is_black = cam_diag.get("is_black_frame", False)

        if cam_diag["is_stale"] or cam_diag["state"] == "reconnecting":
            cam_col = (40, 160, 240)
            cam_text = f"CAM: {cam_state} ({cam_age:.1f}s)"
        elif is_black:
            cam_col = (50, 50, 230)
            cam_text = "CAM: BLACK FRAME"
        elif cam_diag["state"] == "connected":
            cam_col = (70, 210, 100)
            cam_text = f"CAM: {self._ui_fresh_fps:.1f} FPS"
        else:
            cam_col = (50, 50, 230)
            cam_text = f"CAM: {cam_state}"

        cv2.putText(
            canvas, cam_text, (w - 200, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, cam_col, 1, cv2.LINE_AA,
        )

        # YOLO / detector status
        if not self.yolo.available:
            det_text = "YOLO: UNAVAILABLE"
            det_col = (50, 50, 230)
        else:
            target_hits = [
                d for d in evidence.detections
                if d.source == "yolo"
                and d.name.lower() == self._target_object.lower()
            ]
            if stale:
                det_text = f"YOLO: [{self._target_object}] STALE"
                det_col = (40, 160, 240)
            elif target_hits:
                best = max(target_hits, key=lambda d: d.confidence)
                roi_tag = f" in {best.roi}" if best.roi else ""
                det_text = (
                    f"YOLO: [{self._target_object}]{roi_tag} "
                    f"{int(best.confidence * 100)}%"
                )
                det_col = (70, 210, 100)
            else:
                det_text = f"YOLO: No [{self._target_object}]"
                det_col = (100, 100, 220)

        cv2.putText(
            canvas, det_text, (w - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, det_col, 1, cv2.LINE_AA,
        )

        # Black-frame / stale warnings
        if is_black and cam_diag["state"] == "connected":
            cv2.rectangle(canvas, (0, 56), (w, 78), (20, 20, 180), -1)
            cv2.putText(
                canvas,
                "WARNING: BLACK FRAME — check physical lens shutter / lighting",
                (16, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255),
                1, cv2.LINE_AA,
            )
        elif cam_diag["is_stale"] and cam_diag["state"] == "connected":
            cv2.rectangle(canvas, (0, 56), (w, 78), (20, 20, 180), -1)
            cv2.putText(
                canvas,
                f"WARNING: CAMERA FEED STALE / FROZEN ({cam_age:.1f}s without new frame)",
                (16, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255),
                1, cv2.LINE_AA,
            )
