"""Integrated computer vision pipeline coordinating camera, generic object tracking, and local VLM."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from activity_detector.config.settings import AppConfig, RoiRule
from activity_detector.core.engine import DetectionItem, EngineState, EngineUpdate, FrameEvidence
from activity_detector.vision.camera import CameraManager
from activity_detector.vision.detector import DeterministicDetector
from activity_detector.vision.vlm import BaseVlmClient, OllamaVlmClient, VlmInterpretation
from activity_detector.vision.yolo_detector import YoloDetector

logger = logging.getLogger("activity_detector.pipeline")


# Visual color constants for drawing tabletop ROIs
COLOR_ROIS: Dict[str, Tuple[int, int, int]] = {
    "workspace_center": (240, 160, 50),   # Cyan / Blue
    "stowed_area": (80, 200, 120),        # Vibrant Green
    "prep_left": (40, 180, 240),          # Amber / Orange
}

STATE_COLORS: Dict[EngineState, Tuple[int, int, int]] = {
    EngineState.IDLE: (140, 140, 140),
    EngineState.IN_PROGRESS: (240, 160, 50),       # Cyan / Blue
    EngineState.NEEDS_ATTENTION: (50, 50, 230),    # Bright Red
    EngineState.UNCERTAIN: (40, 160, 240),         # Amber / Orange
    EngineState.COMPLETED: (70, 210, 100),         # Vibrant Green
}


class VisionPipeline:
    """Coordinating engine for live video analysis, object detection, and visual HUD drawing."""

    def __init__(self, config: AppConfig, vlm_client: Optional[BaseVlmClient] = None) -> None:
        self.config = config
        self.target_object: str = getattr(config, "target_object", "notebook")
        self.camera = CameraManager(config.camera)
        self.detector = DeterministicDetector(config.vision)
        self.yolo = YoloDetector(config.yolo, config.vision.rois)

        if vlm_client is not None:
            self.vlm = vlm_client
        elif config.vlm.enabled:
            self.vlm = OllamaVlmClient(config.vlm)
        else:
            self.vlm = None

        self._frame_count: int = 0
        self._last_vlm_interpretation: Optional[VlmInterpretation] = None

    def set_target_object(self, object_name: str) -> None:
        """Dynamically updates the target object being monitored."""
        self.target_object = object_name.strip() or "notebook"
        logger.info(f"Vision pipeline target object set to: '{self.target_object}'")

    def start(self) -> bool:
        """Starts video capture and background pipelines."""
        return self.camera.start()

    def stop(self) -> None:
        """Shuts down camera, YOLO, and VLM background threads cleanly."""
        self.camera.stop()
        if self.vlm:
            self.vlm.stop()

    def process_next_frame(self, current_engine_update: Optional[EngineUpdate] = None) -> Tuple[np.ndarray, FrameEvidence]:
        """Pulls latest frame, runs local detectors, samples VLM, and draws HUD annotations."""
        has_frame, raw_frame, fps = self.camera.get_frame()
        self._frame_count += 1
        now = time.time()

        if raw_frame is None:
            raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 1. Deterministic Rule-based Detection (supporting cues)
        deterministic_detections = self.detector.detect(raw_frame)

        # 2. Optional YOLO Object Detection
        yolo_detections = self.yolo.detect(raw_frame) if self.yolo.available else []
        all_detections = deterministic_detections + yolo_detections

        # 3. Sample VLM for Generic Object State
        vlm_summary = None
        vlm_conf = None
        vlm_uncertain = False
        vlm_state: Dict[str, Any] = {}
        vlm_sample_id: Optional[str] = None

        if self.vlm and current_engine_update and current_engine_update.current_step:
            step = current_engine_update.current_step
            # Submit sample for background inference
            self.vlm.submit_sample(
                frame=raw_frame,
                target_object=self.target_object,
                step_name=step.name,
                step_instruction=step.instruction,
                expected_state=step.expected_evidence.expected_state
            )

            # Retrieve latest interpretation if available
            interp = self.vlm.get_latest_interpretation()
            if interp:
                self._last_vlm_interpretation = interp
                vlm_summary = interp.schema_data.object_description or interp.schema_data.reasoning
                vlm_conf = interp.schema_data.confidence
                vlm_uncertain = interp.schema_data.is_uncertain
                vlm_state = interp.schema_data.model_dump()
                vlm_sample_id = str(interp.timestamp)

        # 4. Construct FrameEvidence
        evidence = FrameEvidence(
            frame_number=self._frame_count,
            timestamp=now,
            detections=all_detections,
            target_object=self.target_object,
            vlm_state=vlm_state,
            vlm_summary=vlm_summary,
            vlm_confidence=vlm_conf,
            vlm_uncertain=vlm_uncertain,
            vlm_sample_id=vlm_sample_id,
        )

        # 5. Draw HUD Overlays onto frame copy
        annotated_frame = self._draw_hud(raw_frame, evidence, current_engine_update, fps)
        return annotated_frame, evidence

    def _draw_hud(
        self,
        frame: np.ndarray,
        evidence: FrameEvidence,
        engine_update: Optional[EngineUpdate],
        fps: float,
    ) -> np.ndarray:
        """Renders ROIs, detection bounding boxes, and operator status banner."""
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Draw ROIs
        for roi_name, roi_rule in self.config.vision.rois.items():
            rx1 = int(roi_rule.x1 * w)
            ry1 = int(roi_rule.y1 * h)
            rx2 = int(roi_rule.x2 * w)
            ry2 = int(roi_rule.y2 * h)

            roi_color = COLOR_ROIS.get(roi_name, (200, 200, 200))
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), roi_color, 1, cv2.LINE_AA)

            # ROI Header Tag
            label = roi_name.replace("_", " ").title()
            cv2.rectangle(canvas, (rx1, ry1 - 18), (rx1 + len(label) * 8 + 10, ry1), roi_color, -1)
            cv2.putText(canvas, label, (rx1 + 5, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)

        # Draw Supporting Detected Object Bounding Boxes (if any)
        for item in evidence.detections:
            if not item.bbox:
                continue
            bx, by, bw, bh = item.bbox
            tag_color = (0, 220, 255) if item.source == "color_roi" else (255, 180, 0)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), tag_color, 2, cv2.LINE_AA)

            src_tag = "RULE" if item.source == "color_roi" else "YOLO"
            roi_tag = f" [{item.roi}]" if item.roi else ""
            label_text = f"[{src_tag}] {item.name}: {int(item.confidence * 100)}%{roi_tag}"

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (bx, max(0, by - th - 6)), (bx + tw + 6, by), tag_color, -1)
            cv2.putText(canvas, label_text, (bx + 3, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

        # Draw Top HUD Banner
        self._render_top_banner(canvas, engine_update, fps, evidence)
        return canvas

    def _render_top_banner(
        self,
        canvas: np.ndarray,
        update: Optional[EngineUpdate],
        fps: float,
        evidence: FrameEvidence,
    ) -> None:
        """Renders top status banner with target object, step status, and telemetry."""
        w = canvas.shape[1]
        state = update.state if update else EngineState.IDLE
        banner_color = STATE_COLORS.get(state, (100, 100, 100))

        # Top banner background bar (55px high)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 22, 28), -1)
        cv2.rectangle(overlay, (0, 52), (w, 55), banner_color, -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        # Status badge chip
        state_str = state.value.replace("_", " ").upper()
        (sw, sh), _ = cv2.getTextSize(state_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        chip_x, chip_y = 15, 12
        cv2.rectangle(canvas, (chip_x, chip_y), (chip_x + sw + 16, chip_y + 26), banner_color, -1)
        cv2.putText(canvas, state_str, (chip_x + 8, chip_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Step and Target Object Information
        step_text = f"Target: [{self.target_object}] | Idle"
        if update and update.current_step:
            step_text = f"Target: [{self.target_object}] | Step {update.current_step.order}: {update.current_step.name}"
        elif update and state == EngineState.COMPLETED:
            step_text = f"Target: [{self.target_object}] | Procedure Completed"

        cv2.putText(canvas, step_text, (chip_x + sw + 25, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)

        # Stability bar & sample count (if in progress)
        if update and update.state == EngineState.IN_PROGRESS and update.current_step:
            gauge_w = 120
            gauge_h = 10
            gx = chip_x + sw + 25
            gy = 37
            ratio = max(0.0, min(1.0, update.stability_ratio))
            fill_w = int(gauge_w * ratio)

            cv2.rectangle(canvas, (gx, gy), (gx + gauge_w, gy + gauge_h), (50, 55, 65), -1)
            cv2.rectangle(canvas, (gx, gy), (gx + fill_w, gy + gauge_h), (70, 210, 100), -1)
            cv2.rectangle(canvas, (gx, gy), (gx + gauge_w, gy + gauge_h), (100, 110, 125), 1)

            sample_info = f"Stability: {int(ratio * 100)}% (Samples: {update.vlm_sample_count}/{update.required_vlm_samples})"
            cv2.putText(canvas, sample_info, (gx + gauge_w + 8, gy + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        # Right side: FPS & VLM Physical State
        fps_text = f"FPS: {fps:4.1f}"
        cv2.putText(canvas, fps_text, (w - 240, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, cv2.LINE_AA)

        # VLM State Pill
        vlm_badge = "VLM: OFF"
        vlm_color = (120, 120, 120)
        if self.vlm:
            if self._last_vlm_interpretation:
                conf = int(self._last_vlm_interpretation.schema_data.confidence * 100)
                st = self._last_vlm_interpretation.schema_data
                oc = st.open_or_closed if st.open_or_closed != "unknown" else ""
                loc = st.location if st.location != "unknown" else ""
                state_tags = " | ".join(filter(None, [oc, loc]))
                vlm_badge = f"VLM: {conf}% [{state_tags or 'sampling'}]"
                vlm_color = (70, 210, 100) if not st.is_uncertain else (40, 160, 240)
            else:
                vlm_badge = "VLM: Sampling image..."
                vlm_color = (240, 180, 50)
        cv2.putText(canvas, vlm_badge, (w - 240, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, vlm_color, 1, cv2.LINE_AA)
