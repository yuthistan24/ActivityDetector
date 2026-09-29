"""Optional local YOLO detector integration for people and standard objects."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from activity_detector.config.settings import RoiRule, YoloConfig
from activity_detector.core.engine import DetectionItem
from activity_detector.vision.detector import BaseDetector

logger = logging.getLogger("activity_detector.yolo")


class YoloDetector(BaseDetector):
    """Integrates optional local YOLO models if weights are configured and present."""

    def __init__(self, config: YoloConfig, rois: Dict[str, RoiRule]) -> None:
        self.config = config
        self.rois = rois
        self.available: bool = False
        self.status_message: str = ""
        self._model = None

        if not self.config.enabled:
            self.status_message = "YOLO detector disabled in configuration."
            return

        if not self.config.model_path:
            self.status_message = "YOLO enabled but no model_path specified."
            logger.info(self.status_message)
            return

        model_file = Path(self.config.model_path)
        if not model_file.exists():
            self.status_message = f"YOLO model file not found: {model_file}"
            logger.warning(self.status_message)
            return

        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(str(model_file))
            self.available = True
            self.status_message = f"YOLO model loaded: {model_file.name}"
            logger.info(self.status_message)
        except Exception as e:
            self.available = False
            self.status_message = f"Failed to initialize YOLO model: {e}"
            logger.error(self.status_message)

    def detect(self, frame: np.ndarray) -> List[DetectionItem]:
        """Runs YOLO detection if model is loaded."""
        if not self.available or self._model is None or frame is None:
            return []

        h, w = frame.shape[:2]
        detections: List[DetectionItem] = []

        try:
            results = self._model(
                frame,
                conf=self.config.confidence_threshold,
                verbose=False
            )
            if not results:
                return []

            res = results[0]
            boxes = res.boxes

            # Calculate pixel ROIs
            pixel_rois: Dict[str, Tuple[int, int, int, int]] = {}
            for r_name, r_rule in self.rois.items():
                pixel_rois[r_name] = (
                    int(r_rule.x1 * w),
                    int(r_rule.y1 * h),
                    int(r_rule.x2 * w),
                    int(r_rule.y2 * h)
                )

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = res.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                bw = x2 - x1
                bh = y2 - y1
                cx = x1 + bw // 2
                cy = y1 + bh // 2

                # Check ROI containment
                detected_roi: Optional[str] = None
                for r_name, (rx1, ry1, rx2, ry2) in pixel_rois.items():
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        detected_roi = r_name
                        break

                detections.append(
                    DetectionItem(
                        name=cls_name,
                        source="yolo",
                        roi=detected_roi,
                        confidence=round(conf, 2),
                        bbox=(int(x1), int(y1), int(bw), int(bh)),
                        area=int(bw * bh),
                        attributes={"class_id": cls_id}
                    )
                )
        except Exception as e:
            logger.error(f"Error during YOLO inference: {e}")

        return detections
