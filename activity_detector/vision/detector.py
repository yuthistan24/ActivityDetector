"""Deterministic computer vision detector using HSV color segmentation and ROI spatial reasoning."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from activity_detector.config.settings import ColorRule, RoiRule, VisionConfig
from activity_detector.core.engine import DetectionItem

logger = logging.getLogger("activity_detector.detector")


class BaseDetector(ABC):
    """Abstract interface for local object detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionItem]:
        """Performs detection on a BGR frame and returns detected items."""
        pass


class DeterministicDetector(BaseDetector):
    """Rule-based, explainable computer vision detector."""

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.colors = config.colors
        self.rois = config.rois
        self.min_area = config.min_contour_area

        # Morphological kernels for noise reduction
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    def detect(self, frame: np.ndarray) -> List[DetectionItem]:
        """Detects objects based on calibrated HSV color profiles and ROI containment."""
        if frame is None or frame.size == 0:
            return []

        try:
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            detections: List[DetectionItem] = []

            # Convert normalized ROIs to pixel coordinates
            pixel_rois: Dict[str, Tuple[int, int, int, int]] = {}
            for r_name, r_rule in self.rois.items():
                px1 = int(r_rule.x1 * w)
                py1 = int(r_rule.y1 * h)
                px2 = int(r_rule.x2 * w)
                py2 = int(r_rule.y2 * h)
                pixel_rois[r_name] = (px1, py1, px2, py2)

            for color_name, rule in self.colors.items():
                lower = np.array([rule.h_min, rule.s_min, rule.v_min], dtype=np.uint8)
                upper = np.array([rule.h_max, rule.s_max, rule.v_max], dtype=np.uint8)

                # Support hue wrap-around if h_min > h_max (e.g. red spanning 170-180 and 0-10)
                if rule.h_min > rule.h_max:
                    mask1 = cv2.inRange(hsv, lower, np.array([179, rule.s_max, rule.v_max], dtype=np.uint8))
                    mask2 = cv2.inRange(hsv, np.array([0, rule.s_min, rule.v_min], dtype=np.uint8), upper)
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    mask = cv2.inRange(hsv, lower, upper)

                # Apply opening then closing to eliminate speckles
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < self.min_area:
                        continue

                    x, y, bw, bh = cv2.boundingRect(cnt)
                    cx = x + bw // 2
                    cy = y + bh // 2

                    # Determine which ROI contains this object's centroid
                    detected_roi: Optional[str] = None
                    for r_name, (rx1, ry1, rx2, ry2) in pixel_rois.items():
                        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                            detected_roi = r_name
                            break

                    # Estimate confidence: scales with area up to target plateau
                    # Capped at 0.95 for deterministic vision (leaving headroom)
                    area_score = min(1.0, area / 4000.0)
                    confidence = round(0.50 + 0.45 * area_score, 2)

                    detections.append(
                        DetectionItem(
                            name=color_name,
                            source="color_roi",
                            roi=detected_roi,
                            confidence=confidence,
                            bbox=(x, y, bw, bh),
                            area=int(area),
                            attributes={"centroid": (cx, cy), "color": color_name}
                        )
                    )

            return detections
        except Exception as e:
            logger.warning(f"Detection frame processing error: {e}")
            return []
