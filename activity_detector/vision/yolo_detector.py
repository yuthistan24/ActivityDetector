"""Primary YOLO-based object detector using pretrained COCO weights (YOLO11n/YOLOv8n).

This is the authoritative detector for the application.  The HSV-based
DeterministicDetector remains available as a supporting cue source, but it
never overrides YOLO evidence and must not be presented as equivalent in
reliability.

Key design decisions
--------------------
* `supported_classes` exposes the exact names the loaded model knows.
* `set_target_class` filters subsequent frames to only that class; changing it
  immediately takes effect on the next `detect()` call.
* If the model file is missing or ultralytics is not installed the detector
  marks itself `available = False` and returns [] — callers must check this
  flag rather than silently fall back to colour detection.
* Confidence threshold is configurable and validated on construction.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from activity_detector.config.settings import RoiRule, YoloConfig
from activity_detector.core.engine import DetectionItem
from activity_detector.vision.detector import BaseDetector

logger = logging.getLogger("activity_detector.yolo")

# COCO class names built into YOLO11n / YOLOv8 – used as the fallback labels
# list when the model has not been loaded yet.
COCO80_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class YoloDetector(BaseDetector):
    """Primary YOLO object detector.

    Attributes
    ----------
    available : bool
        True only when the model is loaded and ready.
    status_message : str
        Human-readable reason why the detector is unavailable (or "OK").
    supported_classes : List[str]
        Exact class names the loaded model knows, sorted alphabetically.
        Empty list when the model is not loaded.
    """

    def __init__(self, config: YoloConfig, rois: Dict[str, RoiRule]) -> None:
        self.config = config
        self.rois = rois
        self.available: bool = False
        self.status_message: str = ""
        self._model = None
        self._names: Dict[int, str] = {}
        self._target_class: Optional[str] = None  # None means "all classes"
        self._target_class_id: Optional[int] = None
        self._last_inference_time: float = 0.0
        self._inference_latency_ms: float = 0.0

        if not self.config.enabled:
            self.status_message = "YOLO detector disabled in configuration."
            logger.info(self.status_message)
            return

        model_path = self.config.model_path or "yolo11n.pt"
        model_file = Path(model_path)

        # If a bare filename is given, also check the workspace root
        if not model_file.exists() and not model_file.is_absolute():
            candidates = [
                model_file,
                Path("yolo11n.pt"),
                Path("models") / model_file.name,
            ]
            for c in candidates:
                if c.exists():
                    model_file = c
                    break

        try:
            from ultralytics import YOLO  # type: ignore

            logger.info(f"Loading YOLO model from: {model_file}")
            self._model = YOLO(str(model_file))
            self._names = self._model.names  # dict[int, str]
            self.available = True
            self.status_message = (
                f"YOLO11n loaded ({model_file.name}) — "
                f"{len(self._names)} COCO classes available"
            )
            logger.info(self.status_message)
        except FileNotFoundError:
            self.status_message = (
                f"YOLO model file not found: '{model_file}'. "
                "Run `python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\"` "
                "to download it."
            )
            logger.warning(self.status_message)
        except ImportError:
            self.status_message = (
                "ultralytics package not installed. "
                "Run `pip install ultralytics` and restart."
            )
            logger.error(self.status_message)
        except Exception as exc:
            self.status_message = f"Failed to initialise YOLO model: {exc}"
            logger.error(self.status_message)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def supported_classes(self) -> List[str]:
        """Sorted list of class names the loaded model supports."""
        if self._names:
            return sorted(self._names.values())
        return []

    @property
    def supported_classes_set(self) -> Set[str]:
        """Set of class names for O(1) lookup."""
        return set(self._names.values()) if self._names else set()

    def is_class_supported(self, class_name: str) -> bool:
        """Returns True if *class_name* is in the loaded model's vocabulary."""
        return class_name.strip().lower() in {
            n.lower() for n in self.supported_classes_set
        }

    def set_target_class(self, class_name: str) -> bool:
        """
        Sets the class filter for subsequent detect() calls.

        Parameters
        ----------
        class_name : str
            A COCO class name, e.g. ``"bottle"``.  Pass ``""`` or ``None``
            to remove the filter and detect all classes.

        Returns
        -------
        bool
            True if the class is supported (or filter was cleared).
            False if the name is not in the model vocabulary.
        """
        if not class_name or class_name.strip() == "":
            self._target_class = None
            self._target_class_id = None
            logger.info("YOLO class filter cleared — detecting all classes.")
            return True

        name_lower = class_name.strip().lower()
        for cls_id, cls_name in self._names.items():
            if cls_name.lower() == name_lower:
                self._target_class = cls_name  # use canonical spelling
                self._target_class_id = cls_id
                logger.info(
                    f"YOLO target class set to '{cls_name}' (id={cls_id})"
                )
                return True

        logger.warning(
            f"Class '{class_name}' is NOT in the loaded model's vocabulary. "
            f"Supported: {self.supported_classes}"
        )
        return False

    @property
    def target_class(self) -> Optional[str]:
        """Currently active class filter, or None if all classes are detected."""
        return self._target_class

    @property
    def inference_latency_ms(self) -> float:
        """Wall-clock latency of the most recent inference call (milliseconds)."""
        return self._inference_latency_ms

    # ------------------------------------------------------------------
    # BaseDetector implementation
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[DetectionItem]:
        """
        Runs YOLO on *frame* and returns DetectionItem list.

        Only detections matching the current target class filter are returned.
        Each DetectionItem has ``source="yolo"`` and a bounding box in
        (x, y, w, h) pixel format.
        """
        if not self.available or self._model is None or frame is None:
            return []

        h, w = frame.shape[:2]
        detections: List[DetectionItem] = []
        t0 = time.perf_counter()

        try:
            # Build class-filter kwarg (ultralytics ≥ 8.1 supports `classes=`)
            kwargs: dict = {"conf": self.config.confidence_threshold, "verbose": False}
            if self._target_class_id is not None:
                kwargs["classes"] = [self._target_class_id]

            results = self._model(frame, **kwargs)

            self._inference_latency_ms = (time.perf_counter() - t0) * 1000.0
            self._last_inference_time = time.time()

            if not results:
                return []

            res = results[0]
            boxes = res.boxes

            # Pre-compute pixel ROI rectangles
            pixel_rois: Dict[str, Tuple[int, int, int, int]] = {}
            for r_name, r_rule in self.rois.items():
                pixel_rois[r_name] = (
                    int(r_rule.x1 * w),
                    int(r_rule.y1 * h),
                    int(r_rule.x2 * w),
                    int(r_rule.y2 * h),
                )

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = res.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                cx = x1 + bw // 2
                cy = y1 + bh // 2

                # ROI containment via centroid
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
                        confidence=round(conf, 3),
                        bbox=(int(x1), int(y1), int(bw), int(bh)),
                        area=int(bw * bh),
                        attributes={
                            "class_id": cls_id,
                            "centroid": (cx, cy),
                            "latency_ms": round(self._inference_latency_ms, 1),
                        },
                    )
                )

        except Exception as exc:
            logger.error(f"YOLO inference error: {exc}")
            self._inference_latency_ms = (time.perf_counter() - t0) * 1000.0

        return detections
