"""Target alias resolution and open-vocabulary detection backend registry.

This module provides:
1. ``COCO_ALIASES``   – maps common user names to canonical COCO class names.
2. ``OPEN_VOCAB_QUERIES`` – maps names outside COCO to text queries for an optional
   open-vocabulary backend (e.g. Grounding DINO, YOLO-World).
3. ``resolve_target``  – normalises a user string to (canonical_name, backend, alias_used).
4. ``OpenVocabDetector`` – thin wrapper around an optional Grounding DINO / YOLO-World
   model that returns [] and a clear status message when weights are absent.

Design constraints
------------------
* No cloud inference.  All model loading is local.
* No silent download of large weights.  If weights are absent, the detector
  reports UNAVAILABLE with exact setup instructions.
* YOLO11n (COCO 80 classes) remains the primary detector.  This module only
  adds aliases and an optional open-vocab fallback.
* Earbuds / earphones are genuinely hard to detect reliably with generic
  pretrained weights.  The UI must communicate this honestly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("activity_detector.alias")

# ---------------------------------------------------------------------------
# COCO alias map  (user label → canonical YOLO11n class name)
# ---------------------------------------------------------------------------

COCO_ALIASES: dict[str, str] = {
    # phones
    "phone":          "cell phone",
    "mobile":         "cell phone",
    "mobile phone":   "cell phone",
    "smartphone":     "cell phone",
    "cellphone":      "cell phone",
    "iphone":         "cell phone",
    "android phone":  "cell phone",
    # laptop
    "notebook pc":    "laptop",
    "macbook":        "laptop",
    "computer":       "laptop",
    # common drinks containers
    "water bottle":   "bottle",
    "plastic bottle": "bottle",
    # books
    "notebook":       "book",
    "binder":         "book",
    # food / drink containers
    "mug":            "cup",
    # furniture
    "sofa":           "couch",
    "desk":           "dining table",
    # vehicles
    "motorbike":      "motorcycle",
    "van":            "truck",
}

# ---------------------------------------------------------------------------
# Open-vocabulary queries  (user label → text prompt for open-vocab backend)
# Items here are NOT in COCO 80 and require an optional open-vocab model.
# ---------------------------------------------------------------------------

OPEN_VOCAB_QUERIES: dict[str, str] = {
    "earbud":      "earbud",
    "earbuds":     "earbud",
    "earphone":    "earphone",
    "earphones":   "earphone",
    "headphone":   "headphone",
    "headphones":  "headphone",
    "airpod":      "earbud",
    "airpods":     "earbud",
    "tws":         "true wireless earbud",
    "pen":         "pen",
    "pencil":      "pencil",
    "glasses":     "glasses",
    "sunglasses":  "sunglasses",
    "mask":        "face mask",
    "face mask":   "face mask",
    "glove":       "glove",
    "gloves":      "glove",
    "syringe":     "syringe",
    "key":         "key",
    "keys":        "key",
    "wallet":      "wallet",
    "watch":       "wristwatch",
}

# COCO 80 canonical class names (ground truth from the loaded model)
COCO80: frozenset[str] = frozenset([
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
])


# ---------------------------------------------------------------------------
# Resolution result
# ---------------------------------------------------------------------------

class TargetResolution:
    """Result of resolving a user-supplied target name."""

    __slots__ = ("user_label", "canonical", "backend", "alias_used", "query", "note")

    def __init__(
        self,
        user_label: str,
        canonical: str,
        backend: str,          # "yolo" | "open_vocab" | "unsupported"
        alias_used: bool,
        query: str = "",
        note: str = "",
    ) -> None:
        self.user_label = user_label
        self.canonical = canonical
        self.backend = backend
        self.alias_used = alias_used
        self.query = query or canonical
        self.note = note

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TargetResolution(user={self.user_label!r}, canonical={self.canonical!r}, "
            f"backend={self.backend!r}, alias={self.alias_used})"
        )


def resolve_target(user_label: str) -> TargetResolution:
    """
    Normalises *user_label* to a canonical detector class and backend.

    Priority order:
    1. Exact match in COCO 80  → backend ``"yolo"``
    2. Match in COCO_ALIASES   → canonical COCO name, backend ``"yolo"``
    3. Match in OPEN_VOCAB_QUERIES → text query, backend ``"open_vocab"``
    4. Case-insensitive partial COCO match → backend ``"yolo"``
    5. Unknown                 → backend ``"unsupported"``
    """
    name = user_label.strip().lower()
    if not name:
        return TargetResolution(
            user_label=user_label, canonical="", backend="unsupported",
            alias_used=False, note="Empty target name."
        )

    # 1. Direct COCO match
    for coco in COCO80:
        if coco.lower() == name:
            return TargetResolution(
                user_label=user_label, canonical=coco, backend="yolo", alias_used=False
            )

    # 2. Alias → COCO
    if name in COCO_ALIASES:
        canonical = COCO_ALIASES[name]
        return TargetResolution(
            user_label=user_label,
            canonical=canonical,
            backend="yolo",
            alias_used=True,
            note=f"'{user_label}' mapped to COCO class '{canonical}'",
        )

    # 3. Open-vocabulary query  (checked BEFORE partial COCO match)
    if name in OPEN_VOCAB_QUERIES:
        query = OPEN_VOCAB_QUERIES[name]
        return TargetResolution(
            user_label=user_label,
            canonical=name,
            backend="open_vocab",
            alias_used=False,
            query=query,
            note=(
                f"'{user_label}' is not a COCO class. "
                "Requires optional open-vocabulary detector (Grounding DINO / YOLO-World). "
                "Small/occluded items like earbuds may need close-up, well-lit view."
            ),
        )

    # 4. Partial COCO match (e.g. user typos like "bottl")
    for coco in COCO80:
        if name in coco.lower() or coco.lower() in name:
            return TargetResolution(
                user_label=user_label,
                canonical=coco,
                backend="yolo",
                alias_used=True,
                note=f"'{user_label}' partially matched COCO class '{coco}'",
            )

    # 5. Completely unknown
    suggestions = ", ".join(sorted(COCO_ALIASES.keys())[:10])
    return TargetResolution(
        user_label=user_label,
        canonical="",
        backend="unsupported",
        alias_used=False,
        note=(
            f"'{user_label}' is not recognised by YOLO11n (80 COCO classes) "
            f"and is not in the alias or open-vocabulary list. "
            f"Try one of: {suggestions}, …  "
            f"Open the target selector for the full class list."
        ),
    )


# ---------------------------------------------------------------------------
# Open-vocabulary detector stub
# ---------------------------------------------------------------------------

class OpenVocabDetector:
    """
    Optional Grounding DINO / YOLO-World wrapper for non-COCO items.

    If neither weights file is present this detector marks itself
    ``available = False`` and returns [] — callers must check this
    flag rather than silently fall back.

    Setup instructions (printed once when unavailable):
        pip install groundingdino-py
        # download weights:
        # wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
        # wget https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

    YOLO-World (lighter, faster):
        pip install ultralytics  # already installed
        # weights: yolov8s-worldv2.pt (27 MB) — can be downloaded manually from
        #   https://github.com/ultralytics/assets/releases/
    """

    SETUP_MESSAGE = (
        "Open-vocabulary detector is NOT available.\n"
        "To enable detection of non-COCO items (e.g. earbuds, glasses, pens):\n"
        "\n"
        "  Option A — YOLO-World (recommended, ~27 MB):\n"
        "    Download yolov8s-worldv2.pt from:\n"
        "      https://github.com/ultralytics/assets/releases\n"
        "    Place it in the project root or set open_vocab.model_path in config.\n"
        "\n"
        "  Option B — Grounding DINO (~700 MB):\n"
        "    pip install groundingdino-py\n"
        "    Download groundingdino_swint_ogc.pth from:\n"
        "      https://github.com/IDEA-Research/GroundingDINO/releases\n"
        "\n"
        "No cloud inference is used. Do NOT run with --download-weights; "
        "place files manually to avoid surprise downloads.\n"
        "Note: Small/occluded objects like earbuds may still be unreliable "
        "with generic pretrained weights."
    )

    def __init__(self, model_path: str = "") -> None:
        self.available = False
        self.model_path = model_path
        self.status_message = ""
        self._model = None
        self._backend = "none"

        # Try YOLO-World first (already has ultralytics)
        self._try_yolo_world(model_path)

    def _try_yolo_world(self, model_path: str) -> None:
        """Attempt to load a YOLO-World model from *model_path*."""
        candidates = [
            model_path,
            "yolov8s-worldv2.pt",
            "yolo-world.pt",
        ]
        for path_str in candidates:
            if not path_str:
                continue
            p = Path(path_str)
            if p.exists() and p.stat().st_size > 1_000_000:
                try:
                    from ultralytics import YOLO  # type: ignore
                    m = YOLO(str(p))
                    self._model = m
                    self._backend = "yolo_world"
                    self.available = True
                    self.status_message = f"YOLO-World loaded from {p.name}"
                    logger.info(self.status_message)
                    return
                except Exception as exc:
                    logger.debug(f"YOLO-World load failed ({p}): {exc}")

        # Not available
        self.status_message = self.SETUP_MESSAGE
        logger.info("Open-vocabulary detector: not available (weights absent).")

    def detect(
        self, frame: np.ndarray, text_query: str, confidence_threshold: float = 0.25
    ) -> list:
        """Returns DetectionItems or [] if unavailable."""
        if not self.available or self._model is None:
            return []
        try:
            self._model.set_classes([text_query])
            results = self._model(frame, conf=confidence_threshold, verbose=False)
            from activity_detector.core.engine import DetectionItem
            detections = []
            if results:
                res = results[0]
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    detections.append(
                        DetectionItem(
                            name=text_query,
                            source="open_vocab",
                            confidence=round(conf, 3),
                            bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                            area=int((x2 - x1) * (y2 - y1)),
                        )
                    )
            return detections
        except Exception as exc:
            logger.error(f"Open-vocab inference error: {exc}")
            return []
