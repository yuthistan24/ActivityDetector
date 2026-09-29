"""Unit tests for deterministic HSV and ROI computer vision detector."""

import cv2
import numpy as np
import pytest

from activity_detector.config.settings import ColorRule, RoiRule, VisionConfig
from activity_detector.vision.detector import DeterministicDetector


@pytest.fixture
def vision_config() -> VisionConfig:
    return VisionConfig(
        min_contour_area=200,
        rois={
            "workbench_center": RoiRule(x1=0.25, y1=0.25, x2=0.75, y2=0.75),
            "staging_left": RoiRule(x1=0.0, y1=0.0, x2=0.25, y2=1.0),
        },
        colors={
            "blue_test": ColorRule(h_min=100, s_min=100, v_min=100, h_max=130, s_max=255, v_max=255),
            "yellow_test": ColorRule(h_min=20, s_min=100, v_min=100, h_max=35, s_max=255, v_max=255),
        }
    )


def test_detector_empty_frame(vision_config: VisionConfig):
    detector = DeterministicDetector(vision_config)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank)
    assert len(detections) == 0


def test_detector_finds_blue_in_workbench(vision_config: VisionConfig):
    detector = DeterministicDetector(vision_config)
    # Create 480x640 black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw pure blue rectangle in center: (x=300, y=220, w=80, h=80)
    # OpenCV BGR: Blue is (255, 0, 0)
    cv2.rectangle(frame, (300, 220), (380, 300), (255, 0, 0), -1)

    detections = detector.detect(frame)
    assert len(detections) >= 1
    d = detections[0]
    assert d.name == "blue_test"
    assert d.roi == "workbench_center"
    assert d.confidence >= 0.50
    assert d.area >= 200


def test_detector_finds_yellow_in_staging(vision_config: VisionConfig):
    detector = DeterministicDetector(vision_config)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw yellow rectangle on left staging area (x=50, y=200, w=60, h=60)
    # Yellow in BGR: (0, 255, 255)
    cv2.rectangle(frame, (50, 200), (110, 260), (0, 255, 255), -1)

    detections = detector.detect(frame)
    assert len(detections) >= 1
    d = detections[0]
    assert d.name == "yellow_test"
    assert d.roi == "staging_left"
