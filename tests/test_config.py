"""Unit tests for configuration schema and loading."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from activity_detector.config.settings import (
    AppConfig,
    RoiRule,
    get_default_config,
    load_config,
    save_config,
)


def test_default_config_validity():
    cfg = get_default_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.camera.fps == 30
    assert cfg.vision.min_contour_area > 0
    assert "workspace_center" in cfg.vision.rois
    assert "stowed_area" in cfg.vision.rois
    assert cfg.target_object == "notebook"


def test_roi_rule_validation():
    # Valid ROI
    roi = RoiRule(x1=0.1, y1=0.1, x2=0.5, y2=0.5)
    assert roi.x1 == 0.1
    assert roi.x2 == 0.5

    # Invalid ROI where x2 <= x1
    with pytest.raises(ValidationError):
        RoiRule(x1=0.6, y1=0.1, x2=0.2, y2=0.5)

    # Invalid ROI where y2 <= y1
    with pytest.raises(ValidationError):
        RoiRule(x1=0.1, y1=0.8, x2=0.5, y2=0.2)


def test_save_and_load_config(tmp_path: Path):
    cfg = get_default_config()
    cfg.camera.width = 1920
    cfg.camera.height = 1080
    cfg.audio.volume = 0.5

    test_file = tmp_path / "custom_config.yaml"
    save_config(cfg, test_file)
    assert test_file.exists()

    loaded = load_config(test_file)
    assert loaded.camera.width == 1920
    assert loaded.camera.height == 1080
    assert loaded.audio.volume == 0.5
