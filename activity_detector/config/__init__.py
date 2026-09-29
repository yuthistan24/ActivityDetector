"""Configuration module for ActivityDetector."""

from activity_detector.config.settings import (
    AppConfig,
    CameraConfig,
    ColorRule,
    RoiRule,
    VisionConfig,
    YoloConfig,
    VlmConfig,
    AudioConfig,
    RecordingConfig,
    StreamingConfig,
    load_config,
    save_config,
    get_default_config,
)

__all__ = [
    "AppConfig",
    "CameraConfig",
    "ColorRule",
    "RoiRule",
    "VisionConfig",
    "YoloConfig",
    "VlmConfig",
    "AudioConfig",
    "RecordingConfig",
    "StreamingConfig",
    "load_config",
    "save_config",
    "get_default_config",
]
