"""Configuration schema and loaders for ActivityDetector."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator


class CameraConfig(BaseModel):
    """Webcam / video source configuration."""
    source: Union[int, str] = Field(
        default=0,
        description="Camera index (e.g. 0) or local video file path for testing."
    )
    width: int = Field(default=1280, ge=320, le=3840)
    height: int = Field(default=720, ge=240, le=2160)
    fps: int = Field(default=30, ge=1, le=120)
    mirror_preview: bool = Field(
        default=True,
        description="Horizontally flip the preview and detection boxes (laptop webcam default)."
    )


class ColorRule(BaseModel):
    """HSV color bounds for deterministic object detection."""
    h_min: int = Field(ge=0, le=179)
    s_min: int = Field(ge=0, le=255)
    v_min: int = Field(ge=0, le=255)
    h_max: int = Field(ge=0, le=179)
    s_max: int = Field(ge=0, le=255)
    v_max: int = Field(ge=0, le=255)


class RoiRule(BaseModel):
    """Normalized Region of Interest coordinates (0.0 to 1.0)."""
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @field_validator("x2")
    @classmethod
    def validate_x2(cls, v: float, info: Any) -> float:
        x1 = info.data.get("x1", 0.0)
        if v <= x1:
            raise ValueError(f"x2 ({v}) must be greater than x1 ({x1})")
        return v

    @field_validator("y2")
    @classmethod
    def validate_y2(cls, v: float, info: Any) -> float:
        y1 = info.data.get("y1", 0.0)
        if v <= y1:
            raise ValueError(f"y2 ({v}) must be greater than y1 ({y1})")
        return v


class VisionConfig(BaseModel):
    """Deterministic vision parameters."""
    min_contour_area: int = Field(default=700, ge=50)
    rois: Dict[str, RoiRule] = Field(default_factory=dict)
    colors: Dict[str, ColorRule] = Field(default_factory=dict)


class YoloConfig(BaseModel):
    """Primary YOLO detector settings (YOLO11n / YOLOv8n COCO)."""
    enabled: bool = Field(default=True)
    model_path: str = Field(default="yolo11n.pt")
    confidence_threshold: float = Field(default=0.40, ge=0.0, le=1.0)


class VlmConfig(BaseModel):
    """Local Vision-Language Model configuration (Ollama)."""
    enabled: bool = Field(default=True)
    provider: str = Field(default="ollama")
    model: str = Field(default="gemma4:latest")
    alternative_model: str = Field(default="")
    host: str = Field(default="http://localhost:11434")
    sample_interval_seconds: float = Field(default=3.0, ge=0.1, le=60.0)
    num_ctx: int = Field(default=1024, ge=256, le=32768)
    num_gpu: Optional[int] = Field(default=None, description="None allows Ollama auto GPU detection; 0 forces CPU")
    timeout_seconds: float = Field(default=20.0, ge=2.0, le=120.0)
    image_max_dimension: int = Field(default=480, ge=240, le=1280)
    jpeg_quality: int = Field(default=75, ge=40, le=95)
    failure_cooldown_seconds: float = Field(default=5.0, ge=0.1, le=60.0)
    max_consecutive_failures: int = Field(default=3, ge=1, le=10)


class AudioConfig(BaseModel):
    """Local Text-to-Speech configuration."""
    enabled: bool = Field(default=True)
    rate: int = Field(default=170, ge=80, le=300)
    volume: float = Field(default=0.9, ge=0.0, le=1.0)
    cooldown_seconds: float = Field(default=7.0, ge=1.0, le=60.0)
    voice_index: int = Field(default=0, ge=0)


class RecordingConfig(BaseModel):
    """Local session recording configuration."""
    output_dir: str = Field(default="runs")
    video_format: str = Field(default="mp4")
    record_fps: float = Field(default=25.0, ge=5.0, le=60.0)
    auto_record_on_session_start: bool = Field(default=True)
    record_annotated: bool = Field(default=True)


class StreamingConfig(BaseModel):
    """Optional isolated network streaming configuration."""
    enabled: bool = Field(default=False)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8554, ge=1024, le=65535)
    path: str = Field(default="/video_feed")


class AppConfig(BaseModel):
    """Master application configuration."""
    procedure_file: str = Field(default="procedures/bottle_tabletop_workflow.yaml")
    target_object: str = Field(default="bottle", description="Selected COCO class to monitor")
    camera: CameraConfig = Field(default_factory=CameraConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    yolo: YoloConfig = Field(default_factory=YoloConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)


def get_default_config() -> AppConfig:
    """Returns default configuration for water bottle tabletop demo (YOLO11n)."""
    return AppConfig(
        procedure_file="procedures/bottle_tabletop_workflow.yaml",
        target_object="bottle",
        camera=CameraConfig(source=0, width=1280, height=720, fps=30, mirror_preview=True),
        vision=VisionConfig(
            min_contour_area=700,
            rois={
                "prep_left":        RoiRule(x1=0.02, y1=0.15, x2=0.28, y2=0.92),
                "workspace_center": RoiRule(x1=0.30, y1=0.15, x2=0.68, y2=0.92),
                "stowed_area":      RoiRule(x1=0.70, y1=0.15, x2=0.98, y2=0.92),
            },
            colors={
                # HSV colour cues are supporting evidence only.
                "yellow_cover": ColorRule(h_min=18, s_min=100, v_min=100, h_max=35, s_max=255, v_max=255),
                "blue_marker":  ColorRule(h_min=100, s_min=110, v_min=60, h_max=130, s_max=255, v_max=255),
            }
        ),
        yolo=YoloConfig(enabled=True, model_path="yolo11n.pt", confidence_threshold=0.40),
        vlm=VlmConfig(
            enabled=False,
            provider="ollama",
            model="gemma4:latest",
            alternative_model="",
            host="http://localhost:11434",
            sample_interval_seconds=5.0,
            num_ctx=1024,
            num_gpu=None,
            timeout_seconds=15.0,
        ),
        audio=AudioConfig(
            enabled=True,
            rate=170,
            volume=0.9,
            cooldown_seconds=7.0,
            voice_index=0
        ),
        recording=RecordingConfig(
            output_dir="runs",
            video_format="mp4",
            record_fps=25.0,
            auto_record_on_session_start=True,
            record_annotated=True
        ),
        streaming=StreamingConfig(
            enabled=False,
            host="127.0.0.1",
            port=8554,
            path="/video_feed"
        )
    )


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Loads configuration from YAML/JSON file, falling back to defaults if not found."""
    default_cfg = get_default_config()
    if not config_path:
        default_file = Path("activity_detector/config/default_config.yaml")
        if default_file.exists():
            config_path = default_file
        else:
            return default_cfg

    path = Path(config_path)
    if not path.exists():
        return default_cfg

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig.model_validate(data)


def save_config(config: AppConfig, path: Union[str, Path]) -> None:
    """Saves AppConfig as a documented YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = config.model_dump(mode="json")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)
