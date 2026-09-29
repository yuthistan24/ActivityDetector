"""Preflight diagnostic and health inspection subsystem."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import List, Tuple
from pydantic import BaseModel, Field

from activity_detector.config.settings import AppConfig, load_config
from activity_detector.core.procedure import load_procedure


class CheckResult(BaseModel):
    """Result of an individual subsystem preflight check."""
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    details: str = ""


class PreflightReport(BaseModel):
    """Consolidated preflight diagnostic report."""
    all_critical_passed: bool
    checks: List[CheckResult] = Field(default_factory=list)


def run_preflight_checks(config: AppConfig) -> PreflightReport:
    """Executes end-to-end system diagnostics across hardware, models, and filesystems."""
    checks: List[CheckResult] = []

    # 1. Python Environment Check
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 9):
        checks.append(CheckResult(
            name="Python Version",
            status="PASS",
            message=f"Python {py_ver} meets runtime requirements (>= 3.9)."
        ))
    else:
        checks.append(CheckResult(
            name="Python Version",
            status="FAIL",
            message=f"Python {py_ver} is too old. Requires >= 3.9."
        ))

    # 2. Camera Access & Sustained Capture Verification
    try:
        import cv2
        src = config.camera.source
        cap = None
        backend_name = "Default"

        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            backend_name = "DirectShow"
            if not cap.isOpened():
                if cap:
                    cap.release()
                cap = cv2.VideoCapture(src)
                backend_name = "Default (MSMF/V4L2)"
        else:
            cap = cv2.VideoCapture(str(src))
            backend_name = "VideoFile"

        if cap and cap.isOpened():
            # Sustained capture check: stream for 1.0 second to verify throughput & stability
            frames_captured = 0
            failed_reads = 0
            last_w, last_h = 0, 0
            t_start = time.time()
            max_duration = 1.0
            target_frames = 25

            while (time.time() - t_start) < max_duration and frames_captured < target_frames:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames_captured += 1
                    last_h, last_w = frame.shape[:2]
                else:
                    failed_reads += 1
                time.sleep(0.01)

            elapsed = max(0.001, time.time() - t_start)
            fps_measured = round(frames_captured / elapsed, 1)
            cap.release()

            if frames_captured >= 8 and failed_reads == 0:
                checks.append(CheckResult(
                    name="Webcam Capture",
                    status="PASS",
                    message=(
                        f"Source {src} accessible ({backend_name}). Sustained capture verified: "
                        f"{frames_captured} frames in {elapsed:.1f}s ({fps_measured} FPS, {last_w}x{last_h} px)."
                    )
                ))
            elif frames_captured > 0:
                checks.append(CheckResult(
                    name="Webcam Capture",
                    status="WARN",
                    message=(
                        f"Source {src} opened ({backend_name}) with drops: "
                        f"{frames_captured} ok, {failed_reads} failed in {elapsed:.1f}s ({fps_measured} FPS)."
                    )
                ))
            else:
                checks.append(CheckResult(
                    name="Webcam Capture",
                    status="WARN",
                    message=f"Source {src} opened ({backend_name}) but 0 frames could be read in {elapsed:.1f}s."
                ))
        else:
            if cap:
                cap.release()
            checks.append(CheckResult(
                name="Webcam Capture",
                status="WARN",
                message=f"Could not open source {src} ({backend_name}). Diagnostic synthetic frames will be used."
            ))
    except Exception as e:
        checks.append(CheckResult(
            name="Webcam Capture",
            status="FAIL",
            message=f"Error initializing OpenCV capture: {e}"
        ))

    # 3. Output Directory Write Access
    out_dir = Path(config.recording.output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        test_file = out_dir / ".preflight_write_test"
        with open(test_file, "w") as f:
            f.write("ok")
        test_file.unlink()
        checks.append(CheckResult(
            name="Output Storage",
            status="PASS",
            message=f"Directory '{out_dir.resolve()}' is writable."
        ))
    except Exception as e:
        checks.append(CheckResult(
            name="Output Storage",
            status="FAIL",
            message=f"Output directory '{out_dir}' is not writable: {e}"
        ))

    # 4. Audio / TTS Availability
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.stop()
        if voices:
            checks.append(CheckResult(
                name="Voice Alerts (TTS)",
                status="PASS",
                message=f"pyttsx3 initialized successfully. {len(voices)} voice(s) available."
            ))
        else:
            checks.append(CheckResult(
                name="Voice Alerts (TTS)",
                status="WARN",
                message="pyttsx3 initialized but 0 system voices were returned. Audio will run in silent mode."
            ))
    except Exception as e:
        checks.append(CheckResult(
            name="Voice Alerts (TTS)",
            status="WARN",
            message=f"TTS engine unavailable ({e}). System will fall back to visual alerts only."
        ))

    # 5. Local Ollama VLM Check
    try:
        import ollama
        client = ollama.Client(host=config.vlm.host)
        models_resp = client.list()
        installed_names = []
        if hasattr(models_resp, "models"):
            installed_names = [m.model for m in models_resp.models]
        elif isinstance(models_resp, dict) and "models" in models_resp:
            installed_names = [m.get("name", "") for m in models_resp["models"]]

        has_target = any(config.vlm.model in name for name in installed_names)
        has_alt = any(config.vlm.alternative_model in name for name in installed_names)

        if has_target:
            checks.append(CheckResult(
                name="Local VLM (Ollama)",
                status="PASS",
                message=f"Ollama reachable. Configured model '{config.vlm.model}' is installed.",
                details=f"Models: {', '.join(installed_names[:4])}..."
            ))
        elif has_alt:
            checks.append(CheckResult(
                name="Local VLM (Ollama)",
                status="WARN",
                message=f"Primary model '{config.vlm.model}' not found, but alternative '{config.vlm.alternative_model}' is available.",
                details=f"Models: {', '.join(installed_names[:4])}..."
            ))
        else:
            checks.append(CheckResult(
                name="Local VLM (Ollama)",
                status="WARN",
                message=f"Ollama online, but neither '{config.vlm.model}' nor '{config.vlm.alternative_model}' found. Deterministic rules will operate independently."
            ))
    except Exception as e:
        checks.append(CheckResult(
            name="Local VLM (Ollama)",
            status="WARN",
            message=f"Ollama server not reachable at {config.vlm.host} ({e}). The application will use deterministic vision rules."
        ))

    # 6. Optional YOLO Detector Check
    if config.yolo.enabled:
        if config.yolo.model_path and Path(config.yolo.model_path).exists():
            checks.append(CheckResult(
                name="Optional YOLO Detector",
                status="PASS",
                message=f"YOLO enabled and model found at: {config.yolo.model_path}"
            ))
        else:
            checks.append(CheckResult(
                name="Optional YOLO Detector",
                status="WARN",
                message=f"YOLO enabled but weights not found at: '{config.yolo.model_path}'. YOLO will be inactive."
            ))
    else:
        checks.append(CheckResult(
            name="Optional YOLO Detector",
            status="PASS",
            message="YOLO detector is disabled (using deterministic HSV & ROI vision)."
        ))

    # 7. Procedure Specification Check
    try:
        proc_file = Path(config.procedure_file)
        proc = load_procedure(proc_file)
        checks.append(CheckResult(
            name="Procedure Configuration",
            status="PASS",
            message=f"Successfully loaded '{proc.title}' ({len(proc.steps)} validated steps) from '{proc_file.name}'."
        ))
    except Exception as e:
        checks.append(CheckResult(
            name="Procedure Configuration",
            status="FAIL",
            message=f"Procedure file '{config.procedure_file}' validation failed: {e}"
        ))

    # Critical failures determine overall pass
    has_critical_failure = any(c.status == "FAIL" for c in checks)
    return PreflightReport(all_critical_passed=not has_critical_failure, checks=checks)


def print_preflight_report(report: PreflightReport) -> None:
    """Renders a clean CLI summary table of preflight results."""
    print("\n" + "=" * 70)
    print("        ACTIVITY DETECTOR — SYSTEM PREFLIGHT HEALTH REPORT")
    print("=" * 70)

    for c in report.checks:
        if c.status == "PASS":
            tag = "\033[92m[PASS]\033[0m" if sys.platform != "win32" else "[PASS]"
        elif c.status == "WARN":
            tag = "\033[93m[WARN]\033[0m" if sys.platform != "win32" else "[WARN]"
        else:
            tag = "\033[91m[FAIL]\033[0m" if sys.platform != "win32" else "[FAIL]"

        print(f" {tag:8s} {c.name:<25s} : {c.message}")
        if c.details:
            print(f"          Details: {c.details}")

    print("=" * 70)
    if report.all_critical_passed:
        print(" RESULT: Preflight checks PASSED. System is ready to monitor procedures.\n")
    else:
        print(" RESULT: Critical preflight check(s) FAILED. Please resolve before proceeding.\n")
