# ActivityDetector — Offline-First Experiment Activity Monitor

> **Prototype Disclaimer**: This software is an engineering prototype designed for local laptop monitoring and demonstration of sequential laboratory procedures. It is **not** flight-qualified, certified for safety-critical environments, or suitable for unsupervised mission-critical operations.

---

## Overview

**ActivityDetector** is an offline-first Python application that monitors an operator executing a defined laboratory procedure using a standard webcam. It combines **deterministic computer vision** (HSV color thresholding, contour geometry, and spatial Region-of-Interest tracking) with **multi-frame stability verification**, an optional **local Vision-Language Model** (via Ollama `qwen3.5:latest` or `gemma4:e4b-it-qat`), **audible voice alerts** (local Text-to-Speech), and **local audit logging with video recording**.

### Key Capabilities
- **Live Monitoring Dashboard**: Built in PyQt6 with high-contrast HUD overlays, multi-frame stability gauges, state indicator chips, and step cards.
- **Deterministic Step Engine**: Validates procedure steps against explicit evidence requirements; prevents premature progression using multi-frame stability and time-hold criteria.
- **Anomaly & Error Detection**: Flags skipped steps (detecting future step evidence prematurely), repeated steps, and step timeouts.
- **Explainable Evidence**: Distinguishes deterministic rule detections (`[RULE]`) from local model interpretations (`[VLM]`).
- **Offline & Local**: Requires zero cloud APIs or internet connectivity at runtime. Functions reliably even if Ollama or GPU acceleration is unavailable.
- **Audit Logging & Video Storage**: Records a streaming, timestamped `session_log.jsonl`, a consolidated `session_summary.json`, and an annotated `.mp4` video locally in `runs/`.
- **Isolated Streaming**: Optional local MJPEG video server (`http://127.0.0.1:8554/video_feed`) isolated behind an abstract interface.

---

## Architecture & Layout

```
SIH26/
├── activity_detector/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py           # Pydantic configuration schemas and loaders
│   │   └── default_config.yaml   # Master default configuration file
│   ├── core/
│   │   ├── __init__.py
│   │   ├── procedure.py          # Step and Procedure domain schemas and validation
│   │   ├── engine.py             # Deterministic state machine & sequence engine
│   │   └── session.py            # JSONL session logger and summary report manager
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera.py             # Thread-safe zero-lag webcam/video capture manager
│   │   ├── detector.py           # Deterministic HSV color & ROI spatial detector
│   │   ├── yolo_detector.py      # Optional local YOLO detector wrapper
│   │   ├── vlm.py                # Local Ollama client with strict JSON schema parsing
│   │   └── pipeline.py           # Coordinating vision pipeline & HUD rendering
│   ├── audio/
│   │   ├── __init__.py
│   │   └── speech.py             # Local Text-to-Speech worker with cooldowns & mute
│   ├── video/
│   │   ├── __init__.py
│   │   ├── recorder.py           # Threaded local video recorder with failure safety
│   │   └── streamer.py           # Optional isolated HTTP MJPEG streamer
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── widgets.py            # Reusable UI cards, gauges, badges, and canvas
│   │   └── main_window.py        # PyQt6 live monitoring desktop dashboard
│   ├── cli.py                    # CLI runner, argument parser, and headless mode
│   └── preflight.py              # System health and diagnostic check tool
├── procedures/
│   ├── sample_titration_experiment.yaml  # Labeled chemistry procedure
│   └── sample_circuit_assembly.yaml     # Labeled electronics procedure
├── tests/
│   ├── test_config.py            # Configuration validation tests
│   ├── test_procedure.py         # Procedure schema and prerequisite tests
│   ├── test_engine.py            # State transitions, timeouts, and skipped step tests
│   ├── test_detector.py          # Deterministic vision detection tests
│   ├── test_session.py           # JSONL logging and summary lifecycle tests
│   ├── test_ui.py                # PyQt6 offscreen dashboard tests
│   └── test_vlm.py               # VLM JSON extraction and mock tests
├── run.py                        # Root launcher script
└── README.md
```

---

## Quickstart & Run Instructions

### 1. Run Preflight Health Check
Verify your local environment, camera access, Ollama status, and filesystem permissions:
```bash
python run.py --preflight
```

### 2. Launch Live Monitoring GUI
Start the graphical dashboard:
```bash
python run.py
```

### 3. Run Headless Mode (Terminal Only)
For headless systems, servers, or automated test rigs:
```bash
python run.py --headless
```

### 4. CLI Options & Overrides
```bash
# Use an alternative procedure
python run.py --procedure procedures/sample_circuit_assembly.yaml

# Specify custom camera index or pre-recorded test video
python run.py --camera 1
python run.py --camera path/to/test_recording.mp4

# Run with local MJPEG streaming enabled
python run.py --stream

# Disable Ollama VLM or audio announcements
python run.py --no-vlm
python run.py --no-audio

# Custom configuration file
python run.py --config my_custom_config.yaml
```

---

## Configuration & Replacing the Procedure

### Default Configuration (`activity_detector/config/default_config.yaml`)
Key configuration options include:
- `camera`: resolution (`width`, `height`), `fps`, and `source` (index or video path).
- `vision.rois`: normalized coordinates (`x1`, `y1`, `x2`, `y2` from 0.0 to 1.0) defining:
  - `workbench_center`: central execution zone.
  - `staging_left`: item preparation and staging area.
  - `disposal_right`: waste/finished staging area.
- `vision.colors`: HSV ranges for deterministic color detection (`blue_reagent`, `yellow_flask`, `green_indicator`, `red_pipette`).
- `vlm`: provider (`ollama`), model (`qwen3.5:latest`), fallback (`gemma4:e4b-it-qat`), `sample_interval_seconds` (4.0s), and `num_gpu: 0` (CPU offload to prevent GPU CUDA OOM on laptop chips).
- `audio`: rate, volume, and announcement `cooldown_seconds`.
- `recording`: output directory (`runs/`), format (`mp4`), and auto-record behavior.

### How to Replace the Example Procedure
Create a new YAML or JSON file in `procedures/my_procedure.yaml`:

```yaml
id: "custom_lab_protocol_v1"
title: "Custom Reagent Protocol"
version: "1.0.0"
description: "Description of the protocol."

steps:
  - id: "step_1_prep"
    order: 1
    name: "Position Yellow Flask"
    instruction: "Place the yellow flask in the left staging area."
    expected_evidence:
      required_colors:
        - "yellow_flask"
      roi: "staging_left"
      vlm_keywords:
        - "flask"
        - "staging"
    completion_rule:
      rule_type: "stable_detection"
      stable_frames: 8
      min_confidence: 0.65
      hold_seconds: 1.5
    timeout_seconds: 45.0

  - id: "step_2_transfer"
    order: 2
    name: "Transfer to Center"
    instruction: "Move the flask into the center workbench."
    expected_evidence:
      required_colors:
        - "yellow_flask"
      roi: "workbench_center"
    completion_rule:
      rule_type: "stable_detection"
      stable_frames: 10
      hold_seconds: 2.0
    prerequisites:
      - "step_1_prep"
```
Launch with your custom procedure:
```bash
python run.py --procedure procedures/my_procedure.yaml
```
You can also switch procedures live in the GUI via the **Load Procedure...** button.

---

## Live Demonstration Script (Step-by-Step)

To demonstrate the full monitoring workflow using readily available items:

### Required Demonstration Items
- A **yellow** object (e.g. yellow post-it note, yellow cup, or yellow highlighter) representing the `yellow_flask`.
- A **blue** object (e.g. blue pen, blue bottle cap, or blue card) representing the `blue_reagent`.
- A **green** object (e.g. green marker or green paper) representing the `green_indicator`.

### Demonstration Walkthrough
1. **Launch the Application**:
   ```bash
   python run.py
   ```
2. **Start the Session**:
   - Click the blue **Start Session** button.
   - The system announces: *"Step 1: Inspect Safety & Staging Area. Ensure PPE is equipped and place the yellow reagent flask in the staging area."*
   - Video recording begins automatically (`● REC ON`), and a new session directory is created in `runs/`.
3. **Execute Step 1 (Normal Progression)**:
   - Place the **yellow** item into the left region of the webcam frame labeled **Staging Left**.
   - Notice the yellow bounding box appears with `[RULE] yellow_flask: 90% [staging_left]`.
   - The **Multi-frame Stability** gauge fills up (`0% → 100%`).
   - Once held stable for 1.5 seconds, the system automatically advances to Step 2 with an audible voice prompt: *"Step 2: Transfer Reaction Flask to Center..."*
4. **Trigger a Skipped Step Alert (Sequence Anomaly Detection)**:
   - While on Step 2, instead of moving the yellow flask to the center, introduce the **blue** item directly into the central region.
   - The system detects evidence for Step 3 (`blue_reagent`) while Step 2 is still incomplete.
   - The state transitions to **NEEDS ATTENTION** (Red).
   - An audible alert sounds: *"Alert: Sequence warning: Detected evidence for future Step 3 before completing Step 2!"*
   - The warning is logged in the live event box and written to `session_log.jsonl`.
5. **Operator Recovery**:
   - Move the yellow object to the center as instructed. Step 2 completes and stability advances.
   - Click **Acknowledge Alerts** to clear active flags.
6. **Complete the Remaining Steps**:
   - Introduce the blue item in the center (Step 3).
   - Show the green item in the center (Step 4).
   - Move the yellow item to the right side (Step 5: Disposal).
   - Status badge transitions to **COMPLETED** (Green).
7. **Review Session Audit Records**:
   - Click **Stop Session**.
   - Inspect the generated run directory in `runs/session_YYYYMMDD_HHMMSS_<id>/`:
     - `session_log.jsonl`: Line-by-line audit stream with microsecond ISO timestamps.
     - `session_summary.json`: Executive compliance report with durations and warnings.
     - `session_recording_*.mp4`: Recorded video footage of the full demonstration.

---

## Offline Behavior & Model Limitations

### Offline Resilience
- All core functions (webcam capture, color/ROI detection, state machine sequence engine, audio alerts, and video/log writing) are **100% local**.
- If the local Ollama server is stopped or unavailable, the application logs a warning, falls back to deterministic vision rules, and operates without interruption.
- If audio devices are disabled or pyttsx3 is unavailable, the system switches to visual alert mode.

### Local VLM Limitations
- The local Vision-Language Model (`qwen3.5:latest` or `gemma4:e4b-it-qat`) is an **advisory signal only**.
- VLM calls are sampled at a rate-limited interval (default: every 4.0 seconds) in a background thread to prevent webcam frame drops.
- Model responses are strictly validated against a structured JSON schema. Any malformed, unparseable, or conflicting output is safely classified as **UNCERTAIN**, requiring operator review rather than blindly advancing steps.
- On laptop GPUs with shared or tight VRAM (e.g. RTX 4060 Laptop 8GB), `num_gpu` is set to `0` by default in configuration. This offloads compute buffer allocation to system RAM, avoiding CUDA out-of-memory errors during model startup.

---

## Automated Testing

Run the full automated test suite (independent of hardware webcams, GPUs, Ollama, or networks):
```bash
pytest -v
```
All 23 tests cover:
- Configuration validation and constraints (`test_config.py`)
- Procedure domain schemas and prerequisite checks (`test_procedure.py`)
- Sequence engine transitions, multi-frame stability, timeouts, and skipped steps (`test_engine.py`)
- Deterministic HSV color segmentation and ROI spatial reasoning (`test_detector.py`)
- Session management, JSONL logging, and summary reports (`test_session.py`)
- PyQt6 offscreen dashboard lifecycle and manual overrides (`test_ui.py`)
- VLM schema parsing, Markdown extraction, and mock fallback (`test_vlm.py`)

---

## Troubleshooting Guide

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| `Could not open video source: '0'` | Webcam in use by another app or permissions denied. | Close other apps using the camera or specify an alternate index (`--camera 1`) or test video file (`--camera clip.mp4`). |
| `Ollama server not reachable` | Ollama daemon not running. | Run `ollama serve` in a terminal or launch Ollama from Windows tray. The app works without Ollama using deterministic rules. |
| `CUDA error: requested functionality not supported` | VRAM constrained when allocating compute buffers. | Ensure `vlm.num_gpu: 0` in `default_config.yaml` to use CPU offloading for VLM inference. |
| `TTS engine unavailable` | Windows SAPI audio driver issue. | Verify default playback device in Windows Sound settings. The app falls back to visual alert badges automatically. |
| `RECORDING ERROR: VideoWriter failure` | Missing codec or disk permission. | The app automatically falls back from MP4 (`mp4v`) to AVI (`MJPG`). Check write permissions in `runs/`. |
