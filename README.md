# ActivityDetector — Offline-First Tabletop Procedure Monitor

> **Tabletop Demonstration Disclaimer**: This software is an engineering prototype designed for local laptop monitoring and demonstration of sequential procedures using ordinary tabletop objects (such as a notebook). It is **not** a flight experiment, not flight-qualified, and not certified for unsupervised mission-critical operations.

---

## Overview

**ActivityDetector** is an offline-first Python application that monitors an operator executing a defined object-handling procedure using a standard webcam. It uses a **local Vision-Language Model** (via Ollama `qwen3.5:latest` with automatic fallback to `gemma4:e4b-it-qat`) to inspect physical evidence across time, combined with **spatial Region-of-Interest tracking**, **multi-observation persistence**, **audible voice alerts** (local Text-to-Speech), and **local audit logging with video recording**.

The default demo is a safe, space-relevant **generic object-handling workflow** demonstrated using an ordinary notebook:
1. **Locate & Identify Object**: Place and identify the notebook in the center workspace.
2. **Open Notebook**: Open the notebook flat so pages/interior are visible.
3. **Close Notebook**: Close the notebook covers shut.
4. **Stow in Designated Area**: Move the closed notebook to the designated stowage area and leave it there.

### Key Capabilities
- **Generic Object Selection**: The operator can monitor any target object (default: "notebook") configured via YAML, CLI (`--target-object`), or live in the UI.
- **Strict VLM Schema Validation**: Local Ollama model returns structured physical observations (`object_visible`, `object_description`, `open_or_closed`, `held_or_on_surface`, `location`, `confidence`, `is_uncertain`, and `reasoning`). Ambiguous frames yield `UNCERTAIN` state rather than false completions.
- **Multi-Observation Persistence Across Time**: A single frame or VLM response **never** advances a step. Configurable multi-sample criteria (`min_vlm_samples: 2`) and temporal hold durations prevent flicker advances.
- **Temporal Sequence Anomaly Checks**: Detects out-of-order events (e.g. early stowage before opening/closing) and flags skipped steps.
- **Operator Review & Manual Controls**: Provides explicit "Confirm Step (Manual)" and "Flag Inconclusive" buttons with distinct audit trail logging.
- **100% Offline & Private**: Zero cloud APIs or external data transmission. Webcam video remains on the local machine.
- **Local Audit Records & Video Storage**: Records a streaming, timestamped `session_log.jsonl`, a consolidated `session_summary.json`, and an annotated `.mp4` video locally in `runs/`.

---

## Architecture & Layout

```
SIH26/
├── activity_detector/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py           # Pydantic configuration schemas and loaders
│   │   └── default_config.yaml   # Master default configuration (notebook workflow, tabletop ROIs)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── procedure.py          # Step & Procedure domain schemas (target_object, expected_state)
│   │   ├── engine.py             # Deterministic state machine, multi-observation tracking & anomalies
│   │   └── session.py            # JSONL session logger and summary report manager
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera.py             # DirectShow Windows webcam manager with zero lag
│   │   ├── detector.py           # Supporting HSV color & ROI spatial detector
│   │   ├── yolo_detector.py      # Optional local object detector wrapper
│   │   ├── vlm.py                # Local Ollama client (qwen3.5:latest / gemma4) with strict JSON schema
│   │   └── pipeline.py           # Vision pipeline coordinator & tabletop HUD rendering
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
│   │   └── main_window.py        # PyQt6 live monitoring dashboard with object selection
│   ├── cli.py                    # CLI runner, argument parser, and headless mode
│   └── preflight.py              # System health and diagnostic check tool
├── procedures/
│   ├── default_notebook_handling.yaml   # Default: 4-step generic object handling protocol
│   ├── sample_titration_experiment.yaml # Optional example: chemistry titration
│   └── sample_circuit_assembly.yaml    # Optional example: electronics assembly
├── tests/
│   ├── test_config.py            # Configuration validation tests
│   ├── test_procedure.py         # Procedure schema and prerequisite tests
│   ├── test_engine.py            # State transitions, multi-observation tracking, and anomaly tests
│   ├── test_detector.py          # Supporting vision detection tests
│   ├── test_session.py           # JSONL logging and summary lifecycle tests
│   ├── test_ui.py                # PyQt6 offscreen dashboard tests
│   └── test_vlm.py               # VLM schema extraction and mock tests
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
Start the graphical dashboard with the default notebook procedure:
```bash
python run.py
```

### 3. Change Target Object or Procedure via CLI
```bash
# Change target object name
python run.py --target-object "logbook"

# Load an alternative example procedure
python run.py --procedure procedures/sample_circuit_assembly.yaml

# Specify custom camera index or pre-recorded test video
python run.py --camera 0
python run.py --camera path/to/test_recording.mp4

# Run with local MJPEG streaming enabled
python run.py --stream

# Disable Ollama VLM or audio announcements
python run.py --no-vlm
python run.py --no-audio
```

### 4. Headless Mode (Terminal Only)
For headless systems or automated testing:
```bash
python run.py --headless
```

---

## Configuration & Procedure Specification

### Default Configuration (`activity_detector/config/default_config.yaml`)
Key configuration options:
- `procedure_file`: `procedures/default_notebook_handling.yaml`
- `target_object`: `"notebook"`
- `camera`: resolution (`640x480`), `fps: 30`, and `source: 0`.
- `vision.rois`: tabletop zones:
  - `workspace_center`: central activity zone (`[0.22, 0.20, 0.78, 0.85]`).
  - `stowed_area`: designated stowage location (`[0.68, 0.05, 0.98, 0.50]`).
  - `prep_left`: preparation area (`[0.02, 0.20, 0.28, 0.80]`).
- `vlm`: provider (`ollama`), model (`qwen3.5:latest`), fallback (`gemma4:e4b-it-qat`), `sample_interval_seconds: 3.5`, and `num_gpu: 0` (CPU offloading for laptop stability).
- `recording`: local output directory (`runs/`), format (`mp4`), and automatic recording.

### Defining a Custom Procedure (`procedures/default_notebook_handling.yaml`)
Procedures are defined in declarative YAML:

```yaml
id: "notebook_handling_protocol_v1"
title: "Generic Object Handling Protocol: Notebook"
target_object: "notebook"

steps:
  - id: "step_1_locate"
    order: 1
    name: "Locate & Identify Notebook"
    instruction: "Place the closed notebook flat in the workspace center."
    expected_evidence:
      target_object: "notebook"
      expected_state:
        object_visible: true
        location: "workspace_center"
    completion_rule:
      rule_type: "vlm_state_tracking"
      min_vlm_samples: 2
      stable_frames: 8
      hold_seconds: 1.5

  - id: "step_2_open"
    order: 2
    name: "Open Notebook"
    instruction: "Open the notebook flat so that pages are visible."
    expected_evidence:
      target_object: "notebook"
      expected_state:
        object_visible: true
        open_or_closed: "open"
    completion_rule:
      rule_type: "vlm_state_tracking"
      min_vlm_samples: 2
      stable_frames: 8
      hold_seconds: 1.5
    prerequisites:
      - "step_1_locate"

  - id: "step_3_close"
    order: 3
    name: "Close Notebook"
    instruction: "Close the notebook covers shut."
    expected_evidence:
      target_object: "notebook"
      expected_state:
        object_visible: true
        open_or_closed: "closed"
    completion_rule:
      rule_type: "vlm_state_tracking"
      min_vlm_samples: 2
      stable_frames: 8
      hold_seconds: 1.5
    prerequisites:
      - "step_2_open"

  - id: "step_4_stow"
    order: 4
    name: "Stow in Designated Area"
    instruction: "Move the closed notebook to the stowed area (top right) and leave it there."
    expected_evidence:
      target_object: "notebook"
      expected_state:
        object_visible: true
        location: "stowed_area"
        open_or_closed: "closed"
    completion_rule:
      rule_type: "vlm_state_tracking"
      min_vlm_samples: 2
      stable_frames: 8
      hold_seconds: 2.0
    prerequisites:
      - "step_3_close"
```

---

## Live Demonstration Script (Step-by-Step)

Demonstrate the full procedure monitoring workflow using any standard notebook:

### Required Demonstration Items
- One ordinary notebook (spiral, bound, or notepad).
- A tabletop surface facing your webcam.

### Demonstration Walkthrough
1. **Launch the Application**:
   ```bash
   python run.py
   ```
2. **Start Monitoring Session**:
   - Click the green **Start Session** button.
   - The system announces: *"Step 1: Locate & Identify Notebook. Place the closed notebook flat in the workspace center."*
   - Video recording begins automatically (`● REC ON`), and a new session directory is created in `runs/`.
3. **Execute Step 1 (Locate Notebook)**:
   - Place the notebook in the center of the tabletop (`workspace_center`).
   - The local VLM inspects the frame and reports `object_visible: true, location: "workspace_center"`.
   - The **Multi-Sample Stability** gauge fills as separate observations confirm the state across time.
   - Once stable, the system announces: *"Step 2: Open Notebook. Open the notebook flat so that pages are visible."*
4. **Execute Step 2 (Open Notebook)**:
   - Flip the notebook open so the pages/interior are visible.
   - The VLM confirms `open_or_closed: "open"`.
   - After persistent observation, the system transitions to Step 3: *"Step 3: Close Notebook..."*
5. **Execute Step 3 (Close Notebook)**:
   - Close the notebook covers shut.
   - The VLM confirms `open_or_closed: "closed"`.
   - The system advances to Step 4: *"Step 4: Stow in Designated Area..."*
6. **Trigger an Out-of-Order Anomaly (Early Stowage)**:
   - *Test Anomaly*: If you move the notebook directly to the stowage area during Step 1 or 2 (skipping opening/closing), the system detects `location: "stowed_area"` and emits a **Sequence Anomaly (OUT_OF_ORDER)** warning.
   - The status badge changes to **NEEDS ATTENTION**, and an audible alert informs the operator.
   - Click **Acknowledge Alerts** once corrected.
7. **Execute Step 4 (Stow Notebook)**:
   - Move the closed notebook to the top-right `stowed_area` of the frame.
   - The system verifies `location: "stowed_area", open_or_closed: "closed"`.
   - Session status transitions to **COMPLETED** (Green).
8. **Manual Confirm / Flag Inconclusive**:
   - If lighting is poor or the notebook angle is ambiguous, click **Flag Inconclusive** to transition to `UNCERTAIN` for operator review.
   - Click **Confirm Step (Manual)** to advance with an explicit operator override in the audit log.
9. **Review Session Audit Records**:
   - Click **Stop Session**.
   - Inspect the generated run directory in `runs/session_YYYYMMDD_HHMMSS_<id>/`:
     - `session_log.jsonl`: Line-by-line audit stream with microsecond ISO timestamps.
     - `session_summary.json`: Executive compliance report with durations and warnings.
     - `session_recording_*.mp4`: Recorded video footage of the full demonstration.

---

## Recognition Capabilities & Model Limitations

### Open-Ended Recognition & Best-Effort Nature
- The system uses general-purpose local vision models (`qwen3.5:latest` or `gemma4:e4b-it-qat`) to recognize user-selected objects.
- **No single vision model can reliably classify every arbitrary object under arbitrary lighting**. The UI displays the active target object and model confidence score to provide transparent observability.
- The prompt is strictly focused on **visible physical evidence** (e.g. visible pages vs. closed covers, resting on surface vs. held in hand, position relative to frame). The model is instructed **never** to speculate on human intent or fabricate unseen actions.

### Multi-Observation Hold Rule
- A single still image cannot definitively prove that an action was completed (e.g., whether a notebook was opened and then closed).
- The procedure engine requires **multi-observation persistence across time** (`min_vlm_samples: 2` separated across time intervals and continuous multi-frame holds) to verify physical state changes.
- Ambiguous or low-confidence outputs transition the engine to `UNCERTAIN` rather than guessing.

### Offline & Hardware Resilience
- If the Ollama service is stopped or unavailable, the application displays a clear preflight warning, retains UI functionality, and allows manual operator advancement.
- Laptop GPU VRAM allocation is stabilized by setting `vlm.num_gpu: 0` in configuration, offloading compute buffers to system RAM to prevent CUDA out-of-memory errors.

---

## Automated Test Suite

Run the full automated test suite (runs 100% offline without requiring a physical camera or active Ollama server):
```bash
pytest -v
```
All 26 tests cover:
- Configuration validation and tabletop ROIs (`test_config.py`)
- Procedure schemas, prerequisites, and JSON serialization (`test_procedure.py`)
- Deterministic sequence engine, multi-sample VLM tracking, out-of-order anomalies, and timeouts (`test_engine.py`)
- Supporting spatial ROI detector (`test_detector.py`)
- Session management, JSONL logging, and summary reports (`test_session.py`)
- PyQt6 offscreen dashboard lifecycle and manual overrides (`test_ui.py`)
- VLM strict schema validation, Markdown extraction, and mock client (`test_vlm.py`)
