"""Command line interface and runner for ActivityDetector."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

from activity_detector.audio.speech import SpeechPrompter
from activity_detector.config.settings import AppConfig, load_config
from activity_detector.core.engine import EngineState, EngineUpdate, ProcedureEngine, StepRecord, WarningEvent
from activity_detector.core.procedure import Procedure, load_procedure
from activity_detector.core.session import SessionManager
from activity_detector.preflight import print_preflight_report, run_preflight_checks
from activity_detector.video.recorder import VideoRecorder
from activity_detector.video.streamer import MjpegHttpStreamer
from activity_detector.vision.pipeline import VisionPipeline

logger = logging.getLogger("activity_detector.cli")


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="activity-detector",
        description="Offline-first experiment activity monitor with sequence tracking and local vision.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--procedure",
        type=str,
        default=None,
        help="Path to experiment procedure YAML/JSON file.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera index (e.g. 0) or path to video file.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run system preflight diagnostic check and exit.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless terminal mode without Qt GUI.",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable Ollama Vision-Language Model interpretation.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Mute Text-to-Speech audio announcements.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable local MJPEG video streaming.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for session logs and video recordings.",
    )
    return parser


def run_headless_session(config: AppConfig, procedure: Procedure, max_seconds: int = 60) -> None:
    """Runs a session in headless terminal mode with live console telemetry."""
    print("\n" + "=" * 65)
    print(f" STARTING HEADLESS SESSION: '{procedure.title}'")
    print(f" Procedure contains {len(procedure.steps)} steps.")
    print(f" Video Source: {config.camera.source}")
    print("=" * 65 + "\n")

    session_mgr = SessionManager(config.recording.output_dir)
    session_dir = session_mgr.start_session(procedure)
    engine = ProcedureEngine(procedure)
    pipeline = VisionPipeline(config)
    speech = SpeechPrompter(config.audio)
    recorder = VideoRecorder(config.recording)

    streamer = None
    if config.streaming.enabled:
        streamer = MjpegHttpStreamer(config.streaming)
        streamer.start()

    def on_transition(rec: StepRecord) -> None:
        print(f"\n>>> [STEP COMPLETED] Step {rec.order}: '{rec.name}' in {rec.duration_seconds}s ({rec.evidence_source})")
        nxt = procedure.get_step_by_index(engine.current_step_index)
        if nxt:
            print(f">>> [NEXT STEP] Step {nxt.order}: '{nxt.name}' -> {nxt.instruction}")
            speech.announce_step(nxt.order, nxt.name, nxt.instruction)

    def on_warning(warn: WarningEvent) -> None:
        print(f"\n!!! [WARNING] ({warn.warning_type.value}): {warn.message}")
        speech.announce_warning(warn.warning_type.value, warn.message)

    engine.register_transition_listener(on_transition)
    engine.register_warning_listener(on_warning)

    pipeline.start()
    engine.start_session()

    if config.recording.auto_record_on_session_start:
        recorder.start_recording(
            target_directory=session_dir,
            width=config.camera.width,
            height=config.camera.height
        )

    # Announce first step
    if procedure.steps:
        s1 = procedure.steps[0]
        print(f">>> [ACTIVE STEP] Step 1: '{s1.name}' -> {s1.instruction}")
        speech.announce_step(s1.order, s1.name, s1.instruction)

    start_time = time.time()
    last_printed_sec = -1
    try:
        while time.time() - start_time < max_seconds:
            annotated_frame, evidence = pipeline.process_next_frame(
                EngineUpdate(
                    state=engine.state,
                    current_step_index=engine.current_step_index,
                    current_step=procedure.get_step_by_index(engine.current_step_index),
                    next_step=procedure.get_step_by_index(engine.current_step_index + 1),
                    progress_percentage=0.0,
                    step_records=engine.step_records,
                    recent_warning=None,
                    stability_ratio=0.0,
                    evidence_summary="",
                    evidence_source="none",
                    transition_occurred=False,
                )
            )

            update = engine.process_frame(evidence)
            session_mgr.log_engine_update(update)

            if recorder.is_recording and annotated_frame is not None:
                recorder.push_frame(annotated_frame)

            if streamer and streamer.is_active and annotated_frame is not None:
                streamer.update_frame(annotated_frame)

            # Print periodic line once per 2 seconds
            cur_sec = int(time.time() - start_time)
            if cur_sec != last_printed_sec and cur_sec % 2 == 0:
                last_printed_sec = cur_sec
                step_name = update.current_step.name if update.current_step else "None"
                print(
                    f"[{cur_sec:02d}s] "
                    f"State: {update.state.value:<15s} | "
                    f"Step: {step_name:<30s} | "
                    f"Stability: {int(update.stability_ratio * 100):3d}% | "
                    f"Detections: {len(evidence.detections)}"
                )

            if update.state == EngineState.COMPLETED:
                print("\n>>> All steps satisfied! Completed procedure.")
                break

            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\nHeadless session interrupted by operator (Ctrl+C).")

    finally:
        video_path = recorder.stop_recording()
        if video_path:
            session_mgr.record_video_status(video_path, success=not recorder.has_error)

        summary_path = session_mgr.close_session(
            final_records=engine.step_records,
            all_warnings=engine.history_warnings
        )
        engine.stop_session()
        pipeline.stop()
        speech.stop()
        if streamer:
            streamer.stop()

        print("\n" + "=" * 65)
        print(f" SESSION SUMMARY SAVED: {summary_path}")
        print("=" * 65 + "\n")


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # CLI Overrides
    if args.procedure:
        config.procedure_file = args.procedure
    if args.camera is not None:
        try:
            config.camera.source = int(args.camera)
        except ValueError:
            config.camera.source = args.camera
    if args.no_vlm:
        config.vlm.enabled = False
    if args.no_audio:
        config.audio.enabled = False
    if args.stream:
        config.streaming.enabled = True
    if args.output_dir:
        config.recording.output_dir = args.output_dir

    # Handle preflight check
    if args.preflight:
        report = run_preflight_checks(config)
        print_preflight_report(report)
        return 0 if report.all_critical_passed else 1

    # Load and validate procedure
    try:
        procedure = load_procedure(config.procedure_file)
    except Exception as e:
        print(f"Error loading procedure '{config.procedure_file}': {e}", file=sys.stderr)
        return 1

    # Run in Headless mode
    if args.headless:
        run_headless_session(config, procedure)
        return 0

    # Launch GUI Mode (PyQt6)
    try:
        from PyQt6.QtWidgets import QApplication
        from activity_detector.ui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        window = MainWindow(config, procedure)
        window.start()
        return app.exec()
    except Exception as e:
        print(f"Fatal error launching GUI: {e}. Falling back to headless mode.", file=sys.stderr)
        run_headless_session(config, procedure)
        return 0


if __name__ == "__main__":
    sys.exit(main())
