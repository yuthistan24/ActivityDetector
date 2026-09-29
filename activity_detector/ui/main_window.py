"""Master live monitoring interface for ActivityDetector in PyQt6."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Optional
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from activity_detector.audio.speech import SpeechPrompter
from activity_detector.config.settings import AppConfig
from activity_detector.core.engine import (
    EngineState,
    EngineUpdate,
    ProcedureEngine,
    StepRecord,
    WarningEvent,
)
from activity_detector.core.procedure import Procedure, load_procedure
from activity_detector.core.session import SessionManager
from activity_detector.ui.widgets import (
    STATE_STYLES,
    StatusBadgeWidget,
    StepCardWidget,
    StepListWidget,
    VideoCanvasWidget,
)
from activity_detector.video.recorder import VideoRecorder
from activity_detector.video.streamer import MjpegHttpStreamer
from activity_detector.vision.pipeline import VisionPipeline

logger = logging.getLogger("activity_detector.ui")


class MainWindow(QMainWindow):
    """Primary live monitoring dashboard."""

    def __init__(self, config: AppConfig, procedure: Procedure) -> None:
        super().__init__()
        self.config = config
        self.procedure = procedure
        self.target_object: str = procedure.target_object or getattr(config, "target_object", "notebook")

        # Core subsystems
        self.engine = ProcedureEngine(self.procedure)
        self.engine.set_target_object(self.target_object)

        self.pipeline = VisionPipeline(self.config)
        self.pipeline.set_target_object(self.target_object)

        self.speech = SpeechPrompter(self.config.audio)
        self.session_manager = SessionManager(self.config.recording.output_dir)
        self.recorder = VideoRecorder(self.config.recording)

        self.streamer: Optional[MjpegHttpStreamer] = None
        if self.config.streaming.enabled:
            self.streamer = MjpegHttpStreamer(self.config.streaming)

        self._last_state = EngineState.IDLE
        self._last_step_order = 0
        self._session_start_time: Optional[float] = None

        self._init_window()
        self._setup_ui()
        self._connect_signals()

        # Frame processing timer (~30 FPS)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._process_tick)

    def _init_window(self) -> None:
        self.setWindowTitle("ActivityDetector — Tabletop Procedure Monitoring Prototype")
        self.resize(1380, 840)
        self.setMinimumSize(1024, 680)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0f19;
            }
            QWidget {
                color: #e2e8f0;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            }
            QPushButton {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 7px 12px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #475569;
            }
            QPushButton:pressed {
                background-color: #0f172a;
            }
            QPushButton:disabled {
                background-color: #0f172a;
                color: #475569;
                border-color: #1e293b;
            }
        """)

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 10, 14, 10)
        main_layout.setSpacing(10)

        # Top Header Bar
        header_bar = self._create_header_bar()
        main_layout.addLayout(header_bar)

        # Splitter: Left (Video & Controls) vs Right (Procedure, Evidence & Events)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #1e293b;
                width: 3px;
            }
        """)

        # Left Panel (Video & Control bar)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        # Video canvas
        self.video_widget = VideoCanvasWidget()
        left_layout.addWidget(self.video_widget, stretch=1)

        # Recording error alert banner (hidden by default)
        self.rec_error_banner = QLabel("")
        self.rec_error_banner.setStyleSheet("color: #ffffff; background-color: #dc2626; padding: 6px 12px; border-radius: 4px; font-weight: bold;")
        self.rec_error_banner.setVisible(False)
        left_layout.addWidget(self.rec_error_banner)

        # Control Toolbar
        control_bar = self._create_control_toolbar()
        left_layout.addWidget(control_bar)

        splitter.addWidget(left_panel)

        # Right Panel (Procedure, Evidence, Diagnostics & Logs)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        # 1. Procedure Info & Progress Section
        exp_header = self._create_procedure_info_section()
        right_layout.addWidget(exp_header)

        # 2. Active Step Card
        self.active_step_card = StepCardWidget()
        right_layout.addWidget(self.active_step_card)

        # 3. Next Step Preview Box
        self.next_step_preview = QLabel("Next Step: Initializing...")
        self.next_step_preview.setStyleSheet("color: #94a3b8; background-color: #111827; border: 1px dashed #334155; padding: 8px 12px; border-radius: 6px; font-size: 11px;")
        self.next_step_preview.setWordWrap(True)
        right_layout.addWidget(self.next_step_preview)

        # 4. Diagnostics & Evidence Source Breakdown
        diag_box = self._create_diagnostics_box()
        right_layout.addWidget(diag_box)

        # 5. Timeline Checklist
        self.step_list_widget = StepListWidget()
        self.step_list_widget.setFixedHeight(140)
        right_layout.addWidget(self.step_list_widget)

        # 6. Event Log Box
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setFixedHeight(110)
        self.event_log.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a;
                color: #cbd5e1;
                border: 1px solid #1e293b;
                border-radius: 6px;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 11px;
                padding: 4px;
            }
        """)
        right_layout.addWidget(self.event_log)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 65)
        splitter.setStretchFactor(1, 35)

        main_layout.addWidget(splitter, stretch=1)

        # Status Bar
        self.setStatusBar(QStatusBar())
        self.statusBar().setStyleSheet("color: #64748b; font-size: 11px; background-color: #090d16;")
        self._update_status_bar()

    def _create_header_bar(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(12)

        title_col = QVBoxLayout()
        app_title = QLabel("ACTIVITY DETECTOR")
        app_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        app_title.setStyleSheet("color: #38bdf8; letter-spacing: 1px;")

        app_subtitle = QLabel("Tabletop Prototype • Generic Object Tracking • Local Vision & Explainable State Machine")
        app_subtitle.setFont(QFont("Segoe UI", 9))
        app_subtitle.setStyleSheet("color: #64748b;")

        title_col.addWidget(app_title)
        title_col.addWidget(app_subtitle)
        header.addLayout(title_col)

        header.addStretch()

        # Target Object selector / input
        target_layout = QHBoxLayout()
        lbl_target = QLabel("Target Object:")
        lbl_target.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        lbl_target.setStyleSheet("color: #94a3b8;")

        self.input_target_object = QLineEdit(self.target_object)
        self.input_target_object.setFixedWidth(120)
        self.input_target_object.setStyleSheet("""
            QLineEdit {
                background-color: #1e293b;
                color: #38bdf8;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        self.btn_set_target = QPushButton("Set")
        self.btn_set_target.clicked.connect(self._on_target_object_changed)

        target_layout.addWidget(lbl_target)
        target_layout.addWidget(self.input_target_object)
        target_layout.addWidget(self.btn_set_target)
        header.addLayout(target_layout)

        header.addSpacing(10)

        # Status badge
        self.status_badge = StatusBadgeWidget()
        header.addWidget(self.status_badge)

        # Recording dot
        self.rec_badge = QLabel("● REC OFF")
        self.rec_badge.setStyleSheet("color: #64748b; font-weight: bold; font-size: 11px; padding: 4px 10px; border: 1px solid #334155; border-radius: 12px;")
        header.addWidget(self.rec_badge)

        return header

    def _create_control_toolbar(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 6px;")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.btn_session = QPushButton("Start Session")
        self.btn_session.setStyleSheet("background-color: #0284c7; color: white; border: none;")
        self.btn_session.clicked.connect(self._toggle_session)
        layout.addWidget(self.btn_session)

        # Operator verification controls
        self.btn_confirm = QPushButton("Confirm Step (Manual)")
        self.btn_confirm.setStyleSheet("background-color: #059669; color: white; border: none;")
        self.btn_confirm.clicked.connect(self._manual_confirm_step)
        layout.addWidget(self.btn_confirm)

        self.btn_flag = QPushButton("Flag Inconclusive")
        self.btn_flag.setStyleSheet("background-color: #d97706; color: white; border: none;")
        self.btn_flag.clicked.connect(self._manual_flag_uncertain)
        layout.addWidget(self.btn_flag)

        layout.addSpacing(6)

        # Navigation controls
        self.btn_prev = QPushButton("◀ Step Back")
        self.btn_prev.clicked.connect(self._manual_prev_step)
        layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next Step ▶")
        self.btn_next.clicked.connect(self._manual_next_step)
        layout.addWidget(self.btn_next)

        self.btn_ack = QPushButton("Ack Alerts")
        self.btn_ack.setStyleSheet("background-color: #78350f; color: #fde68a;")
        self.btn_ack.clicked.connect(self._acknowledge_alerts)
        layout.addWidget(self.btn_ack)

        layout.addStretch()

        self.btn_record = QPushButton("Record Video")
        self.btn_record.clicked.connect(self._toggle_recording)
        layout.addWidget(self.btn_record)

        self.btn_mute = QPushButton("Mute Audio" if not self.speech.is_muted else "Unmute Audio")
        self.btn_mute.clicked.connect(self._toggle_mute)
        layout.addWidget(self.btn_mute)

        self.btn_load_proc = QPushButton("Load Procedure...")
        self.btn_load_proc.clicked.connect(self._load_custom_procedure)
        layout.addWidget(self.btn_load_proc)

        return frame

    def _create_procedure_info_section(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        self.proc_title_label = QLabel(self.procedure.title)
        self.proc_title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.proc_title_label.setStyleSheet("color: #f1f5f9;")
        title_row.addWidget(self.proc_title_label)

        self.proc_ver_label = QLabel(f"v{self.procedure.version}")
        self.proc_ver_label.setStyleSheet("color: #64748b; font-size: 10px;")
        title_row.addWidget(self.proc_ver_label)
        title_row.addStretch()

        self.session_time_label = QLabel("Elapsed: 00:00")
        self.session_time_label.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.session_time_label.setStyleSheet("color: #38bdf8;")
        title_row.addWidget(self.session_time_label)

        layout.addLayout(title_row)

        # Progress bar
        prog_row = QHBoxLayout()
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setRange(0, 100)
        self.overall_progress_bar.setValue(0)
        self.overall_progress_bar.setFixedHeight(10)
        self.overall_progress_bar.setTextVisible(False)
        self.overall_progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1e293b;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #0284c7;
                border-radius: 5px;
            }
        """)
        self.overall_progress_text = QLabel("0 / 0 (0%)")
        self.overall_progress_text.setStyleSheet("color: #94a3b8; font-size: 10px; min-width: 70px;")

        prog_row.addWidget(self.overall_progress_bar)
        prog_row.addWidget(self.overall_progress_text)
        layout.addLayout(prog_row)

        return frame

    def _create_diagnostics_box(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QLabel("EVIDENCE & VISUAL DIAGNOSTICS")
        header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        header.setStyleSheet("color: #94a3b8; letter-spacing: 0.5px;")
        layout.addWidget(header)

        self.diag_vlm = QLabel("Visual Evidence (VLM): Awaiting observation...")
        self.diag_vlm.setFont(QFont("Segoe UI", 9))
        self.diag_vlm.setStyleSheet("color: #38bdf8;")
        self.diag_vlm.setWordWrap(True)
        layout.addWidget(self.diag_vlm)

        self.diag_rules = QLabel("Supporting Vision: Online")
        self.diag_rules.setFont(QFont("Segoe UI", 9))
        self.diag_rules.setStyleSheet("color: #94a3b8;")
        self.diag_rules.setWordWrap(True)
        layout.addWidget(self.diag_rules)

        return frame

    def _connect_signals(self) -> None:
        self.engine.register_transition_listener(self._on_step_transition)
        self.engine.register_warning_listener(self._on_engine_warning)

    def _start_pipeline(self) -> None:
        """Starts video pipeline and timer loop."""
        self.pipeline.start()
        if self.streamer:
            self.streamer.start()
        self._timer.start(33)  # ~30 FPS

    def start(self) -> None:
        """Launches the window and starts capture."""
        self.show()
        self._start_pipeline()
        self._append_log(f"System initialized. Target object: '{self.target_object}'")

    def _on_target_object_changed(self) -> None:
        """Handler for target object text update."""
        new_obj = self.input_target_object.text().strip()
        if new_obj:
            self.target_object = new_obj
            self.pipeline.set_target_object(new_obj)
            self.engine.set_target_object(new_obj)
            self._append_log(f"Target object set to: '{new_obj}'")

    def _process_tick(self) -> None:
        """Called every ~33ms: grabs frame, processes detections, updates engine & UI."""
        # 1. Run vision pipeline with current engine state
        annotated_frame, evidence = self.pipeline.process_next_frame(
            EngineUpdate(
                state=self.engine.state,
                current_step_index=self.engine.current_step_index,
                current_step=self.procedure.get_step_by_index(self.engine.current_step_index),
                next_step=self.procedure.get_step_by_index(self.engine.current_step_index + 1),
                progress_percentage=0.0,
                step_records=self.engine.step_records,
                recent_warning=None,
                stability_ratio=0.0,
                evidence_summary="",
                evidence_source="none",
                transition_occurred=False,
                vlm_sample_count=self.engine._vlm_consistent_sample_count,
            )
        )

        # 2. Feed evidence to procedure state machine
        engine_update = self.engine.process_frame(evidence)

        # 3. Log to session audit trail if active
        if self.session_manager.is_active:
            self.session_manager.log_engine_update(engine_update)

        # 4. Stream frame if active
        if self.streamer and self.streamer.is_active:
            self.streamer.update_frame(annotated_frame)

        # 5. Push to video recorder if active
        if self.recorder.is_recording:
            record_frame = annotated_frame if self.config.recording.record_annotated else self.pipeline.camera._latest_frame
            if record_frame is not None:
                self.recorder.push_frame(record_frame)

            if self.recorder.has_error:
                self.rec_error_banner.setText(f"RECORDING ERROR: {self.recorder.error_message}")
                self.rec_error_banner.setVisible(True)
                self.session_manager.record_video_status(
                    video_path=Path(""),
                    success=False,
                    error_msg=self.recorder.error_message
                )

        # 6. Update UI Canvas & Diagnostics
        self.video_widget.update_frame(annotated_frame)
        self._update_dashboard(engine_update, evidence)

    def _update_dashboard(self, update: EngineUpdate, evidence) -> None:
        """Refreshes all cards, timeline, and diagnostic meters."""
        # State badge
        self.status_badge.set_state(update.state)

        # Active Step Card
        curr = update.current_step
        if curr:
            self.active_step_card.update_step(
                order=curr.order,
                total=len(self.procedure.steps),
                name=curr.name,
                instruction=curr.instruction,
                evidence_summary=update.evidence_summary or f"Awaiting physical evidence of '{self.target_object}'",
                stability_ratio=update.stability_ratio,
            )
        elif update.state == EngineState.COMPLETED:
            self.active_step_card.update_step(
                order=len(self.procedure.steps),
                total=len(self.procedure.steps),
                name="Procedure Complete",
                instruction="All protocol actions verified and recorded.",
                evidence_summary="All steps satisfied.",
                stability_ratio=1.0,
            )

        # Next Step Preview
        nxt = update.next_step
        if nxt:
            self.next_step_preview.setText(f"Next: Step {nxt.order} — {nxt.name} ({nxt.instruction[:65]}...)")
        elif update.state == EngineState.COMPLETED:
            self.next_step_preview.setText("Next: None. Procedure finished.")
        else:
            self.next_step_preview.setText("Next: Final step in progress.")

        # Overall progress
        pct = int(update.progress_percentage)
        self.overall_progress_bar.setValue(pct)
        completed_cnt = sum(1 for r in update.step_records if r.status.value == "completed")
        self.overall_progress_text.setText(f"{completed_cnt} / {len(update.step_records)} ({pct}%)")

        # Elapsed Session Time
        if self._session_start_time and self.session_manager.is_active:
            elapsed = int(time.time() - self._session_start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            self.session_time_label.setText(f"Elapsed: {mins:02d}:{secs:02d}")

        # Diagnostics: VLM Physical State Breakdown
        if evidence.vlm_state:
            st = evidence.vlm_state
            vis = "Yes" if st.get("object_visible") else "No"
            oc = st.get("open_or_closed", "unknown")
            loc = st.get("location", "unknown")
            conf = int(st.get("confidence", 0.0) * 100)
            uncertain = " [UNCERTAIN / REVIEW]" if st.get("is_uncertain") else ""
            desc = st.get("object_description", "")
            self.diag_vlm.setText(
                f"VLM ({self.config.vlm.model}): Visible: {vis} | State: {oc} | Location: {loc} | Conf: {conf}%{uncertain}\n"
                f"Description: {desc[:60]}..." if desc else f"VLM: Visible: {vis} | State: {oc} | Location: {loc} | Conf: {conf}%"
            )
        elif self.pipeline.vlm:
            self.diag_vlm.setText(f"VLM ({self.config.vlm.model}): Sampling image for '{self.target_object}'...")
        else:
            self.diag_vlm.setText("VLM: Inactive / Disabled")

        rules_desc = f"Supporting Cues: {len(evidence.detections)} color/ROI items. Status: {update.evidence_summary}"
        self.diag_rules.setText(rules_desc)

        # Step list timeline checklist
        self.step_list_widget.set_steps(update.step_records, update.current_step_index)

    def _manual_confirm_step(self) -> None:
        """Operator explicitly verifies and confirms the current step."""
        ok = self.engine.confirm_current_step(reason="operator_confirmed")
        if ok:
            curr = self.procedure.get_step_by_index(self.engine.current_step_index)
            if curr:
                self.speech.announce_step(curr.order, curr.name, curr.instruction)
            self._append_log("Operator confirmed step completion.")

    def _manual_flag_uncertain(self) -> None:
        """Operator flags current observation as ambiguous or inconclusive."""
        ok = self.engine.flag_uncertain_step(reason="operator_flagged_uncertain")
        if ok:
            self._append_log("Operator flagged step as inconclusive/uncertain.")

    def _toggle_session(self) -> None:
        """Starts or stops the active monitoring session."""
        if not self.session_manager.is_active:
            session_dir = self.session_manager.start_session(self.procedure)
            self._session_start_time = time.time()
            self.engine.start_session()

            if self.config.recording.auto_record_on_session_start:
                self.recorder.start_recording(
                    target_directory=session_dir,
                    width=self.config.camera.width,
                    height=self.config.camera.height
                )
                self.rec_badge.setText("● REC ON")
                self.rec_badge.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 11px; padding: 4px 10px; border: 1px solid #ef4444; border-radius: 12px;")
                self.btn_record.setText("Stop Recording")

            self.btn_session.setText("Stop Session")
            self.btn_session.setStyleSheet("background-color: #dc2626; color: white; border: none;")
            self._append_log(f"Session started for target '{self.target_object}': {self.session_manager.current_session_id}")

            if self.procedure.steps:
                s1 = self.procedure.steps[0]
                self.speech.announce_step(s1.order, s1.name, s1.instruction)

        else:
            video_file = self.recorder.stop_recording()
            if video_file:
                self.session_manager.record_video_status(video_file, success=not self.recorder.has_error)

            summary_path = self.session_manager.close_session(
                final_records=self.engine.step_records,
                all_warnings=self.engine.history_warnings
            )
            self.engine.stop_session()

            self.btn_session.setText("Start Session")
            self.btn_session.setStyleSheet("background-color: #0284c7; color: white; border: none;")
            self.rec_badge.setText("● REC OFF")
            self.rec_badge.setStyleSheet("color: #64748b; font-weight: bold; font-size: 11px; padding: 4px 10px; border: 1px solid #334155; border-radius: 12px;")
            self.btn_record.setText("Record Video")

            self._append_log(f"Session concluded. Summary saved to: {summary_path}")
            QMessageBox.information(
                self,
                "Session Completed",
                f"Session completed successfully.\nAudit report saved to:\n{summary_path}"
            )

    def _toggle_recording(self) -> None:
        """Toggles video recording manually."""
        if not self.recorder.is_recording:
            target_dir = self.session_manager.session_dir or Path(self.config.recording.output_dir)
            ok = self.recorder.start_recording(
                target_directory=target_dir,
                width=self.config.camera.width,
                height=self.config.camera.height
            )
            if ok:
                self.rec_badge.setText("● REC ON")
                self.rec_badge.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 11px; padding: 4px 10px; border: 1px solid #ef4444; border-radius: 12px;")
                self.btn_record.setText("Stop Recording")
                self._append_log("Video recording manually started.")
        else:
            self.recorder.stop_recording()
            self.rec_badge.setText("● REC OFF")
            self.rec_badge.setStyleSheet("color: #64748b; font-weight: bold; font-size: 11px; padding: 4px 10px; border: 1px solid #334155; border-radius: 12px;")
            self.btn_record.setText("Record Video")
            self._append_log("Video recording manually stopped.")

    def _toggle_mute(self) -> None:
        muted = self.speech.toggle_mute()
        self.btn_mute.setText("Unmute Audio" if muted else "Mute Audio")
        self._append_log(f"Audio {'muted' if muted else 'unmuted'}.")

    def _manual_next_step(self) -> None:
        ok = self.engine.advance_step(reason="operator_button")
        if ok:
            curr = self.procedure.get_step_by_index(self.engine.current_step_index)
            if curr:
                self.speech.announce_step(curr.order, curr.name, curr.instruction)
            self._append_log("Operator manually advanced to next step.")

    def _manual_prev_step(self) -> None:
        ok = self.engine.revert_step(reason="operator_button")
        if ok:
            self._append_log("Operator manually reverted to previous step.")

    def _acknowledge_alerts(self) -> None:
        self.engine.acknowledge_warnings()
        self._append_log("Active alerts acknowledged by operator.")

    def _load_custom_procedure(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Experiment Procedure YAML/JSON",
            "procedures",
            "Procedure Files (*.yaml *.yml *.json)"
        )
        if path:
            try:
                new_proc = load_procedure(path)
                self.procedure = new_proc
                self.target_object = new_proc.target_object or self.target_object
                self.input_target_object.setText(self.target_object)

                self.engine = ProcedureEngine(self.procedure)
                self.engine.set_target_object(self.target_object)

                self.pipeline.set_target_object(self.target_object)
                self._connect_signals()

                self.proc_title_label.setText(new_proc.title)
                self.proc_ver_label.setText(f"v{new_proc.version}")
                self._append_log(f"Loaded procedure: '{new_proc.title}' ({len(new_proc.steps)} steps)")
            except Exception as e:
                QMessageBox.critical(self, "Failed to Load Procedure", f"Error: {e}")

    def _on_step_transition(self, record: StepRecord) -> None:
        self._append_log(f"Completed Step {record.order}: '{record.name}' in {record.duration_seconds}s ({record.evidence_source})")
        nxt = self.procedure.get_step_by_index(self.engine.current_step_index)
        if nxt:
            self.speech.announce_step(nxt.order, nxt.name, nxt.instruction)
        elif self.engine.state == EngineState.COMPLETED:
            self.speech.speak("Procedure completed successfully. All steps verified.")

    def _on_engine_warning(self, warning: WarningEvent) -> None:
        self._append_log(f"WARNING [{warning.warning_type.value}]: {warning.message}")
        self.speech.announce_warning(warning.warning_type.value, warning.message)

    def _append_log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.event_log.append(f"[{ts}] {text}")

    def _update_status_bar(self) -> None:
        vlm_model = self.config.vlm.model if self.config.vlm.enabled else "Disabled"
        cam_src = self.config.camera.source
        streaming_status = f"Stream: http://{self.config.streaming.host}:{self.config.streaming.port}{self.config.streaming.path}" if self.config.streaming.enabled else "Streaming: Off"
        msg = f"Camera Source: {cam_src} | Target: {self.target_object} | Local VLM: {vlm_model} | {streaming_status} | Output: {self.config.recording.output_dir}/"
        self.statusBar().showMessage(msg)

    def closeEvent(self, event) -> None:  # type: ignore
        """Ensures all background threads and open writers shutdown safely."""
        self._timer.stop()
        if self.session_manager.is_active:
            self.session_manager.close_session(
                final_records=self.engine.step_records,
                all_warnings=self.engine.history_warnings,
                status_override="ABORTED"
            )
        self.recorder.stop_recording()
        if self.streamer:
            self.streamer.stop()
        self.pipeline.stop()
        self.speech.stop()
        super().closeEvent(event)
