"""Master live monitoring dashboard for ActivityDetector (PyQt6).

Changes vs previous version
----------------------------
* Target selector: validated against YOLO supported classes with inline
  feedback.  Unsupported names are rejected with a clear message.
* YOLO detector status replaces the VLM recognition card as the primary
  evidence panel.
* Stale-result display: previous detections turn amber after STALE_SECONDS.
* All panels show real data or an explicit empty/loading/unavailable message.
* Empty skeleton cards removed.
* Procedure defaults to bottle_tabletop_workflow (no notebook open/close).
* Settings persisted to default_config.yaml on valid change.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import List, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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
from activity_detector.config.settings import AppConfig, save_config
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
from activity_detector.vision.pipeline import STALE_DETECTION_SECONDS, VisionPipeline
from activity_detector.vision.vlm import BaseVlmClient

logger = logging.getLogger("activity_detector.ui")

CONFIG_PATH = Path("activity_detector/config/default_config.yaml")


class MainWindow(QMainWindow):
    """Primary live monitoring dashboard."""

    def __init__(
        self,
        config: AppConfig,
        procedure: Procedure,
        vlm_client: Optional[BaseVlmClient] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.procedure = procedure
        self.target_object: str = procedure.target_object or getattr(
            config, "target_object", "bottle"
        )

        # Core subsystems
        self.engine = ProcedureEngine(self.procedure)
        self.engine.set_target_object(self.target_object)

        self.pipeline = VisionPipeline(self.config, vlm_client=vlm_client)
        # Apply initial target (validates against YOLO vocab)
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
        self._last_detection_timestamp: float = 0.0
        self._last_engine_update: Optional[EngineUpdate] = None

        self._init_window()
        self._setup_ui()
        self._connect_signals()

        # Frame processing timer (~30 FPS)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._process_tick)

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _init_window(self) -> None:
        self.setWindowTitle("ActivityDetector — Tabletop Bottle Tracking Prototype")
        self.resize(1440, 880)
        self.setMinimumSize(1100, 720)
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0f19; }
            QWidget {
                color: #e2e8f0;
                font-family: 'Segoe UI', Roboto, sans-serif;
                font-size: 12px;
            }
            QPushButton {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 7px 14px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #334155; border-color: #475569; }
            QPushButton:pressed { background-color: #0f172a; }
            QPushButton:disabled {
                background-color: #0f172a;
                color: #475569;
                border-color: #1e293b;
            }
            QComboBox {
                background-color: #1e293b;
                color: #38bdf8;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1e293b;
                color: #e2e8f0;
                selection-background-color: #0284c7;
            }
        """)

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        root.addLayout(self._create_header_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(
            "QSplitter::handle { background-color: #1e293b; width: 3px; }"
        )

        # ── Left panel: video + controls ──────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)
        ll.setSpacing(6)

        self.video_widget = VideoCanvasWidget()
        ll.addWidget(self.video_widget, stretch=1)

        self.rec_error_banner = QLabel("")
        self.rec_error_banner.setStyleSheet(
            "color: #fff; background: #dc2626; padding: 5px 12px; border-radius: 4px; font-weight:bold;"
        )
        self.rec_error_banner.setVisible(False)
        ll.addWidget(self.rec_error_banner)

        ll.addWidget(self._create_control_toolbar())
        splitter.addWidget(left)

        # ── Right panel: info cards ───────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        rl.setSpacing(8)

        rl.addWidget(self._create_procedure_info_section())
        rl.addWidget(self._create_detector_status_card())
        rl.addWidget(self._create_active_step_card())
        rl.addWidget(self._create_next_step_section())
        rl.addWidget(self._create_diagnostics_box())

        # Step timeline
        self.step_list_widget = StepListWidget()
        self.step_list_widget.setMinimumHeight(100)
        self.step_list_widget.setMaximumHeight(180)
        rl.addWidget(self.step_list_widget)

        # Event log
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMinimumHeight(90)
        self.event_log.setMaximumHeight(130)
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
        rl.addWidget(self.event_log)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 62)
        splitter.setStretchFactor(1, 38)
        root.addWidget(splitter, stretch=1)

        self.setStatusBar(QStatusBar())
        self.statusBar().setStyleSheet(
            "color: #64748b; font-size: 11px; background-color: #090d16;"
        )
        self._update_status_bar()

    # ------------------------------------------------------------------
    # UI component builders
    # ------------------------------------------------------------------

    def _create_header_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(12)

        title_col = QVBoxLayout()
        title = QLabel("ACTIVITY DETECTOR")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #38bdf8; letter-spacing: 1px;")
        subtitle = QLabel(
            "Tabletop Bottle Tracking  •  YOLO11n (COCO 80 classes)  •  Local Prototype"
        )
        subtitle.setStyleSheet("color: #64748b; font-size: 10px;")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        bar.addLayout(title_col)
        bar.addStretch()

        # ── Target class selector ────────────────────────────────────
        tgt_layout = QVBoxLayout()
        tgt_label = QLabel("Detection Target (COCO class):")
        tgt_label.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: bold;")
        tgt_layout.addWidget(tgt_label)

        tgt_row = QHBoxLayout()
        tgt_row.setSpacing(4)

        self.combo_target = QComboBox()
        self.combo_target.setFixedWidth(160)
        self.combo_target.setEditable(True)
        supported = self.pipeline.supported_classes
        if supported:
            self.combo_target.addItems(supported)
            idx = self.combo_target.findText(self.target_object)
            if idx >= 0:
                self.combo_target.setCurrentIndex(idx)
            else:
                self.combo_target.setCurrentText(self.target_object)
        else:
            self.combo_target.addItem(self.target_object)
            self.combo_target.setCurrentText(self.target_object)

        self.btn_set_target = QPushButton("Apply")
        self.btn_set_target.setFixedWidth(60)
        self.btn_set_target.clicked.connect(self._on_target_changed)
        self.combo_target.lineEdit().returnPressed.connect(self._on_target_changed)  # type: ignore[union-attr]

        tgt_row.addWidget(self.combo_target)
        tgt_row.addWidget(self.btn_set_target)
        tgt_layout.addLayout(tgt_row)

        # Validation feedback label
        self.lbl_target_validation = QLabel("")
        self.lbl_target_validation.setStyleSheet("color: #f87171; font-size: 10px;")
        self.lbl_target_validation.setWordWrap(True)
        self.lbl_target_validation.setMaximumWidth(360)
        tgt_layout.addWidget(self.lbl_target_validation)

        bar.addLayout(tgt_layout)
        bar.addSpacing(16)

        # Status badge
        self.status_badge = StatusBadgeWidget()
        bar.addWidget(self.status_badge)

        # REC dot
        self.rec_badge = QLabel("● REC OFF")
        self.rec_badge.setStyleSheet(
            "color: #64748b; font-weight: bold; font-size: 11px; "
            "padding: 4px 10px; border: 1px solid #334155; border-radius: 12px;"
        )
        bar.addWidget(self.rec_badge)
        return bar

    def _create_control_toolbar(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            "background-color: #111827; border: 1px solid #1f2937; "
            "border-radius: 8px; padding: 6px;"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        self.btn_session = QPushButton("▶  Start Session")
        self.btn_session.setStyleSheet(
            "background-color: #0284c7; color: white; border: none;"
        )
        self.btn_session.clicked.connect(self._toggle_session)
        lay.addWidget(self.btn_session)

        self.btn_confirm = QPushButton("✓  Confirm Step")
        self.btn_confirm.setStyleSheet(
            "background-color: #059669; color: white; border: none;"
        )
        self.btn_confirm.clicked.connect(self._manual_confirm_step)
        lay.addWidget(self.btn_confirm)

        self.btn_flag = QPushButton("⚠  Flag Inconclusive")
        self.btn_flag.setStyleSheet(
            "background-color: #d97706; color: white; border: none;"
        )
        self.btn_flag.clicked.connect(self._manual_flag_uncertain)
        lay.addWidget(self.btn_flag)

        lay.addSpacing(4)

        self.btn_prev = QPushButton("◀ Prev Step")
        self.btn_prev.clicked.connect(self._manual_prev_step)
        lay.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next Step ▶")
        self.btn_next.clicked.connect(self._manual_next_step)
        lay.addWidget(self.btn_next)

        self.btn_ack = QPushButton("Ack Alerts")
        self.btn_ack.setStyleSheet("background-color: #78350f; color: #fde68a;")
        self.btn_ack.clicked.connect(self._acknowledge_alerts)
        lay.addWidget(self.btn_ack)

        lay.addStretch()

        self.btn_record = QPushButton("Record Video")
        self.btn_record.clicked.connect(self._toggle_recording)
        lay.addWidget(self.btn_record)

        self.btn_mute = QPushButton(
            "Mute Audio" if not self.speech.is_muted else "Unmute Audio"
        )
        self.btn_mute.clicked.connect(self._toggle_mute)
        lay.addWidget(self.btn_mute)

        self.btn_load_proc = QPushButton("Load Procedure…")
        self.btn_load_proc.clicked.connect(self._load_custom_procedure)
        lay.addWidget(self.btn_load_proc)

        return frame

    def _create_procedure_info_section(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            "background-color: #111827; border: 1px solid #1f2937; "
            "border-radius: 8px;"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        row = QHBoxLayout()
        self.proc_title_label = QLabel(self.procedure.title)
        self.proc_title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.proc_title_label.setStyleSheet("color: #f1f5f9;")
        row.addWidget(self.proc_title_label)

        self.proc_ver_label = QLabel(f"v{self.procedure.version}")
        self.proc_ver_label.setStyleSheet("color: #64748b; font-size: 10px;")
        row.addWidget(self.proc_ver_label)
        row.addStretch()

        self.session_time_label = QLabel("Elapsed: 00:00")
        self.session_time_label.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.session_time_label.setStyleSheet("color: #38bdf8;")
        row.addWidget(self.session_time_label)
        lay.addLayout(row)

        prog_row = QHBoxLayout()
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setRange(0, 100)
        self.overall_progress_bar.setValue(0)
        self.overall_progress_bar.setFixedHeight(10)
        self.overall_progress_bar.setTextVisible(False)
        self.overall_progress_bar.setStyleSheet("""
            QProgressBar { background-color: #1e293b; border-radius: 5px; }
            QProgressBar::chunk { background-color: #0284c7; border-radius: 5px; }
        """)
        self.overall_progress_text = QLabel("0 / 0 (0%)")
        self.overall_progress_text.setStyleSheet("color: #94a3b8; font-size: 10px; min-width: 80px;")
        prog_row.addWidget(self.overall_progress_bar)
        prog_row.addWidget(self.overall_progress_text)
        lay.addLayout(prog_row)
        return frame

    def _create_detector_status_card(self) -> QFrame:
        """Primary evidence panel — YOLO detector status and live detection."""
        frame = QFrame()
        frame.setStyleSheet(
            "background-color: #111827; border: 1px solid #1f2937; "
            "border-radius: 10px;"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        header_row = QHBoxLayout()
        header_lbl = QLabel("OBJECT DETECTOR  (YOLO11n / COCO)")
        header_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        header_lbl.setStyleSheet("color: #94a3b8; letter-spacing: 0.5px;")
        header_row.addWidget(header_lbl)
        header_row.addStretch()

        self.det_status_pill = QLabel("LOADING")
        self.det_status_pill.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.det_status_pill.setStyleSheet(
            "background: #1e293b; color: #94a3b8; border: 1px solid #334155; "
            "border-radius: 10px; padding: 3px 10px;"
        )
        header_row.addWidget(self.det_status_pill)
        lay.addLayout(header_row)

        self.det_main_label = QLabel("Initialising detector…")
        self.det_main_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.det_main_label.setStyleSheet("color: #38bdf8;")
        self.det_main_label.setWordWrap(True)
        lay.addWidget(self.det_main_label)

        self.det_detail_label = QLabel("—")
        self.det_detail_label.setStyleSheet(
            "color: #94a3b8; background: #1e293b; padding: 6px 8px; border-radius: 6px;"
        )
        self.det_detail_label.setWordWrap(True)
        lay.addWidget(self.det_detail_label)

        self.det_timestamp_label = QLabel("Last update: —")
        self.det_timestamp_label.setStyleSheet("color: #475569; font-size: 10px;")
        lay.addWidget(self.det_timestamp_label)

        return frame

    def _create_active_step_card(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            "background-color: #111827; border: 1px solid #1f2937; "
            "border-radius: 10px;"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        self.step_title_label = QLabel("Step —: Waiting for session")
        self.step_title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.step_title_label.setStyleSheet("color: #38bdf8;")
        lay.addWidget(self.step_title_label)

        self.step_instruction_label = QLabel("Start a session to begin the procedure.")
        self.step_instruction_label.setWordWrap(True)
        self.step_instruction_label.setFont(QFont("Segoe UI", 10))
        self.step_instruction_label.setStyleSheet(
            "color: #e2e8f0; background: #1e293b; padding: 8px; border-radius: 6px;"
        )
        lay.addWidget(self.step_instruction_label)

        self.step_evidence_label = QLabel("Evidence: awaiting detection")
        self.step_evidence_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.step_evidence_label.setWordWrap(True)
        lay.addWidget(self.step_evidence_label)

        stab_row = QHBoxLayout()
        stab_lbl = QLabel("Stability:")
        stab_lbl.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.stability_bar = QProgressBar()
        self.stability_bar.setRange(0, 100)
        self.stability_bar.setValue(0)
        self.stability_bar.setFixedHeight(12)
        self.stability_bar.setTextVisible(False)
        self.stability_bar.setStyleSheet("""
            QProgressBar { background-color: #1e293b; border-radius: 6px; }
            QProgressBar::chunk { background-color: #10b981; border-radius: 6px; }
        """)
        self.stability_pct_label = QLabel("0%")
        self.stability_pct_label.setStyleSheet("color: #10b981; min-width: 35px; font-size: 10px;")
        stab_row.addWidget(stab_lbl)
        stab_row.addWidget(self.stability_bar)
        stab_row.addWidget(self.stability_pct_label)
        lay.addLayout(stab_row)
        return frame

    def _create_next_step_section(self) -> QLabel:
        self.next_step_label = QLabel("Next: —")
        self.next_step_label.setWordWrap(True)
        self.next_step_label.setStyleSheet(
            "color: #94a3b8; background: #111827; border: 1px dashed #334155; "
            "padding: 8px 12px; border-radius: 6px; font-size: 11px;"
        )
        return self.next_step_label

    def _create_diagnostics_box(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            "background-color: #111827; border: 1px solid #1f2937; "
            "border-radius: 8px;"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        hdr = QLabel("CAMERA & RUNTIME DIAGNOSTICS")
        hdr.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        hdr.setStyleSheet("color: #94a3b8; letter-spacing: 0.5px;")
        lay.addWidget(hdr)

        self.diag_camera = QLabel("Camera: Initialising…")
        self.diag_camera.setWordWrap(True)
        self.diag_camera.setStyleSheet("color: #4ade80; font-size: 11px;")
        lay.addWidget(self.diag_camera)

        self.diag_yolo = QLabel("YOLO: Loading model…")
        self.diag_yolo.setWordWrap(True)
        self.diag_yolo.setStyleSheet("color: #38bdf8; font-size: 11px;")
        lay.addWidget(self.diag_yolo)

        self.diag_config = QLabel("")
        self.diag_config.setWordWrap(True)
        self.diag_config.setStyleSheet("color: #64748b; font-size: 10px;")
        lay.addWidget(self.diag_config)
        return frame

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.engine.register_transition_listener(self._on_step_transition)
        self.engine.register_warning_listener(self._on_engine_warning)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.show()
        self.pipeline.start()
        if self.streamer:
            self.streamer.start()
        self._timer.start(33)  # ~30 FPS

        det = self.pipeline.get_detector_status()
        if det["available"]:
            self._append_log(
                f"YOLO detector ready: {det['supported_class_count']} COCO classes. "
                f"Target: '{self.target_object}'"
            )
        else:
            self._append_log(f"YOLO UNAVAILABLE: {det['status_message']}")

        self._append_log(
            f"Procedure: '{self.procedure.title}' ({len(self.procedure.steps)} steps)"
        )

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def _process_tick(self) -> None:
        annotated, evidence = self.pipeline.process_next_frame(
            EngineUpdate(
                state=self.engine.state,
                current_step_index=self.engine.current_step_index,
                current_step=self.procedure.get_step_by_index(
                    self.engine.current_step_index
                ),
                next_step=self.procedure.get_step_by_index(
                    self.engine.current_step_index + 1
                ),
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

        engine_update = self.engine.process_frame(evidence)
        self._last_engine_update = engine_update

        if self.session_manager.is_active:
            self.session_manager.log_engine_update(engine_update)

        if self.streamer and self.streamer.is_active:
            self.streamer.update_frame(annotated)

        if self.recorder.is_recording:
            record_frame = (
                annotated
                if self.config.recording.record_annotated
                else self.pipeline.camera._latest_frame
            )
            if record_frame is not None:
                self.recorder.push_frame(record_frame)
            if self.recorder.has_error:
                self.rec_error_banner.setText(
                    f"RECORDING ERROR: {self.recorder.error_message}"
                )
                self.rec_error_banner.setVisible(True)

        self.video_widget.update_frame(annotated)
        self._update_dashboard(engine_update, evidence)

    # ------------------------------------------------------------------
    # Dashboard refresh
    # ------------------------------------------------------------------

    def _update_dashboard(self, update: EngineUpdate, evidence) -> None:
        now = time.time()

        # ── State badge ───────────────────────────────────────────────
        self.status_badge.set_state(update.state)

        # ── Detector status card ──────────────────────────────────────
        det = self.pipeline.get_detector_status()
        target_hits = [
            d for d in evidence.detections
            if d.source == "yolo"
            and d.name.lower() == self.target_object.lower()
        ]

        if not det["available"]:
            self.det_status_pill.setText("UNAVAILABLE")
            self.det_status_pill.setStyleSheet(
                "background: #450a0a; color: #fca5a5; "
                "border: 1px solid #dc2626; border-radius: 10px; padding: 3px 10px;"
            )
            self.det_main_label.setText("Detector unavailable")
            self.det_main_label.setStyleSheet("color: #ef4444;")
            self.det_detail_label.setText(det["status_message"])
            self.det_timestamp_label.setText("No detections possible.")
        elif det["result_is_stale"]:
            age = det["last_detection_age"]
            self.det_status_pill.setText("STALE")
            self.det_status_pill.setStyleSheet(
                "background: #78350f; color: #fbbf24; "
                "border: 1px solid #d97706; border-radius: 10px; padding: 3px 10px;"
            )
            self.det_main_label.setText(
                f"No {self.target_object} detected (last seen {age:.1f}s ago)"
            )
            self.det_main_label.setStyleSheet("color: #f59e0b;")
            self.det_detail_label.setText(
                "Previous detection is stale. Place object back in view."
            )
            self.det_timestamp_label.setText(
                f"Latency: {det['inference_latency_ms']:.0f} ms | "
                f"Conf threshold: {det['confidence_threshold']:.2f} | "
                f"Stale after: {STALE_DETECTION_SECONDS:.0f}s"
            )
        elif target_hits:
            best = max(target_hits, key=lambda d: d.confidence)
            roi_str = f" in {best.roi.replace('_', ' ').upper()}" if best.roi else " (no ROI match)"
            self.det_status_pill.setText("DETECTED")
            self.det_status_pill.setStyleSheet(
                "background: #064e3b; color: #34d399; "
                "border: 1px solid #059669; border-radius: 10px; padding: 3px 10px;"
            )
            self.det_main_label.setText(
                f"Bottle detected{roi_str} — {int(best.confidence * 100)}% confidence"
            )
            self.det_main_label.setStyleSheet("color: #34d399;")
            # All hits with ROI info
            all_hits_str = "  |  ".join(
                f"{d.name} {int(d.confidence*100)}%"
                + (f" [{d.roi}]" if d.roi else "")
                for d in target_hits
            )
            self.det_detail_label.setText(all_hits_str or "—")
            self.det_timestamp_label.setText(
                f"Latency: {det['inference_latency_ms']:.0f} ms | "
                f"Conf threshold: {det['confidence_threshold']:.2f} | "
                f"Updated: {time.strftime('%H:%M:%S')}"
            )
            self._last_detection_timestamp = now
        else:
            # Detector running but no target found
            self.det_status_pill.setText("NO DETECTION")
            self.det_status_pill.setStyleSheet(
                "background: #0c1a2e; color: #60a5fa; "
                "border: 1px solid #1d4ed8; border-radius: 10px; padding: 3px 10px;"
            )
            if not det["target_supported"]:
                self.det_main_label.setText(
                    f"'{self.target_object}' not supported by this detector"
                )
                self.det_main_label.setStyleSheet("color: #ef4444;")
                self.det_detail_label.setText(det["target_validation_msg"])
            else:
                self.det_main_label.setText(
                    f"No {self.target_object} detected in current frame"
                )
                self.det_main_label.setStyleSheet("color: #60a5fa;")
                age = det.get("last_detection_age")
                if age is not None:
                    self.det_detail_label.setText(
                        f"Last seen {age:.1f}s ago. Point camera at the object."
                    )
                else:
                    self.det_detail_label.setText(
                        "Object not seen yet. Place it in camera view."
                    )
            self.det_timestamp_label.setText(
                f"Latency: {det['inference_latency_ms']:.0f} ms | "
                f"Conf threshold: {det['confidence_threshold']:.2f}"
            )

        # ── Active step card ──────────────────────────────────────────
        curr = update.current_step
        if curr:
            self.step_title_label.setText(
                f"Step {curr.order} of {len(self.procedure.steps)}: {curr.name}"
            )
            self.step_instruction_label.setText(curr.instruction)
            self.step_evidence_label.setText(
                f"Evidence: {update.evidence_summary or 'Awaiting detection…'}"
            )
        elif update.state == EngineState.COMPLETED:
            self.step_title_label.setText("✓  Procedure Complete")
            self.step_instruction_label.setText(
                "All steps verified. Session can be stopped."
            )
            self.step_evidence_label.setText("All evidence confirmed.")
        else:
            self.step_title_label.setText("Step —: Not started")
            self.step_instruction_label.setText("Start a session to begin.")
            self.step_evidence_label.setText("Evidence: —")

        pct = int(max(0.0, min(1.0, update.stability_ratio)) * 100)
        self.stability_bar.setValue(pct)
        self.stability_pct_label.setText(f"{pct}%")

        # ── Next step ─────────────────────────────────────────────────
        nxt = update.next_step
        if nxt:
            instr_snippet = nxt.instruction[:70]
            if len(nxt.instruction) > 70:
                instr_snippet += "…"
            self.next_step_label.setText(
                f"Next: Step {nxt.order} — {nxt.name}  ({instr_snippet})"
            )
        elif update.state == EngineState.COMPLETED:
            self.next_step_label.setText("Next: None — procedure finished.")
        elif curr:
            self.next_step_label.setText("Next: This is the final step.")
        else:
            self.next_step_label.setText("Next: —")

        # ── Overall progress ──────────────────────────────────────────
        prog_pct = int(update.progress_percentage)
        self.overall_progress_bar.setValue(prog_pct)
        completed_cnt = sum(
            1 for r in update.step_records if r.status.value == "completed"
        )
        self.overall_progress_text.setText(
            f"{completed_cnt} / {len(update.step_records)} ({prog_pct}%)"
        )

        # ── Elapsed session time ──────────────────────────────────────
        if self._session_start_time and self.session_manager.is_active:
            elapsed = int(now - self._session_start_time)
            self.session_time_label.setText(
                f"Elapsed: {elapsed // 60:02d}:{elapsed % 60:02d}"
            )

        # ── Camera diagnostics ────────────────────────────────────────
        cam = self.pipeline.get_camera_diagnostics()
        ui_fps = cam.get("ui_fresh_fps", 0.0)
        cam_age = cam["last_frame_age_seconds"]
        is_black = cam.get("is_black_frame", False)
        backend = cam.get("backend", "?")[:12]

        if cam["state"] == "connected":
            if is_black:
                self.diag_camera.setStyleSheet("color: #ef4444; font-weight: bold; font-size:11px;")
                self.diag_camera.setText(
                    f"Camera: BLACK FRAME ({backend}) — check physical lens shutter / lighting"
                )
            elif cam["is_stale"]:
                self.diag_camera.setStyleSheet("color: #f97316; font-size:11px;")
                self.diag_camera.setText(
                    f"Camera: STALE / FROZEN ({backend}) — no fresh frame for {cam_age:.1f}s"
                )
            else:
                self.diag_camera.setStyleSheet("color: #4ade80; font-size:11px;")
                self.diag_camera.setText(
                    f"Camera: CONNECTED ({backend}) | {ui_fps:.1f} FPS | "
                    f"age: {cam_age:.2f}s | "
                    f"mean pixel: {cam.get('pixel_mean', 0.0):.0f}"
                )
        elif cam["state"] == "reconnecting":
            self.diag_camera.setStyleSheet("color: #eab308; font-size:11px;")
            self.diag_camera.setText(
                f"Camera: RECONNECTING (attempt #{cam['reconnect_attempts']}) | "
                f"backend: {backend}"
            )
        else:
            self.diag_camera.setStyleSheet("color: #ef4444; font-size:11px;")
            err = cam.get("last_error") or "Device disconnected"
            self.diag_camera.setText(f"Camera: UNAVAILABLE — {err[:50]}")

        # ── YOLO diagnostics ──────────────────────────────────────────
        if det["available"]:
            self.diag_yolo.setStyleSheet("color: #38bdf8; font-size:11px;")
            self.diag_yolo.setText(
                f"YOLO11n: {det['supported_class_count']} classes | "
                f"target: '{det['target_class']}' | "
                f"conf≥{det['confidence_threshold']:.2f} | "
                f"latency: {det['inference_latency_ms']:.0f} ms"
            )
        else:
            self.diag_yolo.setStyleSheet("color: #ef4444; font-size:11px;")
            self.diag_yolo.setText(f"YOLO: UNAVAILABLE — {det['status_message'][:60]}")

        # ── Active config summary ─────────────────────────────────────
        self.diag_config.setText(
            f"Procedure: {self.procedure.title} | "
            f"Camera src: {self.config.camera.source} | "
            f"ROIs: {', '.join(self.config.vision.rois.keys())}"
        )

        # ── Step timeline ─────────────────────────────────────────────
        self.step_list_widget.set_steps(update.step_records, update.current_step_index)
        self._update_status_bar()

    # ------------------------------------------------------------------
    # Target change handler
    # ------------------------------------------------------------------

    def _on_target_changed(self) -> None:
        new_name = self.combo_target.currentText().strip().lower()
        if not new_name:
            self.lbl_target_validation.setText("Target name must not be empty.")
            return

        ok, msg = self.pipeline.set_target_object(new_name)
        if ok:
            self.target_object = new_name
            self.engine.set_target_object(new_name)
            self.lbl_target_validation.setText("")
            self._append_log(f"Target changed to '{new_name}'.")
            if self.session_manager.is_active:
                self.session_manager.log_event(
                    "TARGET_CHANGED", {"target_object": new_name}
                )
            # Persist to config
            self.config.target_object = new_name
            try:
                save_config(self.config, CONFIG_PATH)
            except Exception as exc:
                logger.warning(f"Could not persist config: {exc}")
        else:
            self.lbl_target_validation.setText(
                f"⚠ Unsupported: '{new_name}' is not in YOLO11n vocabulary. "
                "Detection target unchanged."
            )
            self._append_log(f"REJECTED target '{new_name}': not in YOLO vocabulary.")

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def _toggle_session(self) -> None:
        if not self.session_manager.is_active:
            session_dir = self.session_manager.start_session(self.procedure)
            self._session_start_time = time.time()
            self.engine.start_session()

            if self.config.recording.auto_record_on_session_start:
                self.recorder.start_recording(
                    target_directory=session_dir,
                    width=self.config.camera.width,
                    height=self.config.camera.height,
                )
                self.rec_badge.setText("● REC ON")
                self.rec_badge.setStyleSheet(
                    "color: #ef4444; font-weight: bold; font-size: 11px; "
                    "padding: 4px 10px; border: 1px solid #ef4444; border-radius: 12px;"
                )
                self.btn_record.setText("Stop Recording")

            self.btn_session.setText("■  Stop Session")
            self.btn_session.setStyleSheet(
                "background-color: #dc2626; color: white; border: none;"
            )
            self._append_log(
                f"Session started — target: '{self.target_object}' | "
                f"ID: {self.session_manager.current_session_id}"
            )
            if self.procedure.steps:
                s1 = self.procedure.steps[0]
                self.speech.announce_step(s1.order, s1.name, s1.instruction)
        else:
            video_file = self.recorder.stop_recording()
            if video_file:
                self.session_manager.record_video_status(
                    video_file, success=not self.recorder.has_error
                )

            summary_path = self.session_manager.close_session(
                final_records=self.engine.step_records,
                all_warnings=self.engine.history_warnings,
            )
            self.engine.stop_session()

            self.btn_session.setText("▶  Start Session")
            self.btn_session.setStyleSheet(
                "background-color: #0284c7; color: white; border: none;"
            )
            self.rec_badge.setText("● REC OFF")
            self.rec_badge.setStyleSheet(
                "color: #64748b; font-weight: bold; font-size: 11px; "
                "padding: 4px 10px; border: 1px solid #334155; border-radius: 12px;"
            )
            self.btn_record.setText("Record Video")
            self._append_log(f"Session ended. Report: {summary_path}")
            QMessageBox.information(
                self,
                "Session Completed",
                f"Session completed.\nAudit report: {summary_path}",
            )

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    def _manual_confirm_step(self) -> None:
        ok = self.engine.confirm_current_step(reason="operator_confirmed")
        if ok:
            curr = self.procedure.get_step_by_index(self.engine.current_step_index)
            if curr:
                self.speech.announce_step(curr.order, curr.name, curr.instruction)
            self._append_log("Operator confirmed step.")

    def _manual_flag_uncertain(self) -> None:
        ok = self.engine.flag_uncertain_step(reason="operator_flagged")
        if ok:
            self._append_log("Operator flagged step as inconclusive.")

    def _manual_next_step(self) -> None:
        ok = self.engine.advance_step(reason="operator_button")
        if ok:
            curr = self.procedure.get_step_by_index(self.engine.current_step_index)
            if curr:
                self.speech.announce_step(curr.order, curr.name, curr.instruction)
            self._append_log("Operator advanced to next step.")

    def _manual_prev_step(self) -> None:
        ok = self.engine.revert_step(reason="operator_button")
        if ok:
            self._append_log("Operator reverted to previous step.")

    def _acknowledge_alerts(self) -> None:
        self.engine.acknowledge_warnings()
        self._append_log("Alerts acknowledged.")

    def _toggle_recording(self) -> None:
        if not self.recorder.is_recording:
            target_dir = (
                self.session_manager.session_dir
                or Path(self.config.recording.output_dir)
            )
            ok = self.recorder.start_recording(
                target_directory=target_dir,
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
            if ok:
                self.rec_badge.setText("● REC ON")
                self.rec_badge.setStyleSheet(
                    "color: #ef4444; font-weight: bold; font-size: 11px; "
                    "padding: 4px 10px; border: 1px solid #ef4444; border-radius: 12px;"
                )
                self.btn_record.setText("Stop Recording")
                self._append_log("Recording started.")
        else:
            self.recorder.stop_recording()
            self.rec_badge.setText("● REC OFF")
            self.rec_badge.setStyleSheet(
                "color: #64748b; font-weight: bold; font-size: 11px; "
                "padding: 4px 10px; border: 1px solid #334155; border-radius: 12px;"
            )
            self.btn_record.setText("Record Video")
            self._append_log("Recording stopped.")

    def _toggle_mute(self) -> None:
        muted = self.speech.toggle_mute()
        self.btn_mute.setText("Unmute Audio" if muted else "Mute Audio")
        self._append_log(f"Audio {'muted' if muted else 'unmuted'}.")

    def _load_custom_procedure(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Procedure YAML/JSON",
            "procedures",
            "Procedure Files (*.yaml *.yml *.json)",
        )
        if not path:
            return
        try:
            new_proc = load_procedure(path)
            self.procedure = new_proc
            self.target_object = new_proc.target_object or self.target_object
            self.combo_target.setCurrentText(self.target_object)

            self.engine = ProcedureEngine(self.procedure)
            self.engine.set_target_object(self.target_object)
            self.pipeline.set_target_object(self.target_object)
            self._connect_signals()

            self.proc_title_label.setText(new_proc.title)
            self.proc_ver_label.setText(f"v{new_proc.version}")
            self._append_log(
                f"Loaded: '{new_proc.title}' ({len(new_proc.steps)} steps)"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", f"Error: {exc}")

    # ------------------------------------------------------------------
    # Event listeners
    # ------------------------------------------------------------------

    def _on_step_transition(self, record: StepRecord) -> None:
        self._append_log(
            f"✓ Step {record.order} '{record.name}' completed in "
            f"{record.duration_seconds}s ({record.evidence_source})"
        )
        nxt = self.procedure.get_step_by_index(self.engine.current_step_index)
        if nxt:
            self.speech.announce_step(nxt.order, nxt.name, nxt.instruction)
        elif self.engine.state == EngineState.COMPLETED:
            self.speech.speak("Procedure completed. All steps verified.")

    def _on_engine_warning(self, warning: WarningEvent) -> None:
        self._append_log(
            f"⚠ [{warning.warning_type.value.upper()}] {warning.message}"
        )
        self.speech.announce_warning(warning.warning_type.value, warning.message)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _append_log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.event_log.append(f"[{ts}] {text}")

    def _update_status_bar(self) -> None:
        det_ok = "OK" if self.pipeline.yolo.available else "UNAVAILABLE"
        msg = (
            f"Camera: {self.config.camera.source} | "
            f"Target: {self.target_object} | "
            f"YOLO11n: {det_ok} | "
            f"Conf: {self.config.yolo.confidence_threshold:.2f} | "
            f"Output: {self.config.recording.output_dir}/"
        )
        self.statusBar().showMessage(msg)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        if self.session_manager.is_active:
            self.session_manager.close_session(
                final_records=self.engine.step_records,
                all_warnings=self.engine.history_warnings,
                status_override="ABORTED",
            )
        self.recorder.stop_recording()
        if self.streamer:
            self.streamer.stop()
        self.pipeline.stop()
        self.speech.stop()
        super().closeEvent(event)
