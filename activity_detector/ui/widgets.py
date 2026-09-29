"""Reusable custom UI components for ActivityDetector."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from activity_detector.core.engine import EngineState, StepRecord, StepStatus, WarningEvent


STATE_STYLES = {
    EngineState.IDLE: ("#64748b", "#0f172a", "IDLE"),
    EngineState.IN_PROGRESS: ("#0284c7", "#0c4a6e", "IN PROGRESS"),
    EngineState.NEEDS_ATTENTION: ("#dc2626", "#7f1d1d", "NEEDS ATTENTION"),
    EngineState.UNCERTAIN: ("#d97706", "#78350f", "UNCERTAIN (REVIEW)"),
    EngineState.COMPLETED: ("#059669", "#064e3b", "COMPLETED"),
}


class StatusBadgeWidget(QLabel):
    """Pill badge showing the system's operational state."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(30)
        self.setMinimumWidth(160)
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.set_state(EngineState.IDLE)

    def set_state(self, state: EngineState) -> None:
        border_col, bg_col, text = STATE_STYLES.get(state, ("#64748b", "#0f172a", state.value.upper()))
        self.setText(text)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_col};
                color: #ffffff;
                border: 2px solid {border_col};
                border-radius: 15px;
                padding: 4px 14px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }}
        """)


class VideoCanvasWidget(QLabel):
    """Aspect-ratio preserving video canvas."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #0b0f19; border: 1px solid #1e293b; border-radius: 8px;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 270)
        self._current_pixmap: Optional[QPixmap] = None

    def update_frame(self, bgr_frame: np.ndarray) -> None:
        """Converts BGR numpy array to QPixmap and updates canvas."""
        if bgr_frame is None or bgr_frame.size == 0:
            return

        h, w, ch = bgr_frame.shape
        bytes_per_line = ch * w
        # Convert BGR to RGB
        rgb_frame = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._redraw()

    def resizeEvent(self, event) -> None:  # type: ignore
        super().resizeEvent(event)
        self._redraw()

    def _redraw(self) -> None:
        if self._current_pixmap and not self._current_pixmap.isNull():
            scaled = self._current_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)


class StepCardWidget(QFrame):
    """Detailed visual card for the current active procedure step."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 10px;
                padding: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header: Step index & Title
        self.title_label = QLabel("Step 1: Initializing...")
        self.title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #38bdf8;")
        layout.addWidget(self.title_label)

        # Instruction Text
        self.instruction_box = QLabel("Waiting for procedure to begin.")
        self.instruction_box.setWordWrap(True)
        self.instruction_box.setFont(QFont("Segoe UI", 10))
        self.instruction_box.setStyleSheet("color: #e2e8f0; background-color: #1e293b; padding: 10px; border-radius: 6px;")
        layout.addWidget(self.instruction_box)

        # Expected Evidence Tag
        self.evidence_label = QLabel("Expected: None")
        self.evidence_label.setFont(QFont("Segoe UI", 9))
        self.evidence_label.setStyleSheet("color: #94a3b8;")
        self.evidence_label.setWordWrap(True)
        layout.addWidget(self.evidence_label)

        # Stability Progress Bar
        stab_row = QHBoxLayout()
        stab_lbl = QLabel("Multi-frame Stability:")
        stab_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        stab_lbl.setStyleSheet("color: #94a3b8;")
        self.stability_bar = QProgressBar()
        self.stability_bar.setRange(0, 100)
        self.stability_bar.setValue(0)
        self.stability_bar.setFixedHeight(12)
        self.stability_bar.setTextVisible(False)
        self.stability_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1e293b;
                border-radius: 6px;
            }
            QProgressBar::chunk {
                background-color: #10b981;
                border-radius: 6px;
            }
        """)
        self.stability_percent = QLabel("0%")
        self.stability_percent.setFont(QFont("Segoe UI", 9))
        self.stability_percent.setStyleSheet("color: #10b981; min-width: 35px;")

        stab_row.addWidget(stab_lbl)
        stab_row.addWidget(self.stability_bar)
        stab_row.addWidget(self.stability_percent)
        layout.addLayout(stab_row)

    def update_step(
        self,
        order: int,
        total: int,
        name: str,
        instruction: str,
        evidence_summary: str,
        stability_ratio: float,
    ) -> None:
        self.title_label.setText(f"Step {order} of {total}: {name}")
        self.instruction_box.setText(instruction)
        self.evidence_label.setText(f"Expected Evidence: {evidence_summary}")
        pct = int(max(0.0, min(1.0, stability_ratio)) * 100)
        self.stability_bar.setValue(pct)
        self.stability_percent.setText(f"{pct}%")


class StepListWidget(QScrollArea):
    """Timeline checklist of all procedure steps."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet("""
            QScrollArea {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }
        """)

        self.container = QWidget()
        self.container.setStyleSheet("background-color: transparent;")
        self.layout_steps = QVBoxLayout(self.container)
        self.layout_steps.setContentsMargins(8, 8, 8, 8)
        self.layout_steps.setSpacing(6)
        self.layout_steps.addStretch()
        self.setWidget(self.container)

        self._item_labels: List[QLabel] = []

    def set_steps(self, records: List[StepRecord], active_index: int) -> None:
        """Refreshes step timeline items."""
        # Clear existing labels
        for lbl in self._item_labels:
            self.layout_steps.removeWidget(lbl)
            lbl.deleteLater()
        self._item_labels.clear()

        for idx, rec in enumerate(records):
            is_active = (idx == active_index)
            is_completed = (rec.status == StepStatus.COMPLETED)
            has_warnings = len(rec.warnings) > 0

            if is_completed:
                icon = "✓"
                col = "#10b981"
                bg = "#064e3b"
            elif is_active:
                icon = "▶"
                col = "#38bdf8"
                bg = "#0c4a6e"
            elif has_warnings:
                icon = "⚠"
                col = "#f59e0b"
                bg = "#78350f"
            else:
                icon = "○"
                col = "#64748b"
                bg = "#1e293b"

            dur_str = f" ({rec.duration_seconds}s)" if rec.duration_seconds else ""
            txt = f" {icon}  {rec.order}. {rec.name}{dur_str}"

            lbl = QLabel(txt)
            lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            lbl.setStyleSheet(f"""
                QLabel {{
                    color: {col};
                    background-color: {bg};
                    border: 1px solid {col};
                    border-radius: 6px;
                    padding: 6px 10px;
                }}
            """)
            self.layout_steps.insertWidget(len(self._item_labels), lbl)
            self._item_labels.append(lbl)
