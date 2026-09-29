"""Session logging, audit trail generation, and run artifact management."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from activity_detector.core.engine import EngineUpdate, StepRecord, WarningEvent
from activity_detector.core.procedure import Procedure

logger = logging.getLogger("activity_detector.session")


class LogEvent(BaseModel):
    """Structured line item written to session_log.jsonl."""
    timestamp: float = Field(default_factory=time.time)
    iso_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str
    session_id: str
    step_id: Optional[str] = None
    step_order: Optional[int] = None
    state: Optional[str] = None
    evidence_source: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionSummary(BaseModel):
    """Final consolidated session report saved to session_summary.json."""
    session_id: str
    procedure_id: str
    procedure_title: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    total_steps: int
    completed_steps: int
    compliance_status: str  # "COMPLETED_CLEAN", "COMPLETED_WITH_WARNINGS", "INCOMPLETE", "ABORTED"
    step_records: List[StepRecord] = Field(default_factory=list)
    warning_count: int = 0
    warnings: List[WarningEvent] = Field(default_factory=list)
    video_path: Optional[str] = None
    video_recording_success: bool = True
    system_notes: str = ""


class SessionManager:
    """Manages local run directories, streaming JSONL audit logs, and summary reports."""

    def __init__(self, output_root: str = "runs") -> None:
        self.output_root = Path(output_root)
        self.current_session_id: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self._log_file_handle = None
        self._start_time: Optional[float] = None
        self._procedure: Optional[Procedure] = None
        self._video_path: Optional[Path] = None
        self._video_recording_success: bool = True
        self._active_step_id: Optional[str] = None
        self._last_logged_frame_sec: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.current_session_id is not None and self._log_file_handle is not None

    def start_session(self, procedure: Procedure, custom_session_id: Optional[str] = None) -> Path:
        """Initializes a new session directory and begins JSONL logging."""
        self._procedure = procedure
        self._start_time = time.time()
        self._video_recording_success = True

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid_short = str(uuid.uuid4())[:8]
        self.current_session_id = custom_session_id or f"session_{timestamp_str}_{uid_short}"

        self.session_dir = self.output_root / self.current_session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.session_dir / "session_log.jsonl"
        self._log_file_handle = open(log_path, "a", encoding="utf-8")

        self.log_event(
            event_type="SESSION_STARTED",
            message=f"Started monitoring session for procedure '{procedure.title}'",
            metadata={
                "procedure_id": procedure.id,
                "procedure_version": procedure.version,
                "total_steps": len(procedure.steps),
            }
        )

        logger.info(f"Session started: {self.current_session_id} -> {self.session_dir}")
        return self.session_dir

    def log_event(
        self,
        event_type: str,
        step_id: Optional[str] = None,
        step_order: Optional[int] = None,
        state: Optional[str] = None,
        evidence_source: Optional[str] = None,
        confidence: Optional[float] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Appends a structured event to the session JSONL log with immediate flush."""
        if not self._log_file_handle or not self.current_session_id:
            return

        event = LogEvent(
            timestamp=time.time(),
            iso_time=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            session_id=self.current_session_id,
            step_id=step_id,
            step_order=step_order,
            state=state,
            evidence_source=evidence_source,
            confidence=confidence,
            message=message,
            metadata=metadata or {},
        )

        try:
            line = event.model_dump_json() + "\n"
            self._log_file_handle.write(line)
            self._log_file_handle.flush()
        except Exception as e:
            logger.error(f"Failed writing session log event: {e}")

    def log_engine_update(self, update: EngineUpdate) -> None:
        """Logs transitions, warnings, and sampled frame observations from the engine."""
        if not self.is_active:
            return

        # Check for step transition
        if update.transition_occurred and update.completed_step_id:
            self.log_event(
                event_type="STEP_COMPLETED",
                step_id=update.completed_step_id,
                state=update.state.value,
                evidence_source=update.evidence_source,
                message=f"Completed step '{update.completed_step_id}'",
                metadata={"evidence_summary": update.evidence_summary}
            )

        # Log new warning if present
        if update.recent_warning:
            self.log_event(
                event_type="WARNING_TRIGGERED",
                step_id=update.recent_warning.step_id,
                state=update.state.value,
                message=update.recent_warning.message,
                metadata={
                    "warning_type": update.recent_warning.warning_type.value,
                    "details": update.recent_warning.details,
                }
            )

        # Log periodic frame telemetry (once every 1.5 seconds)
        now = time.time()
        if (now - self._last_logged_frame_sec) >= 1.5 and update.current_step:
            self._last_logged_frame_sec = now
            self.log_event(
                event_type="OBSERVATION_SAMPLE",
                step_id=update.current_step.id,
                step_order=update.current_step.order,
                state=update.state.value,
                evidence_source=update.evidence_source,
                confidence=round(update.stability_ratio, 3),
                metadata={
                    "evidence_summary": update.evidence_summary,
                    "stability_ratio": round(update.stability_ratio, 3),
                }
            )

    def record_video_status(self, video_path: Path, success: bool = True, error_msg: str = "") -> None:
        """Notes the video file output and records any capture/encoding failures."""
        self._video_path = video_path
        self._video_recording_success = success
        if not success:
            self.log_event(
                event_type="VIDEO_RECORDING_ERROR",
                message=f"Video recording failed: {error_msg}",
                metadata={"error": error_msg}
            )

    def close_session(
        self,
        final_records: List[StepRecord],
        all_warnings: List[WarningEvent],
        status_override: Optional[str] = None
    ) -> Optional[Path]:
        """Closes JSONL file and creates the final session_summary.json report."""
        if not self.is_active or not self.session_dir:
            return None

        now = time.time()
        end_iso = datetime.now(timezone.utc).isoformat()
        duration = round(now - (self._start_time or now), 2)

        completed_count = sum(1 for r in final_records if r.status.value == "completed")
        total_steps = len(self._procedure.steps) if self._procedure else len(final_records)

        if status_override:
            compliance = status_override
        elif completed_count == total_steps and not all_warnings:
            compliance = "COMPLETED_CLEAN"
        elif completed_count == total_steps:
            compliance = "COMPLETED_WITH_WARNINGS"
        else:
            compliance = "INCOMPLETE"

        summary = SessionSummary(
            session_id=self.current_session_id or "unknown",
            procedure_id=self._procedure.id if self._procedure else "unknown",
            procedure_title=self._procedure.title if self._procedure else "unknown",
            start_time=datetime.fromtimestamp(self._start_time or now, tz=timezone.utc).isoformat(),
            end_time=end_iso,
            duration_seconds=duration,
            total_steps=total_steps,
            completed_steps=completed_count,
            compliance_status=compliance,
            step_records=final_records,
            warning_count=len(all_warnings),
            warnings=all_warnings,
            video_path=str(self._video_path.name) if self._video_path else None,
            video_recording_success=self._video_recording_success,
            system_notes=(
                "Session recorded offline via ActivityDetector. "
                "Step transitions validated deterministically with multi-frame stability."
            )
        )

        self.log_event(
            event_type="SESSION_CLOSED",
            message=f"Session closed with compliance: {compliance}",
            metadata={"completed_steps": completed_count, "total_steps": total_steps, "duration": duration}
        )

        try:
            if self._log_file_handle:
                self._log_file_handle.close()
                self._log_file_handle = None
        except Exception as e:
            logger.error(f"Error closing log file handle: {e}")

        summary_path = self.session_dir / "session_summary.json"
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary.model_dump_json(indent=2))
            logger.info(f"Saved session summary to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to write session summary: {e}")

        self.current_session_id = None
        return summary_path
