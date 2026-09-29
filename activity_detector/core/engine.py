"""Procedure Sequence Engine: deterministic step progression and error tracking."""

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from activity_detector.core.procedure import Procedure, StepDefinition


class EngineState(str, Enum):
    """Macro state of the monitoring session."""
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    NEEDS_ATTENTION = "needs_attention"
    UNCERTAIN = "uncertain"
    COMPLETED = "completed"


class StepStatus(str, Enum):
    """Lifecycle status of an individual step."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FLAGGED = "flagged"


class WarningType(str, Enum):
    """Categorized sequence warnings."""
    SKIPPED_STEP = "skipped_step"
    OUT_OF_ORDER = "out_of_order"
    REPEATED_STEP = "repeated_step"
    STEP_TIMEOUT = "step_timeout"
    UNCERTAIN_EVIDENCE = "uncertain_evidence"
    OPERATOR_OVERRIDE = "operator_override"


class WarningEvent(BaseModel):
    """Structured record of a warning or sequence anomaly."""
    warning_type: WarningType
    message: str
    step_id: str
    timestamp: float = Field(default_factory=time.time)
    acknowledged: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class StepRecord(BaseModel):
    """Audit record for a step within an active session."""
    step_id: str
    order: int
    name: str
    instruction: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    evidence_source: str = "none"
    confidence: float = 0.0
    stable_frames_observed: int = 0
    warnings: List[str] = Field(default_factory=list)


class DetectionItem(BaseModel):
    """Unified representation of a detected feature."""
    name: str
    source: str  # "color_roi", "yolo", "vlm"
    roi: Optional[str] = None
    confidence: float = 1.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    area: int = 0
    attributes: Dict[str, Any] = Field(default_factory=dict)


class FrameEvidence(BaseModel):
    """Aggregated evidence produced by the vision pipeline for a single frame."""
    frame_number: int
    timestamp: float = Field(default_factory=time.time)
    detections: List[DetectionItem] = Field(default_factory=list)
    vlm_summary: Optional[str] = None
    vlm_confidence: Optional[float] = None
    vlm_uncertain: bool = False


class EngineUpdate(BaseModel):
    """Snapshot update emitted on every frame evaluation."""
    state: EngineState
    current_step_index: int
    current_step: Optional[StepDefinition] = None
    next_step: Optional[StepDefinition] = None
    progress_percentage: float = 0.0
    step_records: List[StepRecord] = Field(default_factory=list)
    recent_warning: Optional[WarningEvent] = None
    stability_ratio: float = 0.0  # 0.0 to 1.0 towards step completion
    evidence_summary: str = ""
    evidence_source: str = "deterministic_rules"
    transition_occurred: bool = False
    completed_step_id: Optional[str] = None


class ProcedureEngine:
    """Deterministic sequence validator and state machine."""

    def __init__(self, procedure: Procedure) -> None:
        self.procedure = procedure
        self.state: EngineState = EngineState.IDLE
        self.current_step_index: int = 0
        self.step_records: List[StepRecord] = []
        self.active_warnings: List[WarningEvent] = []
        self.history_warnings: List[WarningEvent] = []

        # Multi-frame stability tracking
        self._consecutive_match_frames: int = 0
        self._stable_start_time: Optional[float] = None

        # Look-ahead stability tracking for skipped steps
        self._future_step_match_counters: Dict[str, int] = {}

        # Look-behind stability tracking for repeated steps
        self._past_step_match_counters: Dict[str, int] = {}

        # Timeout tracking
        self._current_step_start_time: Optional[float] = None
        self._timeout_warned_steps: set[str] = set()

        # Listeners for reactive events
        self._on_transition_listeners: List[Callable[[StepRecord], None]] = []
        self._on_warning_listeners: List[Callable[[WarningEvent], None]] = []

        self._initialize_records()

    def _initialize_records(self) -> None:
        """Initializes empty step records matching the procedure."""
        self.step_records = [
            StepRecord(
                step_id=step.id,
                order=step.order,
                name=step.name,
                instruction=step.instruction,
                status=StepStatus.PENDING,
            )
            for step in self.procedure.steps
        ]

    def register_transition_listener(self, callback: Callable[[StepRecord], None]) -> None:
        """Adds a callback invoked when a step completes and transitions."""
        self._on_transition_listeners.append(callback)

    def register_warning_listener(self, callback: Callable[[WarningEvent], None]) -> None:
        """Adds a callback invoked when an anomaly or warning occurs."""
        self._on_warning_listeners.append(callback)

    def start_session(self) -> None:
        """Starts monitoring the procedure from step 1."""
        self._initialize_records()
        self.current_step_index = 0
        self.state = EngineState.IN_PROGRESS
        self.active_warnings.clear()
        self.history_warnings.clear()
        self._consecutive_match_frames = 0
        self._stable_start_time = None
        self._future_step_match_counters.clear()
        self._past_step_match_counters.clear()
        self._timeout_warned_steps.clear()

        now = time.time()
        self._current_step_start_time = now
        if self.step_records:
            self.step_records[0].status = StepStatus.ACTIVE
            self.step_records[0].start_time = now

    def stop_session(self) -> None:
        """Stops the active monitoring session."""
        now = time.time()
        if (
            self.state in (EngineState.IN_PROGRESS, EngineState.NEEDS_ATTENTION, EngineState.UNCERTAIN)
            and 0 <= self.current_step_index < len(self.step_records)
        ):
            rec = self.step_records[self.current_step_index]
            if rec.status == StepStatus.ACTIVE:
                rec.end_time = now
                if rec.start_time:
                    rec.duration_seconds = now - rec.start_time

        self.state = EngineState.IDLE

    def process_frame(self, evidence: FrameEvidence) -> EngineUpdate:
        """Processes evidence for a single frame and updates engine state."""
        transition_occurred = False
        completed_step_id: Optional[str] = None
        recent_warning: Optional[WarningEvent] = None
        evidence_summary = ""
        evidence_source = "deterministic_rules"
        stability_ratio = 0.0

        if self.state == EngineState.IDLE or self.current_step_index >= len(self.procedure.steps):
            current_step = self.procedure.get_step_by_index(self.current_step_index)
            next_step = self.procedure.get_step_by_index(self.current_step_index + 1)
            progress = 100.0 if self.state == EngineState.COMPLETED else 0.0
            return EngineUpdate(
                state=self.state,
                current_step_index=self.current_step_index,
                current_step=current_step,
                next_step=next_step,
                progress_percentage=progress,
                step_records=self.step_records,
                recent_warning=None,
                stability_ratio=0.0,
                evidence_summary="Session Idle" if self.state == EngineState.IDLE else "Procedure Completed",
                evidence_source="none",
                transition_occurred=False,
            )

        current_step = self.procedure.steps[self.current_step_index]
        next_step = self.procedure.get_step_by_index(self.current_step_index + 1)
        now = evidence.timestamp

        # 1. Step Timeout Verification
        if (
            current_step.timeout_seconds
            and self._current_step_start_time
            and (now - self._current_step_start_time) > current_step.timeout_seconds
            and current_step.id not in self._timeout_warned_steps
        ):
            elapsed_sec = int(now - self._current_step_start_time)
            warning = WarningEvent(
                warning_type=WarningType.STEP_TIMEOUT,
                message=f"Step '{current_step.name}' timeout: running for {elapsed_sec}s (limit {int(current_step.timeout_seconds)}s).",
                step_id=current_step.id,
                timestamp=now,
                details={"elapsed_seconds": elapsed_sec, "timeout_seconds": current_step.timeout_seconds}
            )
            self._emit_warning(warning)
            self._timeout_warned_steps.add(current_step.id)
            recent_warning = warning
            if self.state == EngineState.IN_PROGRESS:
                self.state = EngineState.NEEDS_ATTENTION

        # 2. Evaluate current step evidence
        current_match, match_conf, match_desc, current_source = self._evaluate_step_evidence(current_step, evidence)
        evidence_summary = match_desc
        evidence_source = current_source

        # 3. Handle VLM uncertainty flag
        if evidence.vlm_uncertain and evidence.vlm_summary:
            if self.state != EngineState.NEEDS_ATTENTION:
                self.state = EngineState.UNCERTAIN
                uncertain_warn = WarningEvent(
                    warning_type=WarningType.UNCERTAIN_EVIDENCE,
                    message=f"VLM uncertain on Step '{current_step.name}': {evidence.vlm_summary}",
                    step_id=current_step.id,
                    timestamp=now
                )
                self._emit_warning(uncertain_warn)
                recent_warning = uncertain_warn

        # 4. Multi-frame stability accumulation for current step
        req_frames = max(1, current_step.completion_rule.stable_frames)
        req_hold = current_step.completion_rule.hold_seconds

        if current_match and match_conf >= current_step.completion_rule.min_confidence:
            if self._consecutive_match_frames == 0:
                self._stable_start_time = now

            self._consecutive_match_frames += 1
            elapsed_hold = (now - self._stable_start_time) if self._stable_start_time else 0.0

            frame_ratio = min(1.0, self._consecutive_match_frames / float(req_frames))
            hold_ratio = 1.0 if req_hold <= 0.0 else min(1.0, elapsed_hold / float(req_hold))
            stability_ratio = min(frame_ratio, hold_ratio)

            # Check if completion condition satisfied
            if self._consecutive_match_frames >= req_frames and elapsed_hold >= req_hold:
                # Step Complete!
                completed_step_id = current_step.id
                transition_occurred = True
                self._complete_current_step(now, evidence_source, match_conf)
                # Advance to next step
                self.current_step_index += 1
                self._consecutive_match_frames = 0
                self._stable_start_time = None
                self._current_step_start_time = now

                if self.current_step_index >= len(self.procedure.steps):
                    self.state = EngineState.COMPLETED
                    stability_ratio = 1.0
                else:
                    self.state = EngineState.IN_PROGRESS
                    self.step_records[self.current_step_index].status = StepStatus.ACTIVE
                    self.step_records[self.current_step_index].start_time = now
                    current_step = self.procedure.steps[self.current_step_index]
                    next_step = self.procedure.get_step_by_index(self.current_step_index + 1)
                    stability_ratio = 0.0
        else:
            # Decay stability gracefully instead of instant cliff on single flicker
            if self._consecutive_match_frames > 0:
                self._consecutive_match_frames = max(0, self._consecutive_match_frames - 2)
            else:
                self._stable_start_time = None
            stability_ratio = min(1.0, self._consecutive_match_frames / float(req_frames))

        # 5. Check for sequence errors (Skipped steps & Repeated steps)
        if not transition_occurred and self.state != EngineState.COMPLETED:
            # Check skipped steps (look-ahead into future steps k+1, k+2, etc.)
            for future_idx in range(self.current_step_index + 1, len(self.procedure.steps)):
                future_step = self.procedure.steps[future_idx]
                f_match, f_conf, _, _ = self._evaluate_step_evidence(future_step, evidence)
                if f_match and f_conf >= 0.70:
                    cnt = self._future_step_match_counters.get(future_step.id, 0) + 1
                    self._future_step_match_counters[future_step.id] = cnt
                    # If future step observed consistently for 6+ frames while current incomplete:
                    if cnt >= 6:
                        warn = WarningEvent(
                            warning_type=WarningType.SKIPPED_STEP,
                            message=(
                                f"Sequence warning: Detected evidence for future Step {future_step.order} "
                                f"('{future_step.name}') while current Step {current_step.order} "
                                f"('{current_step.name}') is incomplete!"
                            ),
                            step_id=current_step.id,
                            timestamp=now,
                            details={"future_step_id": future_step.id, "confidence": f_conf}
                        )
                        self._emit_warning(warn)
                        recent_warning = warn
                        self.state = EngineState.NEEDS_ATTENTION
                        self._future_step_match_counters[future_step.id] = 0
                else:
                    self._future_step_match_counters[future_step.id] = 0

            # Check repeated / out-of-order steps (look-behind into already completed steps)
            for past_idx in range(0, self.current_step_index):
                past_step = self.procedure.steps[past_idx]
                p_match, p_conf, _, _ = self._evaluate_step_evidence(past_step, evidence)
                # Avoid flagging if current step has overlapping colors
                overlapping_colors = set(current_step.expected_evidence.required_colors) & set(
                    past_step.expected_evidence.required_colors
                )
                if p_match and p_conf >= 0.75 and not overlapping_colors:
                    cnt = self._past_step_match_counters.get(past_step.id, 0) + 1
                    self._past_step_match_counters[past_step.id] = cnt
                    if cnt >= 8:
                        warn = WarningEvent(
                            warning_type=WarningType.REPEATED_STEP,
                            message=(
                                f"Repeated action alert: Evidence matches already-completed Step {past_step.order} "
                                f"('{past_step.name}')."
                            ),
                            step_id=current_step.id,
                            timestamp=now,
                            details={"past_step_id": past_step.id}
                        )
                        self._emit_warning(warn)
                        recent_warning = warn
                        self._past_step_match_counters[past_step.id] = 0
                else:
                    self._past_step_match_counters[past_step.id] = 0

        # Progress calculation
        completed_count = sum(1 for r in self.step_records if r.status == StepStatus.COMPLETED)
        progress_percentage = (completed_count / float(len(self.step_records))) * 100.0

        return EngineUpdate(
            state=self.state,
            current_step_index=self.current_step_index,
            current_step=current_step,
            next_step=next_step,
            progress_percentage=progress_percentage,
            step_records=self.step_records,
            recent_warning=recent_warning,
            stability_ratio=stability_ratio,
            evidence_summary=evidence_summary,
            evidence_source=evidence_source,
            transition_occurred=transition_occurred,
            completed_step_id=completed_step_id,
        )

    def _evaluate_step_evidence(
        self,
        step: StepDefinition,
        evidence: FrameEvidence,
    ) -> Tuple[bool, float, str, str]:
        """Evaluates whether frame evidence satisfies a step's requirements."""
        exp = step.expected_evidence
        matched_colors: List[str] = []
        matched_objects: List[str] = []
        confidences: List[float] = []
        source = "deterministic_rules"

        # Check required colors and target ROI
        for req_color in exp.required_colors:
            color_matches = [
                d for d in evidence.detections
                if d.source == "color_roi" and d.name == req_color
            ]
            if exp.roi:
                color_matches = [d for d in color_matches if d.roi == exp.roi]

            if color_matches:
                best = max(color_matches, key=lambda d: d.confidence)
                matched_colors.append(req_color)
                confidences.append(best.confidence)

        # Check required objects (from YOLO or other object detectors)
        for req_obj in exp.required_objects:
            obj_matches = [
                d for d in evidence.detections
                if d.source in ("yolo", "detector") and d.name.lower() == req_obj.lower()
            ]
            if exp.roi:
                obj_matches = [d for d in obj_matches if d.roi == exp.roi]

            if obj_matches:
                best = max(obj_matches, key=lambda d: d.confidence)
                matched_objects.append(req_obj)
                confidences.append(best.confidence)

        # Check VLM hints if available
        vlm_support = False
        if exp.vlm_keywords and evidence.vlm_summary:
            vlm_text = evidence.vlm_summary.lower()
            if any(k.lower() in vlm_text for k in exp.vlm_keywords):
                vlm_support = True
                source = "hybrid"
                if evidence.vlm_confidence:
                    confidences.append(evidence.vlm_confidence)

        # Determine satisfaction
        colors_ok = (len(matched_colors) == len(exp.required_colors))
        objects_ok = (len(matched_objects) == len(exp.required_objects))

        is_match = False
        rule_type = step.completion_rule.rule_type

        if rule_type == "stable_detection":
            is_match = colors_ok and objects_ok
        elif rule_type == "vlm_confirmation":
            is_match = vlm_support and colors_ok
            source = "vlm_ollama"
        elif rule_type == "hybrid":
            is_match = colors_ok and (objects_ok or vlm_support)
            source = "hybrid"
        else:
            is_match = colors_ok and objects_ok

        avg_conf = (sum(confidences) / len(confidences)) if confidences else (0.0 if not is_match else 0.8)

        desc_parts = []
        if matched_colors:
            roi_str = f" in {exp.roi}" if exp.roi else ""
            desc_parts.append(f"Colors: {', '.join(matched_colors)}{roi_str}")
        if matched_objects:
            desc_parts.append(f"Objects: {', '.join(matched_objects)}")
        if vlm_support:
            desc_parts.append("VLM confirmed keywords")

        desc = " | ".join(desc_parts) if desc_parts else "Awaiting expected evidence..."
        return is_match, avg_conf, desc, source

    def _complete_current_step(self, timestamp: float, evidence_source: str, confidence: float) -> None:
        """Records completion for the current active step and fires listeners."""
        if 0 <= self.current_step_index < len(self.step_records):
            rec = self.step_records[self.current_step_index]
            rec.status = StepStatus.COMPLETED
            rec.end_time = timestamp
            if rec.start_time:
                rec.duration_seconds = round(timestamp - rec.start_time, 2)
            rec.evidence_source = evidence_source
            rec.confidence = round(confidence, 3)
            rec.stable_frames_observed = self._consecutive_match_frames

            for listener in self._on_transition_listeners:
                try:
                    listener(rec)
                except Exception:
                    pass

    def _emit_warning(self, warning: WarningEvent) -> None:
        """Emits a warning event, adds it to active list, and triggers listeners."""
        self.active_warnings.append(warning)
        self.history_warnings.append(warning)
        if 0 <= self.current_step_index < len(self.step_records):
            self.step_records[self.current_step_index].warnings.append(warning.message)

        for listener in self._on_warning_listeners:
            try:
                listener(warning)
            except Exception:
                pass

    def advance_step(self, reason: str = "operator_override") -> bool:
        """Manually advances to the next step via operator override."""
        if self.state == EngineState.IDLE or self.current_step_index >= len(self.procedure.steps):
            return False

        now = time.time()
        curr_step = self.procedure.steps[self.current_step_index]
        self._complete_current_step(now, evidence_source=reason, confidence=1.0)

        warn = WarningEvent(
            warning_type=WarningType.OPERATOR_OVERRIDE,
            message=f"Operator manually advanced step '{curr_step.name}' ({reason}).",
            step_id=curr_step.id,
            timestamp=now
        )
        self._emit_warning(warn)

        self.current_step_index += 1
        self._consecutive_match_frames = 0
        self._stable_start_time = None
        self._current_step_start_time = now

        if self.current_step_index >= len(self.procedure.steps):
            self.state = EngineState.COMPLETED
        else:
            self.state = EngineState.IN_PROGRESS
            self.step_records[self.current_step_index].status = StepStatus.ACTIVE
            self.step_records[self.current_step_index].start_time = now

        return True

    def revert_step(self, reason: str = "operator_override") -> bool:
        """Manually reverts to the previous step via operator override."""
        if self.current_step_index <= 0:
            return False

        now = time.time()
        # Reset current step
        if self.current_step_index < len(self.step_records):
            self.step_records[self.current_step_index].status = StepStatus.PENDING
            self.step_records[self.current_step_index].start_time = None

        self.current_step_index -= 1
        prev_record = self.step_records[self.current_step_index]
        prev_record.status = StepStatus.ACTIVE
        prev_record.end_time = None
        prev_record.duration_seconds = None

        self.state = EngineState.IN_PROGRESS
        self._consecutive_match_frames = 0
        self._stable_start_time = None
        self._current_step_start_time = now

        warn = WarningEvent(
            warning_type=WarningType.OPERATOR_OVERRIDE,
            message=f"Operator reverted to Step '{prev_record.name}' ({reason}).",
            step_id=prev_record.step_id,
            timestamp=now
        )
        self._emit_warning(warn)
        return True

    def acknowledge_warnings(self) -> None:
        """Acknowledges all active warnings and resets state to IN_PROGRESS if valid."""
        for w in self.active_warnings:
            w.acknowledged = True
        self.active_warnings.clear()

        if self.state in (EngineState.NEEDS_ATTENTION, EngineState.UNCERTAIN):
            if self.current_step_index < len(self.procedure.steps):
                self.state = EngineState.IN_PROGRESS
            else:
                self.state = EngineState.COMPLETED
