"""Procedure Sequence Engine: deterministic step progression, generic object tracking, and error validation."""

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
    target_object: str = "notebook"
    vlm_state: Dict[str, Any] = Field(default_factory=dict)
    vlm_summary: Optional[str] = None
    vlm_confidence: Optional[float] = None
    vlm_uncertain: bool = False
    vlm_sample_id: Optional[str] = None


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
    vlm_sample_count: int = 0
    required_vlm_samples: int = 2


class ProcedureEngine:
    """Deterministic sequence validator and state machine tracking physical object state across time."""

    def __init__(self, procedure: Procedure) -> None:
        self.procedure = procedure
        self.target_object: str = procedure.target_object or "notebook"
        self.state: EngineState = EngineState.IDLE
        self.current_step_index: int = 0
        self.step_records: List[StepRecord] = []
        self.active_warnings: List[WarningEvent] = []
        self.history_warnings: List[WarningEvent] = []

        # Multi-frame stability tracking
        self._consecutive_match_frames: int = 0
        self._stable_start_time: Optional[float] = None

        # Multi-observation VLM sample tracking (ensures single frame cannot advance step)
        self._vlm_consistent_sample_count: int = 0
        self._last_processed_vlm_sample_id: Optional[str] = None

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

    def set_target_object(self, object_name: str) -> None:
        """Updates the target object to monitor (e.g. 'notebook', 'sample container')."""
        self.target_object = object_name.strip() or "notebook"

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
        self._vlm_consistent_sample_count = 0
        self._last_processed_vlm_sample_id = None
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
        evidence_source = "vlm_state_tracking"
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
                vlm_sample_count=self._vlm_consistent_sample_count,
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
                message=f"Step '{current_step.name}' timeout: active for {elapsed_sec}s (limit {int(current_step.timeout_seconds)}s).",
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

        # 3. Handle VLM uncertainty flag or invalid low-confidence response
        if evidence.vlm_uncertain and evidence.vlm_summary:
            if self.state != EngineState.NEEDS_ATTENTION:
                self.state = EngineState.UNCERTAIN
                uncertain_warn = WarningEvent(
                    warning_type=WarningType.UNCERTAIN_EVIDENCE,
                    message=f"Visual evidence ambiguous on Step '{current_step.name}': {evidence.vlm_summary}",
                    step_id=current_step.id,
                    timestamp=now
                )
                self._emit_warning(uncertain_warn)
                recent_warning = uncertain_warn

        # 4. Multi-observation tracking for VLM samples
        # A single frame must NEVER advance a step. If rule uses VLM, require min_vlm_samples.
        req_vlm_samples = current_step.completion_rule.min_vlm_samples
        if evidence.vlm_sample_id and evidence.vlm_sample_id != self._last_processed_vlm_sample_id:
            self._last_processed_vlm_sample_id = evidence.vlm_sample_id
            if current_match and match_conf >= current_step.completion_rule.min_confidence and not evidence.vlm_uncertain:
                self._vlm_consistent_sample_count += 1
            else:
                self._vlm_consistent_sample_count = max(0, self._vlm_consistent_sample_count - 1)

        req_frames = max(1, current_step.completion_rule.stable_frames)
        req_hold = current_step.completion_rule.hold_seconds

        # Check conditions
        vlm_sample_ok = True
        if current_step.completion_rule.rule_type in ("vlm_state_tracking", "vlm_confirmation"):
            vlm_sample_ok = (self._vlm_consistent_sample_count >= req_vlm_samples)

        if current_match and match_conf >= current_step.completion_rule.min_confidence and not evidence.vlm_uncertain:
            if self._consecutive_match_frames == 0:
                self._stable_start_time = now

            self._consecutive_match_frames += 1
            elapsed_hold = (now - self._stable_start_time) if self._stable_start_time else 0.0

            frame_ratio = min(1.0, self._consecutive_match_frames / float(req_frames))
            hold_ratio = 1.0 if req_hold <= 0.0 else min(1.0, elapsed_hold / float(req_hold))
            vlm_ratio = min(1.0, self._vlm_consistent_sample_count / float(max(1, req_vlm_samples)))

            if current_step.completion_rule.rule_type in ("vlm_state_tracking", "vlm_confirmation"):
                stability_ratio = min(frame_ratio, hold_ratio, vlm_ratio)
            else:
                stability_ratio = min(frame_ratio, hold_ratio)

            # Check if all completion criteria satisfied
            if vlm_sample_ok and self._consecutive_match_frames >= req_frames and elapsed_hold >= req_hold:
                # Step Complete!
                completed_step_id = current_step.id
                transition_occurred = True
                self._complete_current_step(now, evidence_source, match_conf)
                # Advance to next step
                self.current_step_index += 1
                self._consecutive_match_frames = 0
                self._stable_start_time = None
                self._vlm_consistent_sample_count = 0
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
            if self._consecutive_match_frames > 0:
                self._consecutive_match_frames = max(0, self._consecutive_match_frames - 2)
            else:
                self._stable_start_time = None
            stability_ratio = min(1.0, self._consecutive_match_frames / float(req_frames))

        # 5. Temporal Sequence Anomaly Checks
        if not transition_occurred and self.state != EngineState.COMPLETED:
            # Check Out-of-Order / Early Stowage Anomaly:
            # If on step 1 (locate) or step 2 (open), but notebook is already closed in stowed_area:
            v_loc = evidence.vlm_state.get("location", "")
            v_state = evidence.vlm_state.get("open_or_closed", "")
            if current_step.order in (1, 2) and v_loc == "stowed_area" and v_state == "closed":
                cnt = self._future_step_match_counters.get("stowed_early", 0) + 1
                self._future_step_match_counters["stowed_early"] = cnt
                if cnt >= 4:
                    warn = WarningEvent(
                        warning_type=WarningType.OUT_OF_ORDER,
                        message=(
                            f"Sequence anomaly: {self.target_object.capitalize()} observed in stowed area "
                            f"before completing open/close verification sequence!"
                        ),
                        step_id=current_step.id,
                        timestamp=now,
                        details={"location": v_loc, "open_or_closed": v_state}
                    )
                    self._emit_warning(warn)
                    recent_warning = warn
                    self.state = EngineState.NEEDS_ATTENTION
                    self._future_step_match_counters["stowed_early"] = 0
            else:
                self._future_step_match_counters["stowed_early"] = 0

            # Check skipped steps across procedure
            for future_idx in range(self.current_step_index + 1, len(self.procedure.steps)):
                future_step = self.procedure.steps[future_idx]
                f_match, f_conf, _, _ = self._evaluate_step_evidence(future_step, evidence)
                if f_match and f_conf >= future_step.completion_rule.min_confidence:
                    cnt = self._future_step_match_counters.get(future_step.id, 0) + 1
                    self._future_step_match_counters[future_step.id] = cnt
                    if cnt >= 5:
                        warn = WarningEvent(
                            warning_type=WarningType.SKIPPED_STEP,
                            message=(
                                f"Sequence warning: Detected evidence for future Step {future_step.order} "
                                f"('{future_step.name}') while current Step {current_step.order} "
                                f"('{current_step.name}') is incomplete!"
                            ),
                            step_id=future_step.id,
                            timestamp=now,
                            details={"future_step_id": future_step.id, "confidence": f_conf}
                        )
                        self._emit_warning(warn)
                        recent_warning = warn
                        self.state = EngineState.NEEDS_ATTENTION
                        self._future_step_match_counters[future_step.id] = 0
                        break
                else:
                    self._future_step_match_counters[future_step.id] = max(0, self._future_step_match_counters.get(future_step.id, 0) - 1)

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
            vlm_sample_count=self._vlm_consistent_sample_count,
            required_vlm_samples=req_vlm_samples,
        )

    def _evaluate_step_evidence(
        self,
        step: StepDefinition,
        evidence: FrameEvidence,
    ) -> Tuple[bool, float, str, str]:
        """Evaluates whether frame evidence satisfies a step's requirements."""
        exp = step.expected_evidence
        confidences: List[float] = []
        source = "deterministic_rules"
        desc_parts: List[str] = []

        is_match = False
        rule_type = step.completion_rule.rule_type

        # 1. Evaluate VLM State Tracking (primary method for generic objects)
        if exp.expected_state or rule_type in ("vlm_state_tracking", "vlm_confirmation"):
            source = "vlm_ollama"
            v_state = evidence.vlm_state
            state_match = True

            # Target object visible check
            if exp.expected_state.get("object_visible", True):
                if not v_state.get("object_visible", False):
                    state_match = False
                else:
                    desc_parts.append(f"{self.target_object.capitalize()} visible")

            # Open vs closed state check
            expected_oc = exp.expected_state.get("open_or_closed")
            if expected_oc:
                actual_oc = v_state.get("open_or_closed", "unknown")
                if actual_oc.lower() != expected_oc.lower():
                    state_match = False
                else:
                    desc_parts.append(f"State: {actual_oc}")

            # Location / ROI check
            expected_loc = exp.expected_state.get("location")
            if expected_loc:
                actual_loc = v_state.get("location", "unknown")
                if actual_loc.lower() != expected_loc.lower():
                    state_match = False
                else:
                    desc_parts.append(f"Location: {actual_loc}")

            # Keyword support
            if exp.vlm_keywords and evidence.vlm_summary:
                txt = evidence.vlm_summary.lower()
                if any(kw.lower() in txt for kw in exp.vlm_keywords):
                    desc_parts.append("Keywords matched")

            if evidence.vlm_confidence is not None:
                confidences.append(evidence.vlm_confidence)

            if evidence.vlm_uncertain:
                state_match = False

            is_match = state_match

        # 2. Supporting ROI containment / Color / Object checks
        if exp.required_colors or exp.roi or exp.required_objects:
            matched_colors = []
            for req_color in exp.required_colors:
                matches = [d for d in evidence.detections if d.source == "color_roi" and d.name == req_color]
                if exp.roi:
                    matches = [d for d in matches if d.roi == exp.roi]
                if matches:
                    matched_colors.append(req_color)
                    confidences.append(max(d.confidence for d in matches))

            matched_objects = []
            for req_obj in exp.required_objects:
                matches = [d for d in evidence.detections if d.source in ("yolo", "detector") and d.name.lower() == req_obj.lower()]
                if exp.roi:
                    matches = [d for d in matches if d.roi == exp.roi]
                if matches:
                    matched_objects.append(req_obj)
                    confidences.append(max(d.confidence for d in matches))

            colors_ok = (len(matched_colors) == len(exp.required_colors)) if exp.required_colors else True
            objects_ok = (len(matched_objects) == len(exp.required_objects)) if exp.required_objects else True

            if matched_colors:
                desc_parts.append(f"Colors: {', '.join(matched_colors)}")
            if matched_objects:
                desc_parts.append(f"Objects: {', '.join(matched_objects)}")

            if rule_type == "stable_detection":
                is_match = colors_ok and objects_ok
                source = "color_roi" if matched_colors else "detector"
            elif rule_type == "hybrid":
                is_match = is_match and colors_ok and objects_ok
                source = "hybrid"

        avg_conf = (sum(confidences) / len(confidences)) if confidences else (0.80 if is_match else 0.0)
        desc = " | ".join(desc_parts) if desc_parts else "Awaiting expected physical evidence..."
        return is_match, round(avg_conf, 2), desc, source

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

    def confirm_current_step(self, reason: str = "operator_confirmed") -> bool:
        """Operator explicitly verifies and confirms the current step."""
        return self.advance_step(reason=reason)

    def flag_uncertain_step(self, reason: str = "operator_flagged_uncertain") -> bool:
        """Operator flags current observation as ambiguous or inconclusive."""
        if self.state == EngineState.IDLE or self.current_step_index >= len(self.procedure.steps):
            return False

        curr_step = self.procedure.steps[self.current_step_index]
        warn = WarningEvent(
            warning_type=WarningType.UNCERTAIN_EVIDENCE,
            message=f"Operator flagged Step '{curr_step.name}' as ambiguous ({reason}).",
            step_id=curr_step.id,
            timestamp=time.time()
        )
        self._emit_warning(warn)
        self.state = EngineState.UNCERTAIN
        return True

    def advance_step(self, reason: str = "operator_override") -> bool:
        """Manually advances to the next step via operator override."""
        if self.state == EngineState.IDLE or self.current_step_index >= len(self.procedure.steps):
            return False

        now = time.time()
        curr_step = self.procedure.steps[self.current_step_index]
        self._complete_current_step(now, evidence_source=reason, confidence=1.0)

        warn = WarningEvent(
            warning_type=WarningType.OPERATOR_OVERRIDE,
            message=f"Operator manually advanced Step '{curr_step.name}' ({reason}).",
            step_id=curr_step.id,
            timestamp=now
        )
        self._emit_warning(warn)

        self.current_step_index += 1
        self._consecutive_match_frames = 0
        self._stable_start_time = None
        self._vlm_consistent_sample_count = 0
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
        self._vlm_consistent_sample_count = 0
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
