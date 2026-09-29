"""Core domain module for procedures, sequence engine, and session management."""

from activity_detector.core.procedure import (
    Procedure,
    StepDefinition,
    ExpectedEvidence,
    CompletionRule,
    load_procedure,
    save_procedure,
)
from activity_detector.core.engine import (
    ProcedureEngine,
    EngineState,
    StepStatus,
    WarningType,
    WarningEvent,
    StepRecord,
    DetectionItem,
    FrameEvidence,
    EngineUpdate,
)
from activity_detector.core.session import (
    SessionManager,
    LogEvent,
    SessionSummary,
)

__all__ = [
    "Procedure",
    "StepDefinition",
    "ExpectedEvidence",
    "CompletionRule",
    "load_procedure",
    "save_procedure",
    "ProcedureEngine",
    "EngineState",
    "StepStatus",
    "WarningType",
    "WarningEvent",
    "StepRecord",
    "DetectionItem",
    "FrameEvidence",
    "EngineUpdate",
    "SessionManager",
    "LogEvent",
    "SessionSummary",
]
