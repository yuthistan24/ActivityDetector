"""Experiment Procedure schema and validation engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ExpectedEvidence(BaseModel):
    """Specification of observable evidence expected for a step."""
    target_object: Optional[str] = Field(
        default=None,
        description="Target object name (e.g. 'notebook', 'sample container')."
    )
    expected_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected object physical state: {'object_visible': True, 'open_or_closed': 'open', 'location': 'workspace_center'}."
    )
    required_colors: List[str] = Field(
        default_factory=list,
        description="Optional supporting color keys (e.g. ['blue_reagent', 'yellow_flask'])."
    )
    required_objects: List[str] = Field(
        default_factory=list,
        description="Object class names detected by YOLO or detector tags."
    )
    roi: Optional[str] = Field(
        default=None,
        description="Target ROI key (e.g. 'workspace_center', 'stowed_area')."
    )
    vlm_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords expected in VLM narrative analysis (e.g. ['open', 'pages', 'closed'])."
    )


class CompletionRule(BaseModel):
    """Conditions under which a step is declared completed."""
    rule_type: str = Field(
        default="stable_detection",
        description="One of: 'vlm_state_tracking', 'stable_detection', 'vlm_confirmation', 'duration_hold', 'hybrid'"
    )
    min_vlm_samples: int = Field(
        default=0,
        ge=0,
        le=20,
        description="Minimum separate VLM observation samples required across time to confirm step."
    )
    stable_frames: int = Field(
        default=8,
        ge=1,
        le=120,
        description="Consecutive frames required showing matching evidence."
    )
    min_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required."
    )
    hold_seconds: float = Field(
        default=1.5,
        ge=0.0,
        le=60.0,
        description="Minimum duration evidence must persist in seconds."
    )


class StepDefinition(BaseModel):
    """Specification for an individual experiment step."""
    id: str = Field(..., description="Unique step identifier, e.g. 'step_1_locate_notebook'")
    order: int = Field(..., ge=1, description="1-based sequence order")
    name: str = Field(..., min_length=2, description="Short human-readable step name")
    instruction: str = Field(..., min_length=5, description="Clear, detailed guidance for the operator")
    expected_evidence: ExpectedEvidence = Field(default_factory=ExpectedEvidence)
    completion_rule: CompletionRule = Field(default_factory=CompletionRule)
    timeout_seconds: Optional[float] = Field(
        default=None,
        ge=0.1,
        description="Optional timeout in seconds after which a slow step warning triggers"
    )
    optional: bool = Field(default=False)
    prerequisites: List[str] = Field(default_factory=list)


class Procedure(BaseModel):
    """Complete, validated experiment procedure."""
    id: str = Field(..., min_length=2)
    title: str = Field(..., min_length=3)
    version: str = Field(default="1.0.0")
    description: str = Field(default="")
    author: str = Field(default="Tabletop Procedure Monitoring Prototype")
    target_object: str = Field(default="notebook", description="Default user-selected target object")
    steps: List[StepDefinition] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_step_sequence(self) -> Procedure:
        """Validates uniqueness of step IDs, consecutive ordering, and prerequisite references."""
        step_ids = set()
        orders = []
        for step in self.steps:
            if step.id in step_ids:
                raise ValueError(f"Duplicate step id detected: '{step.id}'")
            step_ids.add(step.id)
            orders.append(step.order)

        # Check for duplicated orders
        if len(orders) != len(set(orders)):
            raise ValueError(f"Duplicate step orders detected in procedure '{self.id}'")

        # Sort steps by order
        self.steps.sort(key=lambda s: s.order)

        # Check prerequisites
        seen_ids = set()
        for step in self.steps:
            for prereq in step.prerequisites:
                if prereq not in seen_ids:
                    raise ValueError(
                        f"Step '{step.id}' specifies prerequisite '{prereq}', "
                        "which has not appeared earlier in sequence order."
                    )
            seen_ids.add(step.id)

        return self

    def get_step_by_id(self, step_id: str) -> Optional[StepDefinition]:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_step_by_index(self, index: int) -> Optional[StepDefinition]:
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    @property
    def total_steps(self) -> int:
        return len(self.steps)


def load_procedure(file_path: Union[str, Path]) -> Procedure:
    """Loads a procedure from a YAML or JSON file with rigorous validation."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Procedure file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if path.suffix.lower() in [".yaml", ".yml"]:
        raw = yaml.safe_load(content)
    elif path.suffix.lower() == ".json":
        raw = json.loads(content)
    else:
        # Try YAML parser as fallback for unspecified extension
        raw = yaml.safe_load(content)

    if not isinstance(raw, dict):
        raise ValueError(f"Procedure in '{path}' must be a dictionary/object definition.")

    return Procedure.model_validate(raw)


def save_procedure(procedure: Procedure, file_path: Union[str, Path]) -> None:
    """Saves procedure to YAML or JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = procedure.model_dump(mode="json")
    with open(path, "w", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            json.dump(raw, f, indent=2)
        else:
            yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)
