"""Unit tests for Procedure domain schema and sequence validation."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from activity_detector.core.procedure import (
    CompletionRule,
    ExpectedEvidence,
    Procedure,
    StepDefinition,
    load_procedure,
    save_procedure,
)


def test_valid_procedure_creation():
    proc = Procedure(
        id="test_exp",
        title="Test Experiment",
        version="1.0.0",
        steps=[
            StepDefinition(
                id="step_1",
                order=1,
                name="Step One",
                instruction="Perform first step",
                expected_evidence=ExpectedEvidence(required_colors=["blue_reagent"]),
                completion_rule=CompletionRule(stable_frames=5),
            ),
            StepDefinition(
                id="step_2",
                order=2,
                name="Step Two",
                instruction="Perform second step",
                expected_evidence=ExpectedEvidence(required_colors=["yellow_flask"]),
                completion_rule=CompletionRule(stable_frames=5),
                prerequisites=["step_1"],
            ),
        ]
    )
    assert proc.total_steps == 2
    assert proc.get_step_by_id("step_1") is not None
    assert proc.get_step_by_index(1).id == "step_2"


def test_duplicate_step_id_rejected():
    with pytest.raises(ValidationError):
        Procedure(
            id="bad_exp",
            title="Bad Exp",
            steps=[
                StepDefinition(
                    id="step_dup",
                    order=1,
                    name="Step 1",
                    instruction="Instruction 1",
                ),
                StepDefinition(
                    id="step_dup",
                    order=2,
                    name="Step 2",
                    instruction="Instruction 2",
                ),
            ]
        )


def test_invalid_prerequisite_rejected():
    # step_1 refers to step_2 which has not appeared earlier
    with pytest.raises(ValidationError):
        Procedure(
            id="bad_prereq",
            title="Bad Prereq",
            steps=[
                StepDefinition(
                    id="step_1",
                    order=1,
                    name="Step 1",
                    instruction="Instruction 1",
                    prerequisites=["step_2"],
                ),
                StepDefinition(
                    id="step_2",
                    order=2,
                    name="Step 2",
                    instruction="Instruction 2",
                ),
            ]
        )


def test_load_sample_procedures():
    notebook_path = Path("procedures/default_notebook_handling.yaml")
    assert notebook_path.exists()
    notebook_proc = load_procedure(notebook_path)
    assert notebook_proc.id == "notebook_handling_protocol_v1"
    assert notebook_proc.target_object == "notebook"
    assert len(notebook_proc.steps) == 4
    assert notebook_proc.steps[0].expected_evidence.target_object == "notebook"

    titration_path = Path("procedures/sample_titration_experiment.yaml")
    assert titration_path.exists()
    proc = load_procedure(titration_path)
    assert proc.id == "titration_exp_demo_v1"
    assert len(proc.steps) == 5

    circuit_path = Path("procedures/sample_circuit_assembly.yaml")
    assert circuit_path.exists()
    circuit_proc = load_procedure(circuit_path)
    assert circuit_proc.id == "circuit_assembly_demo_v1"
    assert len(circuit_proc.steps) == 5


def test_save_and_load_procedure_json(tmp_path: Path):
    proc = load_procedure("procedures/sample_titration_experiment.yaml")
    json_path = tmp_path / "proc.json"
    save_procedure(proc, json_path)
    assert json_path.exists()

    loaded = load_procedure(json_path)
    assert loaded.id == proc.id
    assert len(loaded.steps) == len(proc.steps)
