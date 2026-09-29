"""Unit tests for VLM schema parsing, generic object state validation, and client interfaces."""

import numpy as np
import pytest

from activity_detector.config.settings import VlmConfig
from activity_detector.vision.vlm import (
    MockVlmClient,
    OllamaVlmClient,
    VlmInterpretation,
    VlmResponseSchema,
)


def test_vlm_response_schema_validation():
    # Valid generic object observation
    data = {
        "object_visible": True,
        "object_description": "Blue spiral notebook with white lined pages",
        "open_or_closed": "open",
        "held_or_on_surface": "on_surface",
        "location": "workspace_center",
        "confidence": 0.88,
        "is_uncertain": False,
        "reasoning": "Notebook is resting open flat on the mat with pages clearly visible.",
    }
    schema = VlmResponseSchema.model_validate(data)
    assert schema.object_visible is True
    assert schema.open_or_closed == "open"
    assert schema.location == "workspace_center"
    assert schema.confidence == 0.88
    assert schema.is_uncertain is False
    assert "spiral notebook" in schema.object_description


def test_vlm_json_extractor_from_markdown():
    client = OllamaVlmClient(VlmConfig(enabled=False))

    # Raw response with markdown code fences
    raw_markdown = """
    Here is the physical analysis:
    ```json
    {
      "object_visible": true,
      "object_description": "Black cover notebook",
      "open_or_closed": "closed",
      "held_or_on_surface": "held",
      "location": "workspace_center",
      "confidence": 0.82,
      "is_uncertain": false,
      "reasoning": "Operator is holding closed notebook with both hands."
    }
    ```
    """
    schema, ok = client._parse_json_response(raw_markdown)
    assert ok is True
    assert schema.object_visible is True
    assert schema.open_or_closed == "closed"
    assert schema.confidence == 0.82

    # Ambiguous or invalid JSON response
    bad_raw = "I am unsure whether the notebook is open or closed because the lighting is too dark."
    schema_bad, ok_bad = client._parse_json_response(bad_raw)
    assert ok_bad is False
    assert schema_bad.is_uncertain is True
    assert schema_bad.confidence == 0.0


def test_mock_vlm_client():
    mock = MockVlmClient(
        mock_object_visible=True,
        mock_open_closed="open",
        mock_location="workspace_center",
        mock_confidence=0.92,
        mock_uncertain=False,
    )
    health = mock.check_health()
    assert health.available is True

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    submitted = mock.submit_sample(
        frame=frame,
        target_object="notebook",
        step_name="Open Notebook",
        step_instruction="Open the notebook flat",
        expected_state={"object_visible": True, "open_or_closed": "open"}
    )
    assert submitted is True

    latest = mock.get_latest_interpretation()
    assert latest is not None
    assert latest.schema_data.object_visible is True
    assert latest.schema_data.open_or_closed == "open"
    assert latest.schema_data.confidence == 0.92
    assert latest.target_object == "notebook"
