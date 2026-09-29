"""Unit tests for VLM schema parsing and client interfaces."""

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
    data = {
        "step_recognized": True,
        "action_description": "Operator positioned yellow container",
        "detected_items": ["yellow_flask", "person"],
        "confidence": 0.88,
        "is_uncertain": False,
        "reasoning": "Container is steady on workbench."
    }
    schema = VlmResponseSchema.model_validate(data)
    assert schema.step_recognized is True
    assert schema.confidence == 0.88
    assert "yellow_flask" in schema.detected_items


def test_vlm_json_extractor_from_markdown():
    client = OllamaVlmClient(VlmConfig(enabled=False))

    # Raw response with markdown code fences
    raw_markdown = """
    Here is the analysis:
    ```json
    {
      "step_recognized": true,
      "action_description": "Adding reagent",
      "detected_items": ["blue_reagent"],
      "confidence": 0.75,
      "is_uncertain": false,
      "reasoning": "Reagent bottle in hand"
    }
    ```
    """
    schema, ok = client._parse_json_response(raw_markdown)
    assert ok is True
    assert schema.step_recognized is True
    assert schema.confidence == 0.75

    # Invalid JSON string
    bad_raw = "I am an AI and I cannot verify this."
    schema_bad, ok_bad = client._parse_json_response(bad_raw)
    assert ok_bad is False
    assert schema_bad.is_uncertain is True
    assert schema_bad.confidence == 0.0


def test_mock_vlm_client():
    mock = MockVlmClient(mock_step_recognized=True, mock_confidence=0.92)
    health = mock.check_health()
    assert health.available is True

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    submitted = mock.submit_sample(
        frame=frame,
        step_name="Safety Check",
        step_instruction="Put on gloves",
        expected_items=["gloves"]
    )
    assert submitted is True

    latest = mock.get_latest_interpretation()
    assert latest is not None
    assert latest.schema_data.step_recognized is True
    assert latest.schema_data.confidence == 0.92
