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


def test_vlm_pause_and_resume():
    client = OllamaVlmClient(VlmConfig(enabled=True))
    assert client.is_paused is False

    client.pause()
    assert client.is_paused is True

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # When paused, submit_sample must immediately return False
    submitted = client.submit_sample(
        frame=frame,
        target_object="notebook",
        step_name="Step 1",
        step_instruction="Test",
    )
    assert submitted is False

    client.resume()
    assert client.is_paused is False
    client.stop()


def test_pipeline_samples_vlm_in_idle_preview():
    """Verify that when engine is in IDLE state, the vision pipeline samples VLM in preview mode."""
    from activity_detector.config.settings import get_default_config
    from activity_detector.core.engine import EngineState, EngineUpdate
    from activity_detector.core.procedure import StepDefinition, ExpectedEvidence
    from activity_detector.vision.pipeline import VisionPipeline

    mock = MockVlmClient()
    config = get_default_config()
    pipeline = VisionPipeline(config, vlm_client=mock)

    # Frame tick with IDLE state
    idle_update = EngineUpdate(
        state=EngineState.IDLE,
        current_step_index=0,
        current_step=StepDefinition(
            id="step_1",
            order=1,
            name="Locate Notebook",
            instruction="Find notebook",
            expected_evidence=ExpectedEvidence(target_object="notebook", expected_state={}),
        ),
        next_step=None,
        progress_percentage=0.0,
        step_records=[],
        recent_warning=None,
        stability_ratio=0.0,
        evidence_summary="",
        evidence_source="none",
        transition_occurred=False,
    )

    annotated, evidence = pipeline.process_next_frame(idle_update)
    # VLM MUST receive preview sample so operator sees object recognition immediately
    assert mock.submit_count == 1

    pipeline.stop()


def test_vlm_target_object_switching():
    """Verify that changing target object clears previous interpretation and updates prompts."""
    from activity_detector.config.settings import VlmConfig
    from activity_detector.vision.vlm import OllamaVlmClient, VlmInterpretation, VlmResponseSchema

    client = OllamaVlmClient(VlmConfig(enabled=False))
    assert client.target_object == "notebook"

    # Set initial interpretation
    client._latest_interpretation = VlmInterpretation(
        schema_data=VlmResponseSchema(
            object_visible=True,
            object_description="notebook on desk",
            open_or_closed="open",
            held_or_on_surface="on_surface",
            location="workspace_center",
            confidence=0.9,
            is_uncertain=False,
            reasoning="visible",
        ),
        raw_response="",
        latency_seconds=0.5,
        target_object="notebook",
    )
    assert client.get_latest_interpretation() is not None

    # Change target object
    client.set_target_object("wrench")
    assert client.target_object == "wrench"
    # Previous interpretation must be cleared so stale results don't show
    assert client.get_latest_interpretation() is None

    # Check prompt generation reflects new target
    prompt = client._build_prompt("wrench", "Inspect Tool", "Verify wrench is ready", {})
    assert "wrench" in prompt
    assert "notebook" not in prompt

    client.stop()


def test_vlm_single_in_flight_and_sample_interval():
    """Verify that at most one inference runs at a time and sample interval limits submission."""
    import time
    from activity_detector.config.settings import VlmConfig
    from activity_detector.vision.vlm import OllamaVlmClient

    config = VlmConfig(enabled=True, sample_interval_seconds=2.0)
    client = OllamaVlmClient(config)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # First sample submission succeeds
    sub1 = client.submit_sample(frame, "notebook", "Step 1", "Inspect")
    assert sub1 is True

    # Immediate second submission should be dropped due to interval
    sub2 = client.submit_sample(frame, "notebook", "Step 1", "Inspect")
    assert sub2 is False

    # Simulate in-flight analysis flag
    client._last_sample_time = 0.0  # bypass interval
    client._is_analyzing = True
    sub3 = client.submit_sample(frame, "notebook", "Step 1", "Inspect")
    assert sub3 is False  # dropped because already analyzing

    # Test dynamic interval change
    client.set_sample_interval(5.0)
    assert client.config.sample_interval_seconds == 5.0

    client.stop()




def test_vlm_cooldown_and_degraded_state(monkeypatch):
    """Verify that failed inference calls trigger backoff cooldown and degrade after threshold."""
    import time
    from unittest.mock import MagicMock
    from activity_detector.config.settings import VlmConfig
    from activity_detector.vision.vlm import OllamaVlmClient

    config = VlmConfig(
        enabled=True,
        sample_interval_seconds=0.1,
        failure_cooldown_seconds=0.1,
        max_consecutive_failures=3,
    )
    client = OllamaVlmClient(config)

    # Monkeypatch the internal _query_ollama chat call by raising inside client
    mock_ollama_module = MagicMock()
    mock_instance = MagicMock()
    mock_instance.chat.side_effect = TimeoutError("timed out")
    mock_ollama_module.Client.return_value = mock_instance
    monkeypatch.setattr("activity_detector.vision.vlm.ollama", mock_ollama_module, raising=False)
    # Also patch sys.modules in case import ollama is called inside the function
    import sys
    monkeypatch.setitem(sys.modules, "ollama", mock_ollama_module)

    # Test failure #1
    interp1 = client._query_ollama(
        b64_img="abc", width=100, height=100, bytes_len=100,
        target_object="notebook", step_name="Step 1", step_instruction="Test",
        expected_state={}
    )
    assert interp1.is_valid is False
    assert interp1.schema_data.is_uncertain is True
    assert client.consecutive_failures == 1
    assert client.is_degraded is False
    assert client._cooldown_until > time.time()

    # While cooldown is active, submit_sample must drop frame
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    assert client.submit_sample(frame, "notebook", "Step 1", "Test") is False

    # Test failure #2 and #3
    client._query_ollama(
        b64_img="abc", width=100, height=100, bytes_len=100,
        target_object="notebook", step_name="Step 1", step_instruction="Test",
        expected_state={}
    )
    assert client.consecutive_failures == 2
    assert client.is_degraded is False

    client._query_ollama(
        b64_img="abc", width=100, height=100, bytes_len=100,
        target_object="notebook", step_name="Step 1", step_instruction="Test",
        expected_state={}
    )
    assert client.consecutive_failures == 3
    assert client.is_degraded is True
    assert "Timeout" in client.degraded_reason

    # Now simulate a recovery on query #4
    mock_instance.chat.side_effect = None
    mock_instance.chat.return_value = {
        "message": {
            "content": '{"object_visible": true, "object_description": "notebook", "open_or_closed": "open", "held_or_on_surface": "on_surface", "location": "workspace_center", "confidence": 0.9, "is_uncertain": false, "reasoning": "visible"}'
        }
    }
    interp_rec = client._query_ollama(
        b64_img="abc", width=100, height=100, bytes_len=100,
        target_object="notebook", step_name="Step 1", step_instruction="Test",
        expected_state={}
    )
    assert interp_rec.is_valid is True
    assert client.consecutive_failures == 0
    assert client.is_degraded is False

    client.stop()


def test_vlm_frame_resizing_and_compression():
    """Verify that oversized frames are scaled to max_dimension and JPEG compressed."""
    from activity_detector.config.settings import VlmConfig
    from activity_detector.vision.vlm import OllamaVlmClient

    config = VlmConfig(
        enabled=True,
        image_max_dimension=480,
        jpeg_quality=75,
        sample_interval_seconds=0.1,
    )
    client = OllamaVlmClient(config)

    # Frame 1280x720
    large_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    submitted = client.submit_sample(
        frame=large_frame,
        target_object="notebook",
        step_name="Step 1",
        step_instruction="Test instruction",
    )
    assert submitted is True

    # Check queued payload
    queued = client._queue.get_nowait()
    assert queued["width"] == 480
    assert queued["height"] == 270  # 720 * (480 / 1280) = 270
    assert queued["bytes_len"] < 100000  # JPEG compressed

    client.stop()


def test_pipeline_marks_uncertain_when_vlm_degraded():
    """Verify that when VLM is in degraded state, FrameEvidence flags vlm_uncertain = True."""
    from activity_detector.config.settings import get_default_config
    from activity_detector.core.engine import EngineState, EngineUpdate
    from activity_detector.core.procedure import StepDefinition, ExpectedEvidence
    from activity_detector.vision.pipeline import VisionPipeline

    mock = MockVlmClient()
    # Force mock into degraded state
    mock._is_degraded = True
    mock._degraded_reason = "OOM: Out of memory"

    config = get_default_config()
    pipeline = VisionPipeline(config, vlm_client=mock)

    active_update = EngineUpdate(
        state=EngineState.IN_PROGRESS,
        current_step_index=0,
        current_step=StepDefinition(
            id="step_1",
            order=1,
            name="Locate Notebook",
            instruction="Find notebook",
            expected_evidence=ExpectedEvidence(target_object="notebook", expected_state={}),
        ),
        next_step=None,
        progress_percentage=0.0,
        step_records=[],
        recent_warning=None,
        stability_ratio=0.0,
        evidence_summary="",
        evidence_source="none",
        transition_occurred=False,
    )

    _, evidence = pipeline.process_next_frame(active_update)
    assert evidence.vlm_uncertain is True
    assert "degraded" in evidence.vlm_summary.lower()

    pipeline.stop()


