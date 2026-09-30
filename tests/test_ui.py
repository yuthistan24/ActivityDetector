"""Automated headless tests for the PyQt6 user interface."""

import os
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication

from activity_detector.config.settings import get_default_config
from activity_detector.core.procedure import load_procedure
from activity_detector.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_main_window_lifecycle(qapp, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: None)

    config = get_default_config()
    config.vlm.enabled = False
    config.audio.enabled = False
    config.recording.auto_record_on_session_start = False

    procedure = load_procedure(config.procedure_file)
    window = MainWindow(config, procedure)

    # Verify widget instantiation
    assert window.video_widget is not None
    assert window.active_step_card is not None
    assert window.status_badge is not None
    assert window.step_list_widget is not None

    # Run frame processing ticks
    for _ in range(3):
        window._process_tick()

    # Start session
    window._toggle_session()
    assert window.session_manager.is_active
    assert window.engine.state.value == "in_progress"

    # Test manual step advance
    window._manual_next_step()
    assert window.engine.current_step_index == 1

    # Test manual step revert
    window._manual_prev_step()
    assert window.engine.current_step_index == 0

    # Stop session
    window._toggle_session()
    assert not window.session_manager.is_active

    # Test alert acknowledge
    window._acknowledge_alerts()

    window.close()


def test_main_window_object_recognition_updates(qapp, monkeypatch):
    """Verify that fake vision responses update the recognition card dynamically."""
    from activity_detector.vision.vlm import MockVlmClient
    from activity_detector.config.settings import get_default_config
    from activity_detector.core.procedure import load_procedure

    config = get_default_config()
    config.vlm.enabled = True
    config.audio.enabled = False
    config.recording.auto_record_on_session_start = False

    mock_vlm = MockVlmClient(
        mock_object_visible=True,
        mock_open_closed="open",
        mock_location="workspace_center",
        mock_confidence=0.94,
        mock_uncertain=False,
    )

    procedure = load_procedure(config.procedure_file)
    window = MainWindow(config, procedure, vlm_client=mock_vlm)

    # Process frame ticks
    window._process_tick()

    # Verify recognition card is populated
    assert window.recognition_card is not None
    badge_text = window.recognition_card.status_pill.text()
    assert "RECOGNIZED" in badge_text
    assert "open" in window.recognition_card.chips_label.text().lower()
    assert "94%" in window.recognition_card.chips_label.text()
    assert "notebook" in window.recognition_card.lbl_target_title.text().lower()

    window.close()


def test_main_window_dynamic_target_and_interval(qapp, monkeypatch):
    """Verify that changing target object and interval in GUI updates pipeline and UI instantly."""
    from activity_detector.vision.vlm import MockVlmClient
    from activity_detector.config.settings import get_default_config
    from activity_detector.core.procedure import load_procedure

    config = get_default_config()
    config.vlm.enabled = True
    config.audio.enabled = False
    config.recording.auto_record_on_session_start = False

    mock_vlm = MockVlmClient()
    procedure = load_procedure(config.procedure_file)
    window = MainWindow(config, procedure, vlm_client=mock_vlm)

    # Change target object via GUI input
    window.input_target_object.setText("beaker")
    window._on_target_object_changed()

    assert window.target_object == "beaker"
    assert window.pipeline.target_object == "beaker"
    assert window.vlm_client.target_object == "beaker"
    assert "beaker" in window.recognition_card.lbl_target_title.text().lower()
    # Check that procedure step also received updated target object
    assert window.engine.current_step.expected_evidence.target_object == "beaker"

    # Change sample interval via GUI spinbox
    window.spin_interval.setValue(4.5)
    assert window.pipeline.vlm_client.config.sample_interval_seconds == 4.5

    window.close()


def test_main_window_vlm_unavailable_state(qapp, monkeypatch):
    """Verify that model timeout or failure updates UI to unavailable without freezing GUI."""
    from activity_detector.vision.vlm import MockVlmClient
    from activity_detector.config.settings import get_default_config
    from activity_detector.core.procedure import load_procedure

    config = get_default_config()
    config.vlm.enabled = True
    config.audio.enabled = False

    mock_vlm = MockVlmClient()
    # Simulate degraded / timed out model
    mock_vlm._is_degraded = True
    mock_vlm._degraded_reason = "Ollama connection timed out"

    procedure = load_procedure(config.procedure_file)
    window = MainWindow(config, procedure, vlm_client=mock_vlm)

    # Tick GUI frame
    window._process_tick()

    # Recognition card must reflect unavailable state
    badge_text = window.recognition_card.status_pill.text()
    assert "UNAVAILABLE" in badge_text or "UNCERTAIN" in badge_text
    assert "timed out" in window.recognition_card.description_box.text().lower()

    # Video preview and window remain fully responsive (non-blocking)
    window._process_tick()
    window._process_tick()

    window.close()

