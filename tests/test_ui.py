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
