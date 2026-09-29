"""Hardware-independent unit tests for CameraManager capture ownership and reconnect logic."""

import threading
import time
from typing import List, Optional
import numpy as np
import pytest

from activity_detector.config.settings import CameraConfig
from activity_detector.vision.camera import CameraManager, CameraState


class FakeVideoCapture:
    """Mock OpenCV VideoCapture for hardware-independent deterministic testing."""

    def __init__(self, sequence: Optional[List[Optional[np.ndarray]]] = None) -> None:
        self.sequence = list(sequence) if sequence is not None else []
        self._is_opened = True
        self.read_count = 0
        self.release_count = 0
        self.calling_threads = set()
        self._lock = threading.Lock()

    def isOpened(self) -> bool:
        with self._lock:
            return self._is_opened

    def read(self):
        with self._lock:
            self.read_count += 1
            self.calling_threads.add(threading.current_thread().name)
            if not self._is_opened:
                return False, None
            if not self.sequence:
                # Default frame: 100x100 solid image
                f = np.ones((100, 100, 3), dtype=np.uint8) * 128
                return True, f
            item = self.sequence.pop(0)
            if item is None:
                return False, None
            return True, item

    def release(self) -> None:
        with self._lock:
            self.release_count += 1
            self._is_opened = False

    def set(self, prop: int, val: float) -> bool:
        return True


def test_single_worker_owns_capture():
    """Verify that only the dedicated worker thread calls read() and release()."""
    fake_cap = FakeVideoCapture()
    config = CameraConfig(source=0, width=640, height=480)
    cam = CameraManager(config, capture_factory=lambda: fake_cap)

    assert cam.start() is True
    time.sleep(0.08)
    cam.stop()

    # All reads must originate solely from CameraCaptureWorker
    assert len(fake_cap.calling_threads) == 1
    assert "CameraCaptureWorker" in fake_cap.calling_threads
    assert fake_cap.read_count > 0
    assert fake_cap.release_count >= 1
    assert fake_cap.isOpened() is False


def test_transient_read_failure_absorption():
    """Verify transient read failures are absorbed without losing connection or discarding last frame."""
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_a[0, 0] = [255, 0, 0]

    frame_b = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b[0, 0] = [0, 255, 0]

    # Sequence: 2 good frames, 2 transient None failures, then good frames indefinitely
    seq = [frame_a, frame_a, None, None, frame_b, frame_b]
    fake_cap = FakeVideoCapture(sequence=seq)

    config = CameraConfig(source=0, width=640, height=480)
    # Require 5 consecutive failures to trigger reconnect, so 2 failures must be absorbed
    cam = CameraManager(config, max_consecutive_failures=5, capture_factory=lambda: fake_cap)

    assert cam.start() is True
    time.sleep(0.12)

    has_frame, current_frame, fps = cam.get_frame()
    cam.stop()

    assert cam.reconnect_attempts == 0
    assert has_frame is True
    assert current_frame is not None
    # Frame should NOT be the test card, but actual image
    assert current_frame.shape == (100, 100, 3)


def test_persistent_failure_triggers_reconnect_and_clean_release():
    """Verify persistent failures release the broken handle before reopening a new one."""
    opened_caps = []

    def make_cap():
        if len(opened_caps) == 0:
            # First capture: fails immediately
            c = FakeVideoCapture(sequence=[None, None, None, None, None, None])
        else:
            # Second capture (reconnected): succeeds
            c = FakeVideoCapture()
        opened_caps.append(c)
        return c

    config = CameraConfig(source=0, width=640, height=480)
    cam = CameraManager(
        config,
        max_consecutive_failures=3,
        stale_threshold_seconds=0.05,
        capture_factory=make_cap,
    )

    assert cam.start() is True
    # Allow enough time for 3 failures, release of cap 0, backoff, and reconnect of cap 1
    time.sleep(0.65)
    cam.stop()

    # Must have created at least 2 capture objects
    assert len(opened_caps) >= 2
    # The first broken capture MUST have been released before/during reconnect
    assert opened_caps[0].release_count >= 1
    assert opened_caps[0].isOpened() is False


def test_clean_shutdown_interrupts_backoff():
    """Verify calling stop() while in reconnect backoff wait returns immediately."""
    # Always failing capture to force backoff loop
    fake_cap = FakeVideoCapture(sequence=[None] * 50)
    config = CameraConfig(source=0)
    cam = CameraManager(config, max_consecutive_failures=2, capture_factory=lambda: fake_cap)

    cam.start()
    time.sleep(0.1)

    t0 = time.time()
    cam.stop()
    elapsed = time.time() - t0

    # Stop must cleanly interrupt within 0.8s, not hang for full backoff duration
    assert elapsed < 0.8
    assert cam.state == CameraState.DISCONNECTED
    assert cam._worker_thread is None


def test_camera_diagnostics_reporting():
    """Verify get_diagnostics() accurately reflects state, age, and staleness."""
    fake_cap = FakeVideoCapture()
    config = CameraConfig(source=0, width=640, height=480)
    cam = CameraManager(config, stale_threshold_seconds=0.1, capture_factory=lambda: fake_cap)

    cam.start()
    time.sleep(0.08)

    diag = cam.get_diagnostics()
    assert diag["state"] == "connected"
    assert diag["backend"] == "CustomFactory"
    assert diag["frame_count"] > 0
    assert diag["last_frame_age_seconds"] < 0.5
    assert diag["is_stale"] is False

    cam.stop()
    diag_stopped = cam.get_diagnostics()
    assert diag_stopped["state"] == "disconnected"
