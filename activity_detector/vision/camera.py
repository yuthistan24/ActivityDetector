"""Camera and video stream capture manager with single-worker ownership and robust reconnect."""

from __future__ import annotations

from enum import Enum
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from activity_detector.config.settings import CameraConfig

logger = logging.getLogger("activity_detector.camera")


class CameraState(str, Enum):
    """Discrete health states of the camera hardware/stream connection."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    UNAVAILABLE = "unavailable"


class CameraManager:
    """Threaded video capture manager with strict single-worker ownership and error recovery.

    Architecture guarantees:
    1. Exactly one worker thread ('CameraCaptureWorker') creates, reads from, and releases
       the underlying VideoCapture object.
    2. Other threads and consumers (UI, recorder, VLM) only read the bounded latest-frame buffer.
    3. Transient read glitches (dropped frames) are absorbed without destroying the latest frame.
    4. Persistent failures trigger a clean release, backoff wait, and reconnection attempt.
    5. Shutdown interrupts any backoff sleep immediately without hanging.
    """

    def __init__(
        self,
        config: CameraConfig,
        max_consecutive_failures: int = 5,
        max_reconnect_attempts: int = 20,
        stale_threshold_seconds: float = 1.0,
        capture_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self.source: Union[int, str] = config.source
        self.max_consecutive_failures = max_consecutive_failures
        self.max_reconnect_attempts = max_reconnect_attempts
        self.stale_threshold_seconds = stale_threshold_seconds
        self._capture_factory = capture_factory

        # Capture handle owned SOLELY by _capture_worker thread
        self._cap: Optional[cv2.VideoCapture] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Buffer & Performance metrics
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._last_fps_calc_time: float = 0.0
        self._frames_since_calc: int = 0

        # Health & Diagnostic telemetry
        self.state: CameraState = CameraState.DISCONNECTED
        self.backend_name: str = "Uninitialized"
        self.consecutive_read_failures: int = 0
        self.reconnect_attempts: int = 0
        self.last_successful_read_time: float = 0.0
        self.last_error: str = ""

    @property
    def is_connected(self) -> bool:
        """Convenience property for downstream consumers."""
        return self.state == CameraState.CONNECTED

    @property
    def last_frame_age(self) -> float:
        """Returns elapsed seconds since the last successful frame read."""
        if self.last_successful_read_time <= 0.0:
            return 999.0
        return round(time.time() - self.last_successful_read_time, 3)

    @property
    def is_stale(self) -> bool:
        """True if camera is running but no fresh frame has been read within threshold."""
        return self._running and (self.last_frame_age > self.stale_threshold_seconds)

    def start(self) -> bool:
        """Spawns the sole capture worker thread."""
        if self._running:
            return True

        self._running = True
        self._stop_event.clear()
        self.state = CameraState.CONNECTING
        self.consecutive_read_failures = 0
        self.reconnect_attempts = 0
        self.last_error = ""

        # Pre-seed diagnostic fallback in case callers query before first frame
        with self._lock:
            self._latest_frame = self._create_diagnostic_frame(
                "CONNECTING TO CAMERA...",
                f"Source: {self.source}"
            )

        self._worker_thread = threading.Thread(
            target=self._capture_worker,
            name="CameraCaptureWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info(f"CameraManager worker started for source '{self.source}'")
        return True

    def stop(self) -> None:
        """Signals worker to stop, interrupts any backoff sleep, and joins thread."""
        logger.info("Stopping CameraManager...")
        self._running = False
        self._stop_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            # Wait up to 2.5s for worker thread to exit cleanly
            self._worker_thread.join(timeout=2.5)
            if self._worker_thread.is_alive():
                logger.warning("CameraCaptureWorker thread join timed out.")
        self._worker_thread = None

        # Failsafe cleanup if worker was terminated abruptly
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as e:
                    logger.error(f"Error releasing VideoCapture in stop: {e}")
                self._cap = None

        self.state = CameraState.DISCONNECTED
        logger.info("CameraManager stopped and resources released.")

    def _open_capture_under_worker(self) -> bool:
        """Opens capture device inside the worker thread with backend selection."""
        # Ensure previous handle is closed before creating a new one
        self._close_capture_under_worker()

        if self._capture_factory is not None:
            try:
                cap = self._capture_factory()
                if cap and cap.isOpened():
                    self._cap = cap
                    self.backend_name = "CustomFactory"
                    logger.info("Opened capture via injected factory.")
                    return True
                else:
                    self.last_error = "Injected capture factory returned un-opened capture."
                    self.backend_name = "CustomFactory (Failed)"
                    return False
            except Exception as e:
                self.last_error = f"Capture factory exception: {e}"
                self.backend_name = "CustomFactory (Error)"
                return False

        if isinstance(self.source, int):
            # On Windows, try DirectShow first to eliminate MSMF errors
            logger.info(f"Opening camera index {self.source} using cv2.CAP_DSHOW...")
            cap = None
            try:
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if cap and cap.isOpened():
                    self._cap = cap
                    self.backend_name = "DirectShow (CAP_DSHOW)"
                    self._apply_resolution_settings()
                    logger.info(f"Successfully opened camera {self.source} via DirectShow.")
                    return True
            except Exception as e:
                logger.debug(f"DirectShow open failed with exception: {e}")

            # Clean up failed handle before trying fallback
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass

            # Fallback to default backend (MSMF on Windows, V4L2 on Linux)
            logger.info(f"Trying default OpenCV backend for camera {self.source}...")
            try:
                cap = cv2.VideoCapture(self.source)
                if cap and cap.isOpened():
                    self._cap = cap
                    self.backend_name = "Default (MSMF/V4L2)"
                    self._apply_resolution_settings()
                    logger.info(f"Successfully opened camera {self.source} via default backend.")
                    return True
            except Exception as e:
                logger.error(f"Default backend open exception: {e}")

            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            self._cap = None
            self.backend_name = "Unavailable"
            self.last_error = f"Failed to open camera index {self.source} with any backend."
            return False

        # Source is a video file path or URL string
        logger.info(f"Opening video file: '{self.source}'...")
        try:
            cap = cv2.VideoCapture(str(self.source))
            if cap and cap.isOpened():
                self._cap = cap
                self.backend_name = "VideoFile"
                return True
        except Exception as e:
            self.last_error = f"Video file open error: {e}"

        if cap:
            try:
                cap.release()
            except Exception:
                pass
        self._cap = None
        self.backend_name = "Unavailable"
        self.last_error = f"Could not open video file: '{self.source}'"
        return False

    def _apply_resolution_settings(self) -> None:
        """Sets frame resolution if supported by the capture source."""
        if self._cap and self._cap.isOpened():
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            except Exception as e:
                logger.debug(f"Unable to set resolution on capture: {e}")

    def _close_capture_under_worker(self) -> None:
        """Releases the capture device. Must only be called from worker thread."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.debug(f"Exception during capture release: {e}")
            self._cap = None

    def _capture_worker(self) -> None:
        """Sole owner of VideoCapture: reads frames, absorbs transient errors, handles reconnect."""
        logger.info("CameraCaptureWorker thread started.")

        # Initial connection attempt
        if not self._open_capture_under_worker():
            self.state = CameraState.UNAVAILABLE
            self._set_diagnostic_frame(
                "CAMERA NOT AVAILABLE",
                f"Source: {self.source} | Backend: {self.backend_name}"
            )
        else:
            self.state = CameraState.CONNECTED
            self.last_successful_read_time = time.time()
            self._last_fps_calc_time = self.last_successful_read_time

        while self._running:
            # 1. If currently disconnected / reconnecting, attempt reconnect with backoff
            if self._cap is None or not self._cap.isOpened():
                if not self._handle_reconnect_cycle():
                    continue

            # 2. Perform frame read
            try:
                ret, frame = self._cap.read()
            except Exception as read_ex:
                ret = False
                frame = None
                self.last_error = f"Exception during VideoCapture.read(): {read_ex}"
                logger.warning(self.last_error)

            now = time.time()

            # 3. Handle read success
            if ret and frame is not None:
                self.consecutive_read_failures = 0
                self.state = CameraState.CONNECTED
                self.last_successful_read_time = now

                # Bounded buffer: store newest frame under lock
                with self._lock:
                    self._latest_frame = frame
                    self._frame_count += 1
                    self._frames_since_calc += 1

                # Update running FPS calculation
                dt = now - self._last_fps_calc_time
                if dt >= 1.0:
                    self._fps = round(self._frames_since_calc / dt, 1)
                    self._frames_since_calc = 0
                    self._last_fps_calc_time = now

                # Small sleep to yield to OS and other threads
                time.sleep(0.004)
                continue

            # 4. Handle Video File Loop (if reading a test clip)
            if isinstance(self.source, str) and self._cap is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.03)
                continue

            # 5. Handle transient vs persistent read failure
            self.consecutive_read_failures += 1
            age = self.last_frame_age
            logger.warning(
                f"Camera read failure #{self.consecutive_read_failures} on source {self.source} "
                f"({self.backend_name}). Frame age: {age:.2f}s"
            )

            # If within transient tolerance, retry without tearing down connection
            if self.consecutive_read_failures < self.max_consecutive_failures and age < 1.5:
                # Sleep briefly using interruptible stop_event
                self._stop_event.wait(0.02)
                continue

            # 6. Persistent failure reached: Initiate reconnect
            self.state = CameraState.RECONNECTING
            self.reconnect_attempts += 1
            self.last_error = (
                f"Lost connection after {self.consecutive_read_failures} failed reads. "
                f"Last frame age: {age:.2f}s."
            )
            logger.error(f"{self.last_error} Initiating reconnect #{self.reconnect_attempts}...")

            self._set_diagnostic_frame(
                "CAMERA SIGNAL LOST",
                f"Reconnecting attempt #{self.reconnect_attempts} ({self.backend_name})..."
            )

            # Release the broken capture handle before waiting
            self._close_capture_under_worker()

            # Exponential backoff with ceiling (0.4s to 2.5s)
            backoff = min(2.5, 0.4 * (1.4 ** min(self.reconnect_attempts - 1, 5)))
            if self._stop_event.wait(backoff):
                break  # Stop requested during backoff wait

        # Clean up capture when worker loop terminates
        self._close_capture_under_worker()
        self.state = CameraState.DISCONNECTED
        logger.info("CameraCaptureWorker thread terminated.")

    def _handle_reconnect_cycle(self) -> bool:
        """Executes a reconnect attempt. Returns True if successfully connected."""
        if not self._running:
            return False

        self.state = CameraState.RECONNECTING
        self.reconnect_attempts += 1
        logger.info(f"Executing reconnect attempt #{self.reconnect_attempts} on source '{self.source}'...")

        self._set_diagnostic_frame(
            "RECONNECTING CAMERA",
            f"Attempt #{self.reconnect_attempts} on source {self.source}..."
        )

        success = self._open_capture_under_worker()
        if success:
            self.state = CameraState.CONNECTED
            self.consecutive_read_failures = 0
            self.reconnect_attempts = 0
            self.last_successful_read_time = time.time()
            logger.info(f"Camera reconnected successfully! ({self.backend_name})")
            return True

        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.state = CameraState.UNAVAILABLE
            self._set_diagnostic_frame(
                "CAMERA UNAVAILABLE",
                f"Source {self.source} failed after {self.reconnect_attempts} attempts."
            )

        # Wait before next attempt
        backoff = min(3.0, 0.5 * (1.5 ** min(self.reconnect_attempts - 1, 4)))
        self._stop_event.wait(backoff)
        return False

    def _set_diagnostic_frame(self, title: str, subtitle: str) -> None:
        """Sets a synthetic diagnostic card in the frame buffer."""
        card = self._create_diagnostic_frame(title, subtitle)
        with self._lock:
            self._latest_frame = card

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Thread-safe retrieval of the freshest frame copy without locking capture.

        Returns:
            (has_frame, frame_copy, current_fps)
        """
        with self._lock:
            if self._latest_frame is None:
                fallback = self._create_diagnostic_frame("INITIALIZING CAMERA FEED", f"Source: {self.source}")
                return False, fallback, 0.0
            return self.is_connected, self._latest_frame.copy(), self._fps

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic telemetry for UI and audit logs."""
        return {
            "state": self.state.value,
            "backend": self.backend_name,
            "source": self.source,
            "fps": self._fps,
            "frame_count": self._frame_count,
            "consecutive_failures": self.consecutive_read_failures,
            "reconnect_attempts": self.reconnect_attempts,
            "last_frame_age_seconds": self.last_frame_age,
            "is_stale": self.is_stale,
            "last_error": self.last_error,
        }

    def _create_diagnostic_frame(self, title: str, subtitle: str) -> np.ndarray:
        """Generates a high-contrast synthetic test card frame for UI diagnostics."""
        w, h = self.config.width, self.config.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Background grid pattern
        for x in range(0, w, 80):
            cv2.line(frame, (x, 0), (x, h), (25, 25, 30), 1)
        for y in range(0, h, 80):
            cv2.line(frame, (0, y), (w, y), (25, 25, 30), 1)

        # Alert Box
        box_w, box_h = min(w - 40, 600), 190
        x1, y1 = (w - box_w) // 2, (h - box_h) // 2

        # Card color based on state
        border_color = (60, 60, 220) if "LOST" in title or "UNAVAILABLE" in title else (220, 160, 40)
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (35, 38, 48), -1)
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), border_color, 2)

        # Text
        cv2.putText(frame, title, (x1 + 25, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (x1 + 25, y1 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 190, 210), 1, cv2.LINE_AA)

        # Telemetry line
        status_line = f"State: {self.state.value.upper()} | Backend: {self.backend_name}"
        cv2.putText(frame, status_line, (x1 + 25, y1 + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 150, 170), 1, cv2.LINE_AA)

        # Timestamp
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Local Time: {ts}", (x1 + 25, y1 + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 110, 130), 1, cv2.LINE_AA)

        return frame
