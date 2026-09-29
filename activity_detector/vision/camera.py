"""Camera and video stream capture manager with thread-safe frame buffering."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from activity_detector.config.settings import CameraConfig

logger = logging.getLogger("activity_detector.camera")


class CameraManager:
    """Threaded OpenCV video capture manager ensuring zero-lag frame access."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.source = config.source

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._fps: float = 0.0
        self._last_fps_calc_time: float = 0.0
        self._frames_since_calc: int = 0

        self.is_connected: bool = False
        self.last_error: str = ""

    def start(self) -> bool:
        """Opens capture source and starts background capture thread."""
        if self._running:
            return True

        if isinstance(self.source, int):
            # Prefer DirectShow on Windows to avoid MSMF -1072873822 errors
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap or not self._cap.isOpened():
            self.is_connected = False
            self.last_error = f"Could not open video source: '{self.source}'"
            logger.warning(self.last_error)
            # Create synthetic fallback frame so callers can still run
            self._latest_frame = self._create_diagnostic_frame(
                "CAMERA NOT CONNECTED",
                f"Source: {self.source}"
            )
            return False

        # Set requested resolution
        if isinstance(self.source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        self.is_connected = True
        self.last_error = ""
        self._running = True
        self._start_time = time.time()
        self._last_fps_calc_time = self._start_time
        self._frames_since_calc = 0

        self._thread = threading.Thread(target=self._capture_worker, name="CameraCaptureThread", daemon=True)
        self._thread.start()
        logger.info(f"Camera manager started on source '{self.source}'")
        return True

    def stop(self) -> None:
        """Stops background thread and releases video capture hardware."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._cap:
            try:
                self._cap.release()
            except Exception as e:
                logger.error(f"Error releasing VideoCapture: {e}")
            self._cap = None

        self.is_connected = False
        logger.info("Camera manager stopped and resources released.")

    def _capture_worker(self) -> None:
        """Continuously pulls frames from the capture device, retaining only the freshest frame."""
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # If reading from a video file, loop back to start
                if isinstance(self.source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03)
                    continue

                self.is_connected = False
                self.last_error = "Frame read returned False / Device disconnected"
                with self._lock:
                    self._latest_frame = self._create_diagnostic_frame(
                        "CAMERA SIGNAL LOST",
                        "Retrying connection..."
                    )
                time.sleep(0.2)
                continue

            self.is_connected = True
            with self._lock:
                self._latest_frame = frame
                self._frame_count += 1
                self._frames_since_calc += 1

            # Update FPS periodically
            now = time.time()
            dt = now - self._last_fps_calc_time
            if dt >= 1.0:
                self._fps = round(self._frames_since_calc / dt, 1)
                self._frames_since_calc = 0
                self._last_fps_calc_time = now

            # Sleep slightly if capturing from camera to avoid 100% spin
            time.sleep(0.005)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Returns (has_frame, frame_copy, current_fps)."""
        with self._lock:
            if self._latest_frame is None:
                fallback = self._create_diagnostic_frame("INITIALIZING VIDEO FEED", f"Source: {self.source}")
                return False, fallback, 0.0
            return True, self._latest_frame.copy(), self._fps

    def _create_diagnostic_frame(self, title: str, subtitle: str) -> np.ndarray:
        """Generates a synthetic diagnostic test card frame."""
        w, h = self.config.width, self.config.height
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw dark grid pattern
        for x in range(0, w, 80):
            cv2.line(frame, (x, 0), (x, h), (30, 30, 35), 1)
        for y in range(0, h, 80):
            cv2.line(frame, (0, y), (w, y), (30, 30, 35), 1)

        # Draw central alert box
        box_w, box_h = 560, 180
        x1, y1 = (w - box_w) // 2, (h - box_h) // 2
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (45, 45, 55), -1)
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (80, 80, 200), 2)

        # Text
        cv2.putText(frame, title, (x1 + 30, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (x1 + 30, y1 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 200), 1, cv2.LINE_AA)

        # Timestamp
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (x1 + 30, y1 + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 140), 1, cv2.LINE_AA)
        return frame
