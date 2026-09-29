"""Optional isolated HTTP MJPEG video streamer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
import threading
import time
from typing import Optional
import cv2
import numpy as np

from activity_detector.config.settings import StreamingConfig

logger = logging.getLogger("activity_detector.streamer")


class BaseStreamer(ABC):
    """Abstract interface for video streaming."""

    @abstractmethod
    def start(self) -> bool:
        """Starts streaming server."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops streaming server."""
        pass

    @abstractmethod
    def update_frame(self, frame: np.ndarray) -> None:
        """Updates the latest frame available to streaming clients."""
        pass

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether the streamer is actively serving."""
        pass


class MjpegHttpStreamer(BaseStreamer):
    """Lightweight HTTP MJPEG streaming server bound strictly to configured host/port."""

    def __init__(self, config: StreamingConfig) -> None:
        self.config = config
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._is_active: bool = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    def update_frame(self, frame: np.ndarray) -> None:
        """Encodes frame to JPEG bytes for connected HTTP clients."""
        if not self._is_active or frame is None:
            return

        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self._lock:
                self._latest_jpeg = buf.tobytes()
        except Exception as e:
            logger.debug(f"Frame encoding for stream failed: {e}")

    def start(self) -> bool:
        """Starts HTTP server in background daemon thread."""
        if self._is_active:
            return True

        streamer_ref = self

        class MjpegHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                # Suppress noisy HTTP request logging
                pass

            def do_GET(self) -> None:
                if self.path == streamer_ref.config.path or self.path == "/":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                    self.end_headers()

                    try:
                        while streamer_ref._is_active:
                            jpeg = None
                            with streamer_ref._lock:
                                jpeg = streamer_ref._latest_jpeg

                            if jpeg:
                                self.wfile.write(b"--FRAME\r\n")
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Content-Length", str(len(jpeg)))
                                self.end_headers()
                                self.wfile.write(jpeg)
                                self.wfile.write(b"\r\n")

                            time.sleep(0.04)  # ~25 FPS
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)
                    self.end_headers()

        try:
            self._server = ThreadingHTTPServer((self.config.host, self.config.port), MjpegHandler)
            self._is_active = True
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="MjpegStreamServerThread",
                daemon=True
            )
            self._thread.start()
            logger.info(f"MJPEG stream available at http://{self.config.host}:{self.config.port}{self.config.path}")
            return True
        except Exception as e:
            logger.error(f"Failed to start MJPEG stream server: {e}")
            self._is_active = False
            return False

    def stop(self) -> None:
        """Shuts down HTTP server."""
        self._is_active = False
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception as e:
                logger.error(f"Error shutting down streaming server: {e}")
            self._server = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        logger.info("MJPEG streaming server stopped.")
