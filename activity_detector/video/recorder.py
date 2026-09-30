"""Local video recording with background writing and failure resilience."""

from __future__ import annotations

import logging
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from typing import Optional
import cv2
import numpy as np

from activity_detector.config.settings import RecordingConfig

logger = logging.getLogger("activity_detector.recorder")


class VideoRecorder:
    """Threaded local video writer that guarantees session log survival on video failure."""

    def __init__(self, config: RecordingConfig) -> None:
        self.config = config
        self._writer: Optional[cv2.VideoWriter] = None
        self._queue: Queue = Queue(maxsize=120)
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._output_file: Optional[Path] = None

        self.is_recording: bool = False
        self.has_error: bool = False
        self.error_message: str = ""
        self.frames_written: int = 0

    def start_recording(self, target_directory: Path, filename: Optional[str] = None, width: int = 1280, height: int = 720) -> bool:
        """Initializes a new video recording session."""
        if self.is_recording:
            return True

        self.has_error = False
        self.error_message = ""
        self.frames_written = 0
        target_directory = Path(target_directory)
        target_directory.mkdir(parents=True, exist_ok=True)

        ext = self.config.video_format.lower()
        if ext not in ("mp4", "avi"):
            ext = "mp4"

        self._width = width
        self._height = height
        fname = filename or f"session_recording_{int(time.time())}.{ext}"
        self._output_file = target_directory / fname

        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext == "mp4" else "XVID"))

        try:
            self._writer = cv2.VideoWriter(
                str(self._output_file),
                fourcc,
                self.config.record_fps,
                (width, height)
            )

            if not self._writer.isOpened():
                # Fallback to AVI with MJPG if mp4v fails
                fallback_file = target_directory / f"session_recording_{int(time.time())}.avi"
                fallback_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self._writer = cv2.VideoWriter(
                    str(fallback_file),
                    fallback_fourcc,
                    self.config.record_fps,
                    (width, height)
                )
                self._output_file = fallback_file

            if not self._writer.isOpened():
                self.has_error = True
                self.error_message = "Could not initialize video codec / VideoWriter."
                logger.error(self.error_message)
                return False

            self.is_recording = True
            self._running = True
            self._thread = threading.Thread(
                target=self._write_worker,
                name="VideoWriterThread",
                daemon=True
            )
            self._thread.start()
            logger.info(f"Video recording started: {self._output_file}")
            return True

        except Exception as e:
            self.has_error = True
            self.error_message = f"Video recorder startup failure: {e}"
            logger.error(self.error_message)
            return False

    def push_frame(self, frame: np.ndarray) -> bool:
        """Pushes a frame to the recording queue."""
        if not self.is_recording or self.has_error or frame is None:
            return False

        try:
            # If queue is full, drop oldest to avoid memory blowup
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put_nowait(frame.copy())
            return True
        except Exception:
            return False

    def stop_recording(self) -> Optional[Path]:
        """Flushes remaining frames and closes video writer."""
        self._running = False
        self.is_recording = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

        if self._writer:
            try:
                self._writer.release()
            except Exception as e:
                logger.error(f"Error releasing VideoWriter: {e}")
            self._writer = None

        logger.info(f"Video recording stopped. Total frames: {self.frames_written}")
        return self._output_file

    def _write_worker(self) -> None:
        """Background thread writing queued frames to disk."""
        while self._running or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.2)
            except Empty:
                continue

            if frame is None:
                break

            if self._writer and self._writer.isOpened():
                try:
                    if hasattr(self, "_width") and (frame.shape[1], frame.shape[0]) != (self._width, self._height):
                        frame = cv2.resize(frame, (self._width, self._height))
                    self._writer.write(frame)
                    self.frames_written += 1
                except Exception as e:
                    self.has_error = True
                    self.error_message = f"Write error: {e}"
                    logger.error(f"Video frame write failed: {e}")
                    break

        # Flush remaining frames
        while not self._queue.empty():
            try:
                rem_frame = self._queue.get_nowait()
                if self._writer and self._writer.isOpened() and not self.has_error:
                    self._writer.write(rem_frame)
                    self.frames_written += 1
            except Empty:
                break
            except Exception:
                break
