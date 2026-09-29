"""Local Text-to-Speech (TTS) voice announcements with cooldowns and mute control."""

from __future__ import annotations

import logging
from queue import Empty, Queue
import threading
import time
from typing import Dict, Optional

from activity_detector.config.settings import AudioConfig

logger = logging.getLogger("activity_detector.audio")


class SpeechPrompter:
    """Asynchronous, thread-safe voice prompter using local TTS."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.is_muted: bool = not config.enabled
        self.available: bool = False
        self._queue: Queue = Queue(maxsize=10)
        self._running: bool = True
        self._lock = threading.Lock()
        self._last_spoken: Dict[str, float] = {}

        # Test TTS availability
        try:
            import pyttsx3  # type: ignore
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            self.available = len(voices) > 0
            engine.stop()
            del engine
        except Exception as e:
            self.available = False
            logger.warning(f"pyttsx3 TTS unavailable: {e}. Voice alerts will be silent.")

        if self.available and self.config.enabled:
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="TTSWorkerThread",
                daemon=True
            )
            self._thread.start()
        else:
            self._thread = None

    def toggle_mute(self) -> bool:
        """Toggles mute state and returns the new muted status."""
        self.is_muted = not self.is_muted
        logger.info(f"Audio muted: {self.is_muted}")
        return self.is_muted

    def set_muted(self, muted: bool) -> None:
        """Explicitly sets mute state."""
        self.is_muted = muted

    def announce_step(self, step_order: int, step_name: str, instruction: str) -> None:
        """Announces a newly active step."""
        text = f"Step {step_order}: {step_name}. {instruction}"
        cooldown_key = f"step_{step_order}"
        self.speak(text, cooldown_key=cooldown_key)

    def announce_warning(self, warning_type: str, message: str) -> None:
        """Announces an operational warning or sequence anomaly."""
        text = f"Alert: {message}"
        cooldown_key = f"warning_{warning_type}"
        self.speak(text, cooldown_key=cooldown_key)

    def speak(self, text: str, cooldown_key: Optional[str] = None) -> bool:
        """Pushes speech utterance to the queue if cooldown has expired and unmuted."""
        if not self.available or self.is_muted:
            return False

        now = time.time()
        if cooldown_key:
            with self._lock:
                last_time = self._last_spoken.get(cooldown_key, 0.0)
                if (now - last_time) < self.config.cooldown_seconds:
                    return False
                self._last_spoken[cooldown_key] = now

        try:
            self._queue.put_nowait(text)
            return True
        except Exception:
            return False

    def stop(self) -> None:
        """Gracefully halts TTS thread."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def _worker_loop(self) -> None:
        """Background thread initializing dedicated COM/TTS engine."""
        try:
            import pyttsx3  # type: ignore
            engine = pyttsx3.init()
            engine.setProperty("rate", self.config.rate)
            engine.setProperty("volume", self.config.volume)

            voices = engine.getProperty("voices")
            if voices and 0 <= self.config.voice_index < len(voices):
                engine.setProperty("voice", voices[self.config.voice_index].id)

            while self._running:
                try:
                    text = self._queue.get(timeout=0.5)
                except Empty:
                    continue

                if text is None or not self._running:
                    break

                if not self.is_muted:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as err:
                        logger.error(f"TTS utterance failure: {err}")
                        time.sleep(0.5)

            engine.stop()
        except Exception as e:
            logger.error(f"TTS background thread terminated: {e}")
            self.available = False
