"""Local Vision-Language Model (VLM) client supporting Ollama with strict JSON parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import json
import logging
from queue import Empty, Queue
import re
import threading
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from activity_detector.config.settings import VlmConfig

logger = logging.getLogger("activity_detector.vlm")


class VlmHealthStatus(BaseModel):
    """Health diagnostic for the local VLM engine."""
    available: bool
    provider: str
    model_name: str
    message: str
    supports_vision: bool = True
    installed_models: List[str] = Field(default_factory=list)


class VlmResponseSchema(BaseModel):
    """Strict JSON schema required from the VLM."""
    step_recognized: bool = Field(
        default=False,
        description="Whether the expected action/step was observed in the image."
    )
    action_description: str = Field(
        default="No action described",
        description="Concise description of the operator's current action."
    )
    detected_items: List[str] = Field(
        default_factory=list,
        description="Items, containers, or tools visible in the frame."
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Self-reported confidence score between 0.0 and 1.0."
    )
    is_uncertain: bool = Field(
        default=False,
        description="Set to true if image is ambiguous, occluded, or inconclusive."
    )
    reasoning: str = Field(
        default="",
        description="Brief physical evidence justifying the conclusion."
    )


class VlmInterpretation(BaseModel):
    """Audited result of a single sampled frame analysis."""
    timestamp: float = Field(default_factory=time.time)
    schema_data: VlmResponseSchema
    raw_text: str
    model_name: str
    is_valid: bool = True
    latency_seconds: float = 0.0


class BaseVlmClient(ABC):
    """Abstract interface for local Vision-Language Model providers."""

    @abstractmethod
    def submit_sample(
        self,
        frame: np.ndarray,
        step_name: str,
        step_instruction: str,
        expected_items: List[str],
    ) -> bool:
        """Submits a sampled frame for background analysis."""
        pass

    @abstractmethod
    def get_latest_interpretation(self) -> Optional[VlmInterpretation]:
        """Returns the most recent interpretation if available."""
        pass

    @abstractmethod
    def check_health(self) -> VlmHealthStatus:
        """Probes the local model provider and reports readiness."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Shuts down background worker threads cleanly."""
        pass


class OllamaVlmClient(BaseVlmClient):
    """Threaded Ollama vision client with rate limiting and strict schema validation."""

    def __init__(self, config: VlmConfig) -> None:
        self.config = config
        self._queue: Queue = Queue(maxsize=1)  # Drop old samples so we never lag behind
        self._running: bool = True
        self._lock = threading.Lock()
        self._latest_interpretation: Optional[VlmInterpretation] = None
        self._last_sample_time: float = 0.0
        self._active_model: str = config.model

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="OllamaVlmWorkerThread",
            daemon=True
        )
        self._thread.start()

    def submit_sample(
        self,
        frame: np.ndarray,
        step_name: str,
        step_instruction: str,
        expected_items: List[str],
    ) -> bool:
        """Pushes a sampled frame to the background queue if interval has elapsed."""
        if not self.config.enabled or frame is None:
            return False

        now = time.time()
        if (now - self._last_sample_time) < self.config.sample_interval_seconds:
            return False

        self._last_sample_time = now

        # Compress to 640px wide JPEG to accelerate local base64 transfer & inference
        h, w = frame.shape[:2]
        target_w = 640
        target_h = int(h * (target_w / float(w)))
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Clear any stale queued item
        try:
            self._queue.get_nowait()
        except Empty:
            pass

        try:
            self._queue.put_nowait({
                "frame": resized,
                "step_name": step_name,
                "step_instruction": step_instruction,
                "expected_items": expected_items,
                "timestamp": now,
            })
            return True
        except Exception:
            return False

    def get_latest_interpretation(self) -> Optional[VlmInterpretation]:
        """Thread-safe retrieval of the latest VLM interpretation."""
        with self._lock:
            return self._latest_interpretation

    def stop(self) -> None:
        """Stops the background worker thread."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _worker_loop(self) -> None:
        """Continuously pulls samples and queries Ollama."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None or not self._running:
                break

            t0 = time.time()
            interp = self._query_ollama(
                frame=item["frame"],
                step_name=item["step_name"],
                step_instruction=item["step_instruction"],
                expected_items=item["expected_items"]
            )
            interp.latency_seconds = round(time.time() - t0, 2)

            with self._lock:
                self._latest_interpretation = interp

    def _query_ollama(
        self,
        frame: np.ndarray,
        step_name: str,
        step_instruction: str,
        expected_items: List[str],
    ) -> VlmInterpretation:
        """Sends single frame and strict JSON prompt to Ollama."""
        try:
            import ollama  # type: ignore

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_img = base64.b64encode(buf).decode("utf-8")

            items_str = ", ".join(expected_items) if expected_items else "reagents, tools, containers"
            prompt = (
                f"You are an offline laboratory activity monitor. Analyze this webcam image for:\n"
                f"Step: {step_name}\n"
                f"Instruction: {step_instruction}\n"
                f"Expected items: {items_str}\n\n"
                f"Reply ONLY in valid JSON matching this schema exactly:\n"
                f'{{"step_recognized": true, "action_description": "...", "detected_items": ["..."], '
                f'"confidence": 0.85, "is_uncertain": false, "reasoning": "..."}}\n'
                f"If the image is blurry, occluded, or inconclusive, set is_uncertain to true."
            )

            client = ollama.Client(host=self.config.host)
            options = {
                "num_ctx": self.config.num_ctx,
                "num_gpu": self.config.num_gpu,
            }

            resp = client.chat(
                model=self._active_model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [b64_img],
                }],
                options=options,
            )

            raw_content = resp["message"]["content"]
            schema_data, is_valid = self._parse_json_response(raw_content)

            return VlmInterpretation(
                timestamp=time.time(),
                schema_data=schema_data,
                raw_text=raw_content,
                model_name=self._active_model,
                is_valid=is_valid,
            )

        except Exception as e:
            logger.warning(f"Ollama inference error with {self._active_model}: {e}")
            # Return safe uncertain fallback
            fallback_schema = VlmResponseSchema(
                step_recognized=False,
                action_description=f"Inference error: {type(e).__name__}",
                detected_items=[],
                confidence=0.0,
                is_uncertain=True,
                reasoning=str(e),
            )
            return VlmInterpretation(
                timestamp=time.time(),
                schema_data=fallback_schema,
                raw_text=str(e),
                model_name=self._active_model,
                is_valid=False,
            )

    def _parse_json_response(self, text: str) -> Tuple[VlmResponseSchema, bool]:
        """Extracts and validates JSON from raw model output."""
        cleaned = text.strip()

        # Handle ```json ... ``` blocks
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

        # Handle bare { ... }
        if not (cleaned.startswith("{") and cleaned.endswith("}")):
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
            schema = VlmResponseSchema.model_validate(parsed)
            return schema, True
        except (json.JSONDecodeError, ValidationError) as err:
            logger.debug(f"Failed to parse VLM JSON: {err}. Raw text: {text[:120]}")
            # Return uncertain fallback
            return VlmResponseSchema(
                step_recognized=False,
                action_description="Output format invalid; flagged for operator review",
                detected_items=[],
                confidence=0.0,
                is_uncertain=True,
                reasoning="VLM produced unparseable JSON",
            ), False

    def check_health(self) -> VlmHealthStatus:
        """Tests connection to local Ollama server and lists installed vision models."""
        try:
            import ollama  # type: ignore
            client = ollama.Client(host=self.config.host)
            models_resp = client.list()

            installed_names = []
            if hasattr(models_resp, "models"):
                installed_names = [m.model for m in models_resp.models]
            elif isinstance(models_resp, dict) and "models" in models_resp:
                installed_names = [m.get("name", "") for m in models_resp["models"]]

            has_primary = any(self.config.model in name for name in installed_names)
            has_alt = any(self.config.alternative_model in name for name in installed_names)

            if has_primary:
                self._active_model = self.config.model
                msg = f"Ollama online. Primary model '{self.config.model}' is available."
            elif has_alt:
                self._active_model = self.config.alternative_model
                msg = f"Primary model missing; using alternative '{self.config.alternative_model}'."
            else:
                msg = f"Ollama online, but neither '{self.config.model}' nor '{self.config.alternative_model}' was found."

            return VlmHealthStatus(
                available=True,
                provider="ollama",
                model_name=self._active_model,
                message=msg,
                supports_vision=True,
                installed_models=installed_names,
            )
        except Exception as e:
            return VlmHealthStatus(
                available=False,
                provider="ollama",
                model_name=self.config.model,
                message=f"Cannot reach Ollama at {self.config.host}: {e}",
                supports_vision=False,
                installed_models=[],
            )


class MockVlmClient(BaseVlmClient):
    """Deterministic mock VLM client for unit testing and offline development."""

    def __init__(self, mock_step_recognized: bool = True, mock_confidence: float = 0.85) -> None:
        self.mock_step_recognized = mock_step_recognized
        self.mock_confidence = mock_confidence
        self._latest: Optional[VlmInterpretation] = None

    def submit_sample(
        self,
        frame: np.ndarray,
        step_name: str,
        step_instruction: str,
        expected_items: List[str],
    ) -> bool:
        schema = VlmResponseSchema(
            step_recognized=self.mock_step_recognized,
            action_description=f"Mock detected action for {step_name}",
            detected_items=expected_items,
            confidence=self.mock_confidence,
            is_uncertain=False,
            reasoning="Mock verification",
        )
        self._latest = VlmInterpretation(
            timestamp=time.time(),
            schema_data=schema,
            raw_text='{"mock": true}',
            model_name="mock-vlm",
            is_valid=True,
            latency_seconds=0.05,
        )
        return True

    def get_latest_interpretation(self) -> Optional[VlmInterpretation]:
        return self._latest

    def check_health(self) -> VlmHealthStatus:
        return VlmHealthStatus(
            available=True,
            provider="mock",
            model_name="mock-vlm",
            message="Mock VLM is operational",
            supports_vision=True,
            installed_models=["mock-vlm"],
        )

    def stop(self) -> None:
        pass
