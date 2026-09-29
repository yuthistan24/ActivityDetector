"""Local Vision-Language Model (VLM) client for generic object inspection via Ollama."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import json
import logging
from queue import Empty, Queue
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
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
    """Strict, validated JSON schema for generic object visual evidence."""
    object_visible: bool = Field(
        default=False,
        description="Whether the specified target object (e.g. notebook) is clearly visible in the image."
    )
    object_description: str = Field(
        default="",
        description="Visual description of the target object (color, cover texture, appearance)."
    )
    open_or_closed: str = Field(
        default="unknown",
        description="'open' if pages/interior are visible, 'closed' if covers are shut, or 'unknown'."
    )
    held_or_on_surface: str = Field(
        default="unknown",
        description="'held' if held by hand, 'on_surface' if resting on table/mat, or 'unknown'."
    )
    location: str = Field(
        default="unknown",
        description="Approximate position: 'workspace_center', 'stowed_area', 'prep_left', or 'unknown'."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this visual observation (0.0 to 1.0)."
    )
    is_uncertain: bool = Field(
        default=False,
        description="True if image is blurry, partially out of frame, ambiguous, or lighting is poor."
    )
    reasoning: str = Field(
        default="",
        description="Concise physical evidence explaining the classification."
    )
    step_recognized: Optional[bool] = Field(
        default=None,
        description="Optional alias/legacy flag indicating if step criteria was met."
    )
    action_description: Optional[str] = Field(
        default=None,
        description="Optional legacy description of action observed."
    )
    detected_items: List[str] = Field(
        default_factory=list,
        description="Optional detected item labels."
    )


class VlmInterpretation(BaseModel):
    """Audited result of a single sampled frame analysis."""
    timestamp: float = Field(default_factory=time.time)
    schema_data: VlmResponseSchema
    target_object: str = "notebook"
    raw_text: str = ""
    model_name: str = ""
    is_valid: bool = True
    latency_seconds: float = 0.0


class BaseVlmClient(ABC):
    """Abstract interface for local Vision-Language Model providers."""

    @abstractmethod
    def submit_sample(
        self,
        frame: np.ndarray,
        target_object: str,
        step_name: str,
        step_instruction: str,
        expected_state: Optional[Dict[str, Any]] = None,
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
        target_object: str,
        step_name: str,
        step_instruction: str,
        expected_state: Optional[Dict[str, Any]] = None,
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
                "target_object": target_object,
                "step_name": step_name,
                "step_instruction": step_instruction,
                "expected_state": expected_state or {},
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
                target_object=item["target_object"],
                step_name=item["step_name"],
                step_instruction=item["step_instruction"],
                expected_state=item["expected_state"],
            )
            interp.latency_seconds = round(time.time() - t0, 2)

            with self._lock:
                self._latest_interpretation = interp

    def _query_ollama(
        self,
        frame: np.ndarray,
        target_object: str,
        step_name: str,
        step_instruction: str,
        expected_state: Dict[str, Any],
    ) -> VlmInterpretation:
        """Sends single frame and strict JSON prompt to Ollama."""
        try:
            import ollama  # type: ignore

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_img = base64.b64encode(buf).decode("utf-8")

            prompt = (
                f"You are an offline lab vision assistant inspecting tabletop procedure execution.\n"
                f"Target object to inspect: '{target_object}'\n"
                f"Active step: '{step_name}'\n"
                f"Current instruction: '{step_instruction}'\n\n"
                f"Carefully inspect the image for visible physical evidence only. Do NOT speculate or guess operator intent.\n"
                f"Report the visible physical state of the '{target_object}'.\n"
                f"Respond ONLY in valid JSON matching this schema exactly:\n"
                f'{{\n'
                f'  "object_visible": true,\n'
                f'  "object_description": "visual appearance of the {target_object}",\n'
                f'  "open_or_closed": "open" | "closed" | "unknown",\n'
                f'  "held_or_on_surface": "held" | "on_surface" | "unknown",\n'
                f'  "location": "workspace_center" | "stowed_area" | "prep_left" | "unknown",\n'
                f'  "confidence": 0.85,\n'
                f'  "is_uncertain": false,\n'
                f'  "reasoning": "physical evidence observed"\n'
                f'}}\n\n'
                f"If the image is blurry, occluded, or inconclusive, set \"is_uncertain\": true."
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
                target_object=target_object,
                raw_text=raw_content,
                model_name=self._active_model,
                is_valid=is_valid,
            )

        except Exception as e:
            logger.warning(f"Ollama inference error with {self._active_model}: {e}")
            fallback_schema = VlmResponseSchema(
                object_visible=False,
                object_description=f"Inference error: {type(e).__name__}",
                open_or_closed="unknown",
                held_or_on_surface="unknown",
                location="unknown",
                confidence=0.0,
                is_uncertain=True,
                reasoning=str(e),
            )
            return VlmInterpretation(
                timestamp=time.time(),
                schema_data=fallback_schema,
                target_object=target_object,
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
            # Normalize fields if legacy keys present
            if "step_recognized" in parsed and "object_visible" not in parsed:
                parsed["object_visible"] = parsed["step_recognized"]
            if "action_description" in parsed and "object_description" not in parsed:
                parsed["object_description"] = parsed["action_description"]

            schema = VlmResponseSchema.model_validate(parsed)
            # Low confidence must be treated as uncertain
            if schema.confidence < 0.40:
                schema.is_uncertain = True

            return schema, True
        except (json.JSONDecodeError, ValidationError) as err:
            logger.debug(f"Failed to parse VLM JSON: {err}. Raw text: {text[:120]}")
            return VlmResponseSchema(
                object_visible=False,
                object_description="Output format invalid; flagged for operator review",
                open_or_closed="unknown",
                held_or_on_surface="unknown",
                location="unknown",
                confidence=0.0,
                is_uncertain=True,
                reasoning="VLM produced unparseable JSON",
            ), False

    def check_health(self) -> VlmHealthStatus:
        """Tests connection to local Ollama server and checks installed vision models."""
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
                msg = f"Ollama online. Primary vision model '{self.config.model}' is available."
            elif has_alt:
                self._active_model = self.config.alternative_model
                msg = f"Primary model missing; using installed alternative '{self.config.alternative_model}'."
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

    def __init__(
        self,
        mock_object_visible: bool = True,
        mock_open_closed: str = "closed",
        mock_location: str = "workspace_center",
        mock_confidence: float = 0.85,
        mock_uncertain: bool = False,
        mock_step_recognized: Optional[bool] = None,
    ) -> None:
        self.mock_object_visible = mock_object_visible
        self.mock_open_closed = mock_open_closed
        self.mock_location = mock_location
        self.mock_confidence = mock_confidence
        self.mock_uncertain = mock_uncertain
        self.mock_step_recognized = mock_step_recognized
        self._latest: Optional[VlmInterpretation] = None

    def submit_sample(
        self,
        frame: np.ndarray,
        target_object: str,
        step_name: str,
        step_instruction: str,
        expected_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        schema = VlmResponseSchema(
            object_visible=self.mock_object_visible,
            object_description=f"Mock description of {target_object}",
            open_or_closed=self.mock_open_closed,
            held_or_on_surface="on_surface",
            location=self.mock_location,
            confidence=self.mock_confidence,
            is_uncertain=self.mock_uncertain,
            reasoning="Mock verification evidence",
            step_recognized=self.mock_step_recognized,
        )
        self._latest = VlmInterpretation(
            timestamp=time.time(),
            schema_data=schema,
            target_object=target_object,
            raw_text='{"mock": true}',
            model_name="mock-vlm",
            is_valid=True,
            latency_seconds=0.02,
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
