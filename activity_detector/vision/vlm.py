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

    @property
    def is_paused(self) -> bool:
        """True if background sampling is paused."""
        return False

    def pause(self) -> None:
        """Pauses background sampling."""
        pass

    def resume(self) -> None:
        """Resumes background sampling."""
        pass

    @property
    def is_degraded(self) -> bool:
        """True if the VLM client has encountered persistent failures and is degraded."""
        return False

    @property
    def degraded_reason(self) -> str:
        """Human-readable reason for degradation if degraded."""
        return ""


class OllamaVlmClient(BaseVlmClient):
    """Threaded Ollama vision client with rate limiting and strict schema validation."""

    def __init__(self, config: VlmConfig) -> None:
        self.config = config
        self._queue: Queue = Queue(maxsize=1)  # Drop old samples so we never lag behind
        self._running: bool = True
        self._paused: bool = False
        self._lock = threading.Lock()
        self._latest_interpretation: Optional[VlmInterpretation] = None
        self._last_sample_time: float = 0.0
        self._active_model: str = config.model

        # Cooldown & degradation tracking
        self._consecutive_failures: int = 0
        self._cooldown_until: float = 0.0
        self._is_degraded: bool = False
        self._degraded_reason: str = ""

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="OllamaVlmWorkerThread",
            daemon=True
        )
        self._thread.start()

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_degraded(self) -> bool:
        return self._is_degraded

    @property
    def degraded_reason(self) -> str:
        return self._degraded_reason

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def pause(self) -> None:
        self._paused = True
        try:
            self._queue.get_nowait()
        except Empty:
            pass
        logger.info("VLM background sampling PAUSED.")

    def resume(self) -> None:
        self._paused = False
        logger.info("VLM background sampling RESUMED.")

    def submit_sample(
        self,
        frame: np.ndarray,
        target_object: str,
        step_name: str,
        step_instruction: str,
        expected_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Pushes a sampled frame to the background queue if interval has elapsed."""
        if not self.config.enabled or self._paused or frame is None:
            return False

        now = time.time()
        # Suppress queueing during failure cooldown
        if now < self._cooldown_until:
            return False

        if (now - self._last_sample_time) < self.config.sample_interval_seconds:
            return False

        self._last_sample_time = now

        # Compress to configurable dimensions (default max 480px) and quality (default 75)
        h, w = frame.shape[:2]
        max_dim = getattr(self.config, "image_max_dimension", 480)
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            target_w = int(w * scale)
            target_h = int(h * scale)
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            target_w, target_h = w, h
            resized = frame

        quality = getattr(self.config, "jpeg_quality", 75)
        _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        b64_img = base64.b64encode(buf).decode("utf-8")

        # Clear any stale queued item
        try:
            self._queue.get_nowait()
        except Empty:
            pass

        try:
            self._queue.put_nowait({
                "b64_img": b64_img,
                "width": target_w,
                "height": target_h,
                "bytes_len": len(buf),
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

            # If currently in failure cooldown, wait before sending any new request
            now = time.time()
            if now < self._cooldown_until:
                continue

            # Discard outdated pending frames if queued longer than 3.0s
            queue_age = now - item.get("timestamp", now)
            if queue_age > 3.0:
                logger.info(f"Discarding outdated pending VLM sample ({queue_age:.2f}s old).")
                continue

            t0 = time.time()
            interp = self._query_ollama(
                b64_img=item["b64_img"],
                width=item["width"],
                height=item["height"],
                bytes_len=item["bytes_len"],
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
        b64_img: str,
        width: int,
        height: int,
        bytes_len: int,
        target_object: str,
        step_name: str,
        step_instruction: str,
        expected_state: Dict[str, Any],
    ) -> VlmInterpretation:
        """Sends pre-encoded frame and strict JSON prompt to Ollama with timeout and logging."""
        t_start = time.time()
        try:
            import ollama  # type: ignore

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

            # Set explicit timeout so unresponsive daemon cannot hang worker
            client = ollama.Client(host=self.config.host, timeout=self.config.timeout_seconds)
            options = {
                "num_ctx": self.config.num_ctx,
                "num_gpu": self.config.num_gpu,
                "num_predict": 60,
            }

            logger.info(
                f"Submitting frame to Ollama ({self._active_model}) for '{target_object}' "
                f"(step: '{step_name}') | size: {width}x{height} ({bytes_len} bytes JPEG)..."
            )
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
            duration = round(time.time() - t_start, 2)
            logger.info(
                f"Ollama inference completed in {duration}s with {self._active_model} "
                f"(valid={is_valid}, conf={schema_data.confidence:.2f})."
            )

            # Success resets failure counter and cooldown
            self._consecutive_failures = 0
            self._cooldown_until = 0.0
            self._is_degraded = False
            self._degraded_reason = ""

            return VlmInterpretation(
                timestamp=time.time(),
                schema_data=schema_data,
                target_object=target_object,
                raw_text=raw_content,
                model_name=self._active_model,
                is_valid=is_valid,
                latency_seconds=duration,
            )

        except Exception as e:
            duration = round(time.time() - t_start, 2)
            self._consecutive_failures += 1
            base_cd = getattr(self.config, "failure_cooldown_seconds", 5.0)
            backoff = min(60.0, base_cd * (1.5 ** min(self._consecutive_failures - 1, 5)))
            self._cooldown_until = time.time() + backoff

            err_msg = str(e)
            if "out-of-memory" in err_msg.lower() or "failed to allocate" in err_msg.lower():
                short_err = "OOM: Insufficient CPU RAM for model"
            elif "timed out" in err_msg.lower():
                short_err = f"Timeout after {duration}s"
            else:
                short_err = f"{type(e).__name__}: {err_msg[:45]}"

            if self._consecutive_failures >= getattr(self.config, "max_consecutive_failures", 3):
                self._is_degraded = True
                self._degraded_reason = short_err

            logger.warning(
                f"Ollama inference exception #{self._consecutive_failures} after {duration}s with {self._active_model}: {e}. "
                f"Backing off for {backoff:.1f}s."
            )
            fallback_schema = VlmResponseSchema(
                object_visible=False,
                object_description=f"Inference error: {type(e).__name__}",
                open_or_closed="unknown",
                held_or_on_surface="unknown",
                location="unknown",
                confidence=0.0,
                is_uncertain=True,
                reasoning=short_err,
            )
            return VlmInterpretation(
                timestamp=time.time(),
                schema_data=fallback_schema,
                target_object=target_object,
                raw_text=str(e),
                model_name=self._active_model,
                is_valid=False,
                latency_seconds=duration,
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
        self._paused: bool = False
        self.submit_count: int = 0
        self._is_degraded: bool = False
        self._degraded_reason: str = ""

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_degraded(self) -> bool:
        return self._is_degraded

    @property
    def degraded_reason(self) -> str:
        return self._degraded_reason

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def submit_sample(
        self,
        frame: np.ndarray,
        target_object: str,
        step_name: str,
        step_instruction: str,
        expected_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self._paused:
            return False
        self.submit_count += 1
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
