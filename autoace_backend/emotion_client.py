from __future__ import annotations

import base64
import os
import time

import httpx
from pydantic import BaseModel, Field, ValidationError


DEFAULT_EMOTION_SERVICE_URL = (
    "https://mbakajoe26--autoace-emotion-service-emotionmodel-predict.modal.run"
)


class EmotionSegmentResult(BaseModel):
    index: int
    start: float
    end: float
    duration: float = Field(gt=0)
    text: str = ""
    scores: dict[str, float]
    predicted: str
    inference_seconds: float = Field(ge=0)


class EmotionAggregate(BaseModel):
    mode: str
    segment_count: int = Field(gt=0)
    customer_speech_seconds: float = Field(gt=0)
    segments: list[EmotionSegmentResult]
    aggregate_scores: dict[str, float]
    aggregate_predicted: str
    negative_affect: float = Field(ge=0)
    positive_affect: float = Field(ge=0)
    neutral_affect: float = Field(ge=0)
    inference_seconds: float = Field(ge=0)
    request_seconds: float = Field(ge=0)


class EmotionServiceClient:
    def __init__(self, *, client: httpx.Client | None = None) -> None:
        self.url = os.getenv("EMOTION_SERVICE_URL", DEFAULT_EMOTION_SERVICE_URL)
        self.timeout = float(os.getenv("EMOTION_SERVICE_TIMEOUT_SECONDS", "240"))
        self._client = client

    def analyze(self, audio_bytes: bytes, segments: list[dict]) -> EmotionAggregate:
        if not segments:
            raise ValueError("At least one customer speech segment is required.")
        started = time.perf_counter()
        payload = {
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "segments": segments,
        }
        try:
            if self._client is not None:
                response = self._client.post(self.url, json=payload)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.url, json=payload)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Emotion service request failed: {exc}") from exc

        if response.is_error:
            raise RuntimeError(
                f"Emotion service failed ({response.status_code}): {response.text[:500]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("Emotion service returned a non-JSON response.") from exc
        if not isinstance(body, dict):
            raise RuntimeError("Emotion service returned an unexpected response shape.")
        if body.get("error"):
            raise RuntimeError(f"Emotion service provider error: {body['error']}")
        body["request_seconds"] = round(time.perf_counter() - started, 3)
        try:
            result = EmotionAggregate.model_validate(body)
        except ValidationError as exc:
            raise RuntimeError(f"Emotion service response validation failed: {exc}") from exc
        if result.mode != "segments" or result.segment_count != len(result.segments):
            raise RuntimeError("Emotion service returned inconsistent segment metadata.")
        return result
