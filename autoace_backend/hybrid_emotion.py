from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError


GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_GROQ_MODEL = "whisper-large-v3"
DEFAULT_NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"


class EmotionPrediction(BaseModel):
    emotional_tone: Literal[
        "neutral", "satisfied", "frustrated", "upset", "distressed"
    ]
    emotional_intensity: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)


class TranscriptSegment(BaseModel):
    start: float | None = None
    end: float | None = None
    text: str = ""


class TranscriptResult(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    model: str
    request_seconds: float


class LlmUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class EmotionClassificationResult(BaseModel):
    prediction: EmotionPrediction
    model: str
    request_seconds: float
    usage: LlmUsage = Field(default_factory=LlmUsage)


@dataclass(slots=True)
class HybridEmotionResult:
    name: str
    transcript: TranscriptResult
    classification: EmotionClassificationResult


NEMOTRON_SYSTEM_PROMPT = """/no_think
You classify the CUSTOMER's emotional state in an automotive dealership service-call transcript.
The transcript may contain both the customer and the dealership representative/AI agent, and it may not contain explicit speaker labels. Infer the customer from conversational role and content.

Classify only these two dimensions:

EMOTIONAL TONE
- neutral: no clear positive or negative emotion; ordinary transactional conversation.
- satisfied: pleased, relieved, appreciative, grateful, or clearly positive about the interaction/outcome.
- frustrated: annoyed, impatient, or dissatisfied, but not strongly angry or emotionally escalated.
- upset: clearly angry, agitated, strongly dissatisfied, or forcefully complaining.
- distressed: overwhelmed, panicked, crying, fearful, or otherwise highly emotionally escalated.

EMOTIONAL INTENSITY
- low: subtle or mild emotion.
- medium: clear and sustained emotion.
- high: strong/escalated emotion likely to require attention.

Important rules:
- Judge the CUSTOMER, not the service representative/AI agent.
- Use the complete conversation and distinguish what the customer says from what the agent says.
- Do not assume that describing a vehicle problem means the customer is frustrated.
- Do not assume that requesting service, correcting details, or asking questions is negative emotion.
- Do not infer acoustic properties such as noise, overlap, silence, or vocal loudness from this transcript; this stage is text-only.
- If the evidence is ambiguous, prefer neutral and lower confidence rather than inventing emotion.

Return one JSON object only, with exactly:
{
  "emotional_tone": "neutral|satisfied|frustrated|upset|distressed",
  "emotional_intensity": "low|medium|high",
  "confidence": 0.0
}
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Nemotron response did not contain a JSON object.")
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Nemotron response contained invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Nemotron response JSON must be an object.")
    return payload


class GroqWhisperTranscriber:
    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        self.model = os.getenv("GROQ_WHISPER_MODEL", DEFAULT_GROQ_MODEL)
        self.timeout = float(os.getenv("GROQ_TIMEOUT_SECONDS", "180"))

    def transcribe(self, name: str, audio_bytes: bytes, mime_type: str) -> TranscriptResult:
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    GROQ_TRANSCRIPTIONS_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": (name, audio_bytes, mime_type)},
                    data=[
                        ("model", self.model),
                        ("response_format", "verbose_json"),
                        ("language", "en"),
                        ("temperature", "0"),
                        ("timestamp_granularities[]", "segment"),
                    ],
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Groq transcription request failed: {exc}") from exc

        request_seconds = time.perf_counter() - started
        if response.is_error:
            raise RuntimeError(
                f"Groq transcription failed ({response.status_code}): {response.text[:500]}"
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("Groq returned a non-JSON transcription response.") from exc

        text = str(payload.get("text") or "").strip()
        if not text:
            raise RuntimeError("Groq returned an empty transcript.")

        segments: list[TranscriptSegment] = []
        raw_segments = payload.get("segments") or []
        if isinstance(raw_segments, list):
            for raw_segment in raw_segments:
                if not isinstance(raw_segment, dict):
                    continue
                segments.append(
                    TranscriptSegment(
                        start=_as_float(raw_segment.get("start")),
                        end=_as_float(raw_segment.get("end")),
                        text=str(raw_segment.get("text") or "").strip(),
                    )
                )

        return TranscriptResult(
            text=text,
            language=_as_optional_string(payload.get("language")),
            duration=_as_float(payload.get("duration")),
            segments=segments,
            model=self.model,
            request_seconds=round(request_seconds, 3),
        )


class NemotronEmotionClassifier:
    def __init__(self) -> None:
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is not configured.")
        self.model = os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
        self.timeout = float(os.getenv("NVIDIA_TIMEOUT_SECONDS", "180"))

    def classify(self, transcript: str) -> EmotionClassificationResult:
        user_prompt = (
            "Classify the customer's emotional tone and intensity from this complete transcript. "
            "Return only the JSON object requested by the system instructions.\n\n"
            "TRANSCRIPT:\n"
            f"{transcript.strip()}"
        )

        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    NVIDIA_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": NEMOTRON_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_tokens": 256,
                        "stream": False,
                    },
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"NVIDIA Nemotron request failed: {exc}") from exc

        request_seconds = time.perf_counter() - started
        if response.is_error:
            raise RuntimeError(
                f"NVIDIA Nemotron failed ({response.status_code}): {response.text[:500]}"
            )

        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("NVIDIA returned an unexpected chat-completion response.") from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("NVIDIA returned an empty classification response.")

        try:
            prediction = EmotionPrediction.model_validate(_extract_json_object(content))
        except (ValidationError, ValueError) as exc:
            raise RuntimeError(f"Nemotron structured-output validation failed: {exc}") from exc

        raw_usage = payload.get("usage") if isinstance(payload, dict) else None
        usage = LlmUsage()
        if isinstance(raw_usage, dict):
            usage = LlmUsage(
                prompt_tokens=_as_int(raw_usage.get("prompt_tokens")),
                completion_tokens=_as_int(raw_usage.get("completion_tokens")),
                total_tokens=_as_int(raw_usage.get("total_tokens")),
            )

        return EmotionClassificationResult(
            prediction=prediction,
            model=self.model,
            request_seconds=round(request_seconds, 3),
            usage=usage,
        )


class HybridEmotionAnalyzer:
    def __init__(self) -> None:
        self.transcriber = GroqWhisperTranscriber()
        self.classifier = NemotronEmotionClassifier()

    def analyze(self, name: str, audio_bytes: bytes, mime_type: str) -> HybridEmotionResult:
        transcript = self.transcriber.transcribe(name, audio_bytes, mime_type)
        classification = self.classifier.classify(transcript.text)
        return HybridEmotionResult(
            name=name,
            transcript=transcript,
            classification=classification,
        )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
