from __future__ import annotations

import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field


SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_SCRIBE_MODEL = "scribe_v2"


class ScribeToken(BaseModel):
    text: str = ""
    type: str = ""
    start: float | None = None
    end: float | None = None
    speaker_id: str | None = None
    logprob: float | None = None


class OverlapInterval(BaseModel):
    start: float
    end: float
    duration: float
    speaker_a: str
    speaker_b: str
    text_a: str
    text_b: str


class ScribeDiagnosticsResult(BaseModel):
    name: str
    model: str
    request_seconds: float
    language_code: str | None = None
    language_probability: float | None = None
    transcript: str
    speaker_ids: list[str] = Field(default_factory=list)
    audio_events: list[ScribeToken] = Field(default_factory=list)
    overlap_intervals: list[OverlapInterval] = Field(default_factory=list)
    total_overlap_seconds: float = 0.0
    max_interword_gap_seconds: float | None = None
    raw_word_count: int = 0


class ElevenLabsScribeDiagnostics:
    """Diagnostic-only Scribe v2 client.

    This is intentionally separate from the production classifier. It is used to
    inspect specialist diarization, audio-event tags, overlap timing, and silence
    structure before deciding whether any of those signals should be reproduced
    with a production-cost-compliant approach.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not configured.")
        self.model = os.getenv("ELEVENLABS_SCRIBE_MODEL", DEFAULT_SCRIBE_MODEL)
        self.timeout = float(os.getenv("ELEVENLABS_TIMEOUT_SECONDS", "180"))
        self.overlap_min_seconds = float(os.getenv("SCRIBE_OVERLAP_MIN_SECONDS", "0.05"))

    def analyze(self, name: str, audio_bytes: bytes, mime_type: str) -> ScribeDiagnosticsResult:
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    SCRIBE_URL,
                    headers={"xi-api-key": self.api_key},
                    files={"file": (name, audio_bytes, mime_type)},
                    data={
                        "model_id": self.model,
                        "tag_audio_events": "true",
                        "diarize": "true",
                        "timestamps_granularity": "word",
                    },
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"ElevenLabs Scribe request failed: {exc}") from exc

        request_seconds = time.perf_counter() - started
        if response.is_error:
            raise RuntimeError(
                f"ElevenLabs Scribe failed ({response.status_code}): {response.text[:500]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("ElevenLabs returned a non-JSON Scribe response.") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("ElevenLabs returned an unexpected Scribe response shape.")

        tokens = _parse_tokens(payload.get("words"))
        spoken_words = [token for token in tokens if token.type == "word"]
        audio_events = [
            token for token in tokens if token.type not in {"word", "spacing"}
        ]
        speaker_ids = sorted(
            {token.speaker_id for token in spoken_words if token.speaker_id}
        )
        overlap_intervals = _find_overlap_intervals(
            spoken_words, min_seconds=self.overlap_min_seconds
        )
        max_gap = _max_interword_gap(spoken_words)

        return ScribeDiagnosticsResult(
            name=name,
            model=self.model,
            request_seconds=round(request_seconds, 3),
            language_code=_optional_str(payload.get("language_code")),
            language_probability=_optional_float(payload.get("language_probability")),
            transcript=str(payload.get("text") or "").strip(),
            speaker_ids=speaker_ids,
            audio_events=audio_events,
            overlap_intervals=overlap_intervals,
            total_overlap_seconds=round(
                sum(interval.duration for interval in overlap_intervals), 3
            ),
            max_interword_gap_seconds=round(max_gap, 3) if max_gap is not None else None,
            raw_word_count=len(spoken_words),
        )


def _parse_tokens(value: Any) -> list[ScribeToken]:
    if not isinstance(value, list):
        return []
    tokens: list[ScribeToken] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        tokens.append(
            ScribeToken(
                text=str(item.get("text") or ""),
                type=str(item.get("type") or ""),
                start=_optional_float(item.get("start")),
                end=_optional_float(item.get("end")),
                speaker_id=_optional_str(item.get("speaker_id")),
                logprob=_optional_float(item.get("logprob")),
            )
        )
    return tokens


def _find_overlap_intervals(
    words: list[ScribeToken], *, min_seconds: float
) -> list[OverlapInterval]:
    timed = [
        word
        for word in words
        if word.start is not None and word.end is not None and word.speaker_id
    ]
    overlaps: list[OverlapInterval] = []
    for index, first in enumerate(timed):
        assert first.start is not None and first.end is not None and first.speaker_id
        for second in timed[index + 1 :]:
            assert second.start is not None and second.end is not None and second.speaker_id
            if second.start >= first.end:
                break
            if second.speaker_id == first.speaker_id:
                continue
            start = max(first.start, second.start)
            end = min(first.end, second.end)
            duration = end - start
            if duration < min_seconds:
                continue
            overlaps.append(
                OverlapInterval(
                    start=round(start, 3),
                    end=round(end, 3),
                    duration=round(duration, 3),
                    speaker_a=first.speaker_id,
                    speaker_b=second.speaker_id,
                    text_a=first.text,
                    text_b=second.text,
                )
            )
    return overlaps


def _max_interword_gap(words: list[ScribeToken]) -> float | None:
    timed = sorted(
        (
            word
            for word in words
            if word.start is not None and word.end is not None
        ),
        key=lambda word: float(word.start or 0.0),
    )
    if len(timed) < 2:
        return None

    max_gap = 0.0
    previous_end = float(timed[0].end or 0.0)
    for word in timed[1:]:
        start = float(word.start or 0.0)
        gap = max(0.0, start - previous_end)
        max_gap = max(max_gap, gap)
        previous_end = max(previous_end, float(word.end or previous_end))
    return max_gap


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
