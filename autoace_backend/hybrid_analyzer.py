from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Literal

from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from .customer_segments import build_customer_segments, infer_customer_speaker
from .emotion_client import EmotionAggregate, EmotionServiceClient
from .schemas import AnalysisResult, AudioAnalysis, UsageMetadata
from .scribe_diagnostics import ElevenLabsScribeDiagnostics, ScribeDiagnosticsResult
from .services import GeminiAnalyzer, THINKING_LEVEL


UPSET_NEGATIVE_THRESHOLD = 0.35
SATISFIED_NEGATIVE_CEILING = 0.15
FRUSTRATED_NEGATIVE_THRESHOLD = 0.15
HIGH_INTENSITY_NEGATIVE_THRESHOLD = 0.30
SEMANTIC_HIGH_NEGATIVE_THRESHOLD = 0.20
PERSISTENT_NOISE_SECONDS = float(os.getenv("PERSISTENT_NOISE_SECONDS", "2"))
LONG_SILENCE_THRESHOLD_SECONDS = float(os.getenv("LONG_SILENCE_THRESHOLD_SECONDS", "12"))


class SemanticEvidence(BaseModel):
    request_resolved: bool
    positive_ending: bool
    complaint_or_blockage: bool
    repeated_failed_contact: bool
    distress_or_panic: bool
    strong_anger_or_agitation: bool
    semantic_intensity: Literal["low", "medium", "high"]


SEMANTIC_PROMPT = """Analyze the CUSTOMER in this automotive service call and return the requested structured evidence.
These are evidence fields, not the final AutoAce emotional label. Do not classify raw-audio emotion.
Do not infer emotion solely from politeness or words such as "thank you". Distinguish a genuinely
resolved/positive ending from ordinary neutral conversation. Use the full transcript for context but
judge the customer; the customer-only transcript is provided to make their role explicit.

Definitions and safeguards:
- complaint_or_blockage means the customer is emotionally dissatisfied with, or obstructed by, the
  current interaction/service. Ordinary scheduling requests or corrections are not complaints.
- Requesting another language is not frustration.
- Describing a product or vehicle detail that needs changing is not automatically dissatisfaction.
- request_resolved includes the customer and agent reaching and accepting a workable plan or next
  step; it does not require the requested service itself to have already been performed.
- positive_ending means a clearly cooperative, appreciative, or resolved close in context, including
  calm acceptance of the agreed next step; it is not a keyword match.
- Judge the customer's interaction state from context; profanity or any other token in isolation does
  not establish anger, agitation, frustration, or a complaint.

FULL TRANSCRIPT:
{full_transcript}

CUSTOMER TRANSCRIPT:
{customer_transcript}

CUSTOMER EMOTION2VEC AGGREGATE (supporting acoustic evidence only):
{emotion_json}
"""


@dataclass(slots=True)
class HybridDetails:
    analysis: AnalysisResult
    semantic_evidence: SemanticEvidence
    emotion: EmotionAggregate
    scribe: ScribeDiagnosticsResult
    customer_speaker: str
    customer_transcript: str
    speaker_diagnostics: dict
    provider_latency: dict[str, float]


def fuse_tone(negative_affect: float, evidence: SemanticEvidence) -> tuple[str, str]:
    """Fuse general acoustic and semantic evidence using named, tunable thresholds."""
    if evidence.distress_or_panic:
        tone = "distressed"
    elif negative_affect >= UPSET_NEGATIVE_THRESHOLD and (
        evidence.strong_anger_or_agitation
        or evidence.complaint_or_blockage
        or evidence.repeated_failed_contact
    ):
        tone = "upset"
    elif (
        evidence.request_resolved
        and evidence.positive_ending
        and negative_affect < SATISFIED_NEGATIVE_CEILING
    ):
        tone = "satisfied"
    elif negative_affect >= FRUSTRATED_NEGATIVE_THRESHOLD and evidence.complaint_or_blockage:
        tone = "frustrated"
    else:
        tone = "neutral"

    if tone == "distressed":
        intensity = "high"
    elif tone == "upset" and negative_affect >= HIGH_INTENSITY_NEGATIVE_THRESHOLD:
        intensity = "high"
    elif (
        evidence.semantic_intensity == "high"
        and negative_affect >= SEMANTIC_HIGH_NEGATIVE_THRESHOLD
    ):
        intensity = "high"
    else:
        intensity = "medium"
    return tone, intensity


_COOPERATIVE_ENDING_PATTERNS = (
    r"\bthank(?:s| you)\b",
    r"\bi appreciate (?:it|that|your help)\b",
    r"\b(?:okay|ok|yes|yep),? (?:that works|sounds good|please do)\b",
    r"\b(?:that works|sounds good)\b",
)
_ACCEPTED_ACTION_PATTERNS = (
    r"\btransferr?ing you\b",
    r"\btransfer(?:red)? to (?:an?|the)\b",
    r"\bappointment (?:is |has been )?(?:booked|confirmed|scheduled)\b",
    r"\b(?:booked|confirmed|scheduled) (?:the |your |an? )?appointment\b",
    r"\b(?:we(?:'ll| will)|i(?:'ll| will)) (?:book|schedule|transfer|arrange|send|call)\b",
    r"\b(?:agreed|accepted) (?:plan|next step|action|appointment|transfer)\b",
)


def stabilize_semantic_evidence(
    negative_affect: float,
    evidence: SemanticEvidence,
    *,
    transcript: str,
    customer_transcript: str,
) -> SemanticEvidence:
    """Repair a narrow resolved-positive contradiction using conversation evidence."""
    if (
        negative_affect >= 0.05
        or evidence.distress_or_panic
        or evidence.strong_anger_or_agitation
        or evidence.repeated_failed_contact
    ):
        return evidence

    customer_ending = customer_transcript.casefold().strip()[-240:]
    full_transcript = transcript.casefold()
    cooperative_ending = any(
        re.search(pattern, customer_ending) for pattern in _COOPERATIVE_ENDING_PATTERNS
    )
    accepted_action = any(
        re.search(pattern, full_transcript) for pattern in _ACCEPTED_ACTION_PATTERNS
    )
    if not (cooperative_ending and accepted_action):
        return evidence

    return evidence.model_copy(
        update={"request_resolved": True, "positive_ending": True}
    )


def _event_texts(scribe: ScribeDiagnosticsResult) -> list[str]:
    return [event.text.strip().casefold() for event in scribe.audio_events if event.text.strip()]


def _event_duration_seconds(scribe: ScribeDiagnosticsResult, marker: str) -> float:
    return sum(
        max(0.0, float(event.end) - float(event.start))
        for event in scribe.audio_events
        if marker in event.text.casefold() and event.start is not None and event.end is not None
    )


def _speaker_duration_seconds(scribe: ScribeDiagnosticsResult, speakers: set[str]) -> float:
    return sum(
        max(0.0, float(word.end) - float(word.start))
        for word in scribe.spoken_words
        if word.speaker_id in speakers and word.start is not None and word.end is not None
    )


_BROADCAST_NOISE_MARKERS = (
    "tv",
    "television",
    "background speech",
    "background talk",
    "radio speech",
    "broadcast speech",
)
_SUSTAINED_NOISE_MARKERS = ("continuous", "sustained", "constant", "ongoing")
_TECHNICAL_DEGRADATION_MARKERS = (
    "clipping",
    "clipped",
    "distortion",
    "distorted",
    "packet loss",
    "robotic",
    "muffled",
    "garbled",
    "unintelligible",
    "low volume",
)


def _contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(re.search(rf"\b{re.escape(marker)}\b", normalized) for marker in markers)


def _stabilize_audio_quality(
    baseline: AudioAnalysis,
    scribe: ScribeDiagnosticsResult,
    *,
    noise_severity: str,
) -> str:
    if baseline.audio_quality == "clear":
        return "clear"

    evidence_text = " ".join(
        [baseline.background_noise_type, *[event.text for event in scribe.audio_events]]
    )
    technical_degradation = _contains_marker(
        evidence_text, _TECHNICAL_DEGRADATION_MARKERS
    )
    usable_transcript = (
        bool(scribe.transcript.strip())
        and bool(scribe.spoken_words)
        and len(scribe.speaker_ids) >= 2
    )
    if usable_transcript and not technical_degradation and noise_severity != "high":
        return "clear"
    return baseline.audio_quality


def fuse_acoustic_fields(
    baseline: AudioAnalysis,
    scribe: ScribeDiagnosticsResult,
    *,
    agent_speaker: str,
    customer_speaker: str,
) -> dict:
    events = _event_texts(scribe)
    extra_speakers = set(scribe.speaker_ids) - {agent_speaker, customer_speaker}
    if any("static" in event for event in events):
        duration = _event_duration_seconds(scribe, "static")
        noise_present, noise_type = True, "sharp static"
        noise_severity = "medium" if duration >= PERSISTENT_NOISE_SECONDS else "low"
    elif extra_speakers:
        # A third diarized voice excluded from the two conversational roles is specialist
        # evidence of continuous background speech/broadcast audio.
        duration = _speaker_duration_seconds(scribe, extra_speakers)
        noise_present, noise_type = True, "TV"
        noise_severity = "medium" if duration >= PERSISTENT_NOISE_SECONDS else "low"
    elif baseline.background_noise_present and _contains_marker(
        baseline.background_noise_type, _BROADCAST_NOISE_MARKERS
    ):
        noise_present, noise_type = True, "TV"
        noise_severity = baseline.background_noise_severity
        if _contains_marker(
            baseline.background_noise_type, _SUSTAINED_NOISE_MARKERS
        ):
            noise_severity = "medium" if noise_severity == "low" else noise_severity
    elif not events and set(scribe.speaker_ids) == {agent_speaker, customer_speaker}:
        noise_present, noise_type, noise_severity = False, "", "none"
    else:
        noise_present = baseline.background_noise_present
        noise_type = baseline.background_noise_type if noise_present else ""
        noise_severity = baseline.background_noise_severity if noise_present else "none"

    audio_quality = _stabilize_audio_quality(
        baseline, scribe, noise_severity=noise_severity
    )
    if noise_present and audio_quality == "severely_impaired":
        noise_severity = "high"

    overlap = baseline.speaker_overlap_present or bool(scribe.overlap_intervals)
    long_silence = bool(
        scribe.max_interword_gap_seconds is not None
        and scribe.max_interword_gap_seconds >= LONG_SILENCE_THRESHOLD_SECONDS
    )
    return {
        "background_noise_present": noise_present,
        "background_noise_type": noise_type,
        "background_noise_severity": noise_severity,
        "audio_quality": audio_quality,
        "speaker_overlap_present": overlap,
        "long_silence_present": long_silence,
    }


class HybridAnalyzer:
    """Production accuracy-first pipeline; GeminiAnalyzer remains the raw-audio baseline."""

    def __init__(self) -> None:
        self.baseline = GeminiAnalyzer()
        self.scribe = ElevenLabsScribeDiagnostics()
        self.emotion = EmotionServiceClient()
        self.model = f"hybrid:{self.baseline.model}+scribe+emotion2vec"

    def _semantic(self, transcript: str, customer_transcript: str, emotion: EmotionAggregate) -> tuple[SemanticEvidence, UsageMetadata]:
        response = self.baseline.client.models.generate_content(
            model=self.baseline.model,
            contents=SEMANTIC_PROMPT.format(
                full_transcript=transcript,
                customer_transcript=customer_transcript,
                emotion_json=json.dumps({
                    "aggregate_scores": emotion.aggregate_scores,
                    "negative_affect": emotion.negative_affect,
                    "positive_affect": emotion.positive_affect,
                    "neutral_affect": emotion.neutral_affect,
                }, separators=(",", ":")),
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SemanticEvidence,
                thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned no semantic evidence.")
        evidence = SemanticEvidence.model_validate_json(response.text)
        raw_usage = response.usage_metadata
        return evidence, UsageMetadata(
            input_tokens=getattr(raw_usage, "prompt_token_count", None) if raw_usage else None,
            output_tokens=getattr(raw_usage, "candidates_token_count", None) if raw_usage else None,
            total_tokens=getattr(raw_usage, "total_token_count", None) if raw_usage else None,
            thinking_tokens=getattr(raw_usage, "thoughts_token_count", None) if raw_usage else None,
        )

    async def analyze_detailed(self, name: str, audio_bytes: bytes, mime_type: str) -> HybridDetails:
        started = time.perf_counter()
        try:
            scribe, baseline = await asyncio.gather(
                asyncio.to_thread(self.scribe.analyze, name, audio_bytes, mime_type),
                self.baseline.analyze(name, audio_bytes, mime_type),
            )
            customer_speaker, speaker_diagnostics = infer_customer_speaker(scribe.spoken_words)
            segments = build_customer_segments(scribe.spoken_words, customer_speaker)
            customer_transcript = " ".join(segment["text"] for segment in segments)
            emotion = await asyncio.to_thread(self.emotion.analyze, audio_bytes, segments)
            semantic_started = time.perf_counter()
            evidence, semantic_usage = await asyncio.to_thread(
                self._semantic, scribe.transcript, customer_transcript, emotion
            )
            semantic_seconds = time.perf_counter() - semantic_started
        except (errors.APIError, ValidationError) as exc:
            raise RuntimeError(f"Hybrid provider/structured-output failure: {exc}") from exc

        evidence = stabilize_semantic_evidence(
            emotion.negative_affect,
            evidence,
            transcript=scribe.transcript,
            customer_transcript=customer_transcript,
        )
        tone, intensity = fuse_tone(emotion.negative_affect, evidence)
        acoustics = fuse_acoustic_fields(
            baseline.result,
            scribe,
            agent_speaker=speaker_diagnostics["agent_speaker"],
            customer_speaker=customer_speaker,
        )
        result = AudioAnalysis(
            emotional_tone=tone,
            emotional_intensity=intensity,
            confidence=baseline.result.confidence,
            **acoustics,
        )
        analysis = AnalysisResult(
            name=name,
            result=result,
            model=self.model,
            request_seconds=round(time.perf_counter() - started, 3),
            usage=semantic_usage,
        )
        return HybridDetails(
            analysis=analysis,
            semantic_evidence=evidence,
            emotion=emotion,
            scribe=scribe,
            customer_speaker=customer_speaker,
            customer_transcript=customer_transcript,
            speaker_diagnostics=speaker_diagnostics,
            provider_latency={
                "scribe": scribe.request_seconds,
                "emotion": emotion.request_seconds,
                "gemini_raw_audio": baseline.request_seconds,
                "gemini_semantic": round(semantic_seconds, 3),
                "total": analysis.request_seconds,
            },
        )

    async def analyze(self, name: str, audio_bytes: bytes, mime_type: str) -> AnalysisResult:
        return (await self.analyze_detailed(name, audio_bytes, mime_type)).analysis
