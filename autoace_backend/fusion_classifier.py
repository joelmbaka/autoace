from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from .hybrid_emotion import GroqWhisperTranscriber, TranscriptResult
from .schemas import AnalysisResult, AudioAnalysis, UsageMetadata


DEFAULT_FUSION_MODEL = "gemini-3.5-flash-lite"
DEFAULT_FUSION_GROQ_MODEL = "whisper-large-v3-turbo"


FUSION_PROMPT = """You are evaluating a real automotive dealership service-call recording for AutoAce.
You receive BOTH the original raw audio and an ASR transcript. Use the transcript only as semantic/speaker-role assistance; the raw audio remains authoritative for vocal affect and acoustic conditions. The ASR transcript may contain errors, especially around accents, cross-talk, or non-English speech.

Return exactly the supplied structured schema. Treat every dimension independently.

CUSTOMER ROLE
- Emotional tone and intensity describe the CUSTOMER, not the dealership representative or AI agent.
- Infer the customer from conversational role and the entire interaction; do not assume a fixed channel or first speaker.

EMOTIONAL TONE
- neutral: ordinary transactional conversation with no clear positive or negative affect.
- satisfied: pleased, appreciative, relieved, cooperative with clearly positive affect, or content with the interaction/outcome.
- frustrated: annoyed, impatient, or dissatisfied, but not strongly angry or substantially escalated.
- upset: clearly angry, agitated, strongly dissatisfied, or repeatedly escalating because the interaction is failing.
- distressed: overwhelmed, panicked, crying, fearful, or otherwise highly emotionally escalated.
Use BOTH vocal prosody and semantic context. Do not classify the customer as frustrated merely because the requested appointment/service cannot be fulfilled. Conversely, repeated impatient attempts, strong agitation, or a communication breakdown can indicate upset even if the transcript words are sparse.

EMOTIONAL INTENSITY
- low: subtle or mild emotion.
- medium: clearly expressed and sustained emotion.
- high: strong, repeated, escalated, or likely to require attention.
Intensity is independent of tone; do not default to medium.

BACKGROUND NOISE
- background_noise_present=true only for meaningful NON-SPEECH sound audible behind/around the call.
- Another conversational speaker is not background noise.
- Ordinary codec/telephone artifacts alone should not count.
- background_noise_type should be a concise source/character description such as TV, traffic, background chatter, music, wind, static, or sharp static when that is genuinely audible.
- If no meaningful background noise is present, type must be an empty string and severity must be none.
- low: audible but not interfering; medium: sometimes interferes; high: materially impairs the conversation/analysis.

AUDIO QUALITY
- Judge technical intelligibility independently from background noise.
- clear is valid even when some background noise is present if speech remains technically intelligible.
- slightly_impaired means technical degradation is noticeable and sometimes affects understanding.
- severely_impaired means degradation materially prevents understanding.

SPEAKER OVERLAP
- true only when two or more speakers audibly speak at the same time enough to matter.
- Do not mark true for sequential turns or merely because multiple speakers are present.

LONG SILENCE
- true only for unusually prolonged dead air that would be operationally meaningful in a phone call.
- Ordinary pauses, thinking pauses, and short gaps are false.

CONFIDENCE
- Return calibrated overall confidence from 0 to 1. Reduce confidence when several fields are ambiguous.

Consistency:
- background_noise_present=false => background_noise_type="" and background_noise_severity="none".
- background_noise_present=true => background_noise_severity cannot be "none".

ASR TRANSCRIPT (semantic aid only):
{transcript}
"""


@dataclass(slots=True)
class FusionResult:
    name: str
    transcript: TranscriptResult
    analysis: AnalysisResult


class FusionAnalyzer:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        groq_model = os.getenv("FUSION_GROQ_MODEL", DEFAULT_FUSION_GROQ_MODEL)
        self.transcriber = GroqWhisperTranscriber()
        self.transcriber.model = groq_model
        self.client = genai.Client(api_key=api_key)
        self.model = os.getenv("FUSION_GEMINI_MODEL", DEFAULT_FUSION_MODEL)

    def _classify(
        self,
        transcript: str,
        audio_bytes: bytes,
        mime_type: str,
    ) -> tuple[AudioAnalysis, UsageMetadata, float]:
        prompt = FUSION_PROMPT.format(transcript=transcript.strip())
        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AudioAnalysis,
                ),
            )
        except errors.APIError as exc:
            status = getattr(exc, "code", None)
            message = getattr(exc, "message", str(exc))
            raise RuntimeError(
                f"Gemini fusion classification failed ({status or 'unknown'}): {message}"
            ) from exc

        request_seconds = round(time.perf_counter() - started, 3)
        if not response.text:
            raise RuntimeError("Gemini fusion classifier returned no structured response.")
        try:
            result = AudioAnalysis.model_validate_json(response.text)
        except ValidationError as exc:
            raise RuntimeError(f"Gemini fusion structured output failed validation: {exc}") from exc

        usage = response.usage_metadata
        usage_result = UsageMetadata(
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            total_tokens=getattr(usage, "total_token_count", None) if usage else None,
            thinking_tokens=getattr(usage, "thoughts_token_count", None) if usage else None,
        )
        return result, usage_result, request_seconds

    async def analyze(
        self,
        name: str,
        audio_bytes: bytes,
        mime_type: str,
    ) -> FusionResult:
        transcript = await asyncio.to_thread(
            self.transcriber.transcribe,
            name,
            audio_bytes,
            mime_type,
        )
        result, usage, request_seconds = await asyncio.to_thread(
            self._classify,
            transcript.text,
            audio_bytes,
            mime_type,
        )
        return FusionResult(
            name=name,
            transcript=transcript,
            analysis=AnalysisResult(
                name=name,
                result=result,
                model=self.model,
                request_seconds=request_seconds,
                usage=usage,
            ),
        )
