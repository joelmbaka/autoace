from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from .schemas import AnalysisResult, AudioAnalysis, UsageMetadata


MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low")
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(20 * 1024 * 1024)))
MAX_BATCH_UNCOMPRESSED_BYTES = int(
    os.getenv("MAX_BATCH_UNCOMPRESSED_BYTES", str(100 * 1024 * 1024))
)

SUPPORTED_AUDIO_TYPES: dict[str, str] = {
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}

SCORED_FIELDS = (
    "emotional_tone",
    "emotional_intensity",
    "background_noise_present",
    "background_noise_type",
    "background_noise_severity",
    "audio_quality",
    "speaker_overlap_present",
    "long_silence_present",
)

ANALYSIS_PROMPT = """You are evaluating a real automotive dealership service-call recording for AutoAce.
Analyze the complete raw audio and return exactly the requested schema. Treat each output dimension independently.

CUSTOMER ROLE
- emotional_tone and emotional_intensity describe the CUSTOMER, not the dealership representative or AI agent.
- Infer the customer from the dialogue and conversational role; do not assume a fixed channel or first speaker.

EMOTIONAL TONE
- neutral: no clear positive or negative emotion.
- satisfied: pleased, relieved, appreciative, or clearly positive.
- frustrated: annoyed, impatient, or dissatisfied without strong anger or distress.
- upset: clearly angry, agitated, or strongly dissatisfied.
- distressed: highly emotional, overwhelmed, panicked, crying, or otherwise emotionally escalated.
Do not infer frustration, upset, or distress from loudness alone. Use semantic content, prosody, pacing, and the overall interaction.

EMOTIONAL INTENSITY
- low: subtle or mild.
- medium: clear and sustained.
- high: strong, escalated, or likely to require attention.

BACKGROUND NOISE
- background_noise_present is true only when meaningful NON-SPEECH sound is audible in the background.
- Barely perceptible codec/telephone artifacts should not automatically count.
- Another conversational speaker is not background noise.
- background_noise_type is a short description of the dominant noise. It MUST be an empty string when no meaningful background noise is present.
- severity none: no meaningful noise.
- severity low: audible but does not interfere.
- severity medium: occasionally interferes with understanding.
- severity high: materially impairs the conversation or analysis.

AUDIO QUALITY
- Judge overall technical quality independently of emotion and independently of whether background noise exists.
- Consider distortion, clipping, echo, static, low volume, muffled speech, robotic audio, and packet loss.
- A call can contain audible background noise and still be clear if speech remains technically intelligible.

SPEAKER OVERLAP
- true only when two or more speakers talk at the same time enough to affect understanding or analysis.
- Do not mark true merely because the call has multiple speakers, short acknowledgements, or normal turn-taking.

LONG SILENCE
- true only for an unusually long silence/dead-air period that may indicate a call-flow or audio problem.
- Ordinary conversational pauses, thinking pauses, or brief hold-like gaps are false.

CONFIDENCE
- Give a calibrated 0.0-1.0 confidence for the overall result. Do not use a high confidence score when several dimensions are ambiguous.

Consistency requirements:
- If background_noise_present is false, background_noise_type must be "" and background_noise_severity must be "none".
- If background_noise_present is true, background_noise_severity must not be "none".
Return only data matching the supplied structured schema."""


@dataclass(slots=True)
class BatchItem:
    name: str
    data: bytes | None
    mime_type: str | None
    expected: AudioAnalysis | None = None
    preflight_error: str | None = None


@dataclass(slots=True)
class PreparedBatch:
    items: list[BatchItem]
    warnings: list[str]
    labeled_count: int


def _basename(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).name


def _decode_csv(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV manifest must be UTF-8 encoded.") from exc


def _read_manifest(data: bytes) -> tuple[list[str], dict[str, AudioAnalysis | None], list[str]]:
    reader = csv.DictReader(io.StringIO(_decode_csv(data)))
    if not reader.fieldnames:
        raise ValueError("CSV manifest has no header row.")

    required = {"name", "result_json"}
    missing_columns = required.difference(reader.fieldnames)
    if missing_columns:
        raise ValueError(
            "CSV manifest must contain name and result_json columns. Missing: "
            + ", ".join(sorted(missing_columns))
        )

    ordered_names: list[str] = []
    expected_by_name: dict[str, AudioAnalysis | None] = {}
    warnings: list[str] = []

    for row_number, row in enumerate(reader, start=2):
        name = (row.get("name") or "").strip()
        if not name:
            raise ValueError(f"CSV row {row_number} has an empty name.")
        if name in expected_by_name:
            raise ValueError(f"CSV manifest contains duplicate filename: {name}")

        expected: AudioAnalysis | None = None
        raw_result = (row.get("result_json") or "").strip()
        if raw_result:
            try:
                expected = AudioAnalysis.model_validate(json.loads(raw_result))
            except (json.JSONDecodeError, ValidationError) as exc:
                warnings.append(
                    f"Ground-truth JSON for {name} is invalid and will not be used for validation: {exc}"
                )

        ordered_names.append(name)
        expected_by_name[name] = expected

    if not ordered_names:
        raise ValueError("CSV manifest contains no data rows.")

    return ordered_names, expected_by_name, warnings


def _files_from_zip(data: bytes) -> list[tuple[str, bytes]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("Uploaded ZIP archive is malformed or unsupported.") from exc

    files: list[tuple[str, bytes]] = []
    seen_basenames: set[str] = set()
    total_uncompressed = 0

    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            normalized = info.filename.replace("\\", "/")
            if normalized.startswith("__MACOSX/"):
                continue
            name = _basename(normalized)
            if not name or name.startswith("."):
                continue
            if name in seen_basenames:
                raise ValueError(
                    f"ZIP contains duplicate basename {name}; filenames must be unique."
                )

            total_uncompressed += info.file_size
            if total_uncompressed > MAX_BATCH_UNCOMPRESSED_BYTES:
                raise ValueError("ZIP exceeds the configured uncompressed batch-size limit.")

            seen_basenames.add(name)
            files.append((name, archive.read(info)))

    return files


def prepare_batch(uploaded_files: list[tuple[str, bytes]]) -> PreparedBatch:
    if not uploaded_files:
        raise ValueError("No files were uploaded.")

    if len(uploaded_files) == 1 and uploaded_files[0][0].lower().endswith(".zip"):
        files = _files_from_zip(uploaded_files[0][1])
    elif any(name.lower().endswith(".zip") for name, _ in uploaded_files):
        raise ValueError("Upload either one ZIP archive or a folder/file selection, not both.")
    else:
        files = [(_basename(name), data) for name, data in uploaded_files]

    csv_files = [(name, data) for name, data in files if name.lower().endswith(".csv")]
    if len(csv_files) != 1:
        raise ValueError(
            f"Batch must contain exactly one CSV manifest; found {len(csv_files)}."
        )

    manifest_name, manifest_bytes = csv_files[0]
    ordered_names, expected_by_name, warnings = _read_manifest(manifest_bytes)

    audio_by_name: dict[str, bytes] = {}
    for name, data in files:
        if name == manifest_name:
            continue
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_TYPES:
            continue
        if name in audio_by_name:
            raise ValueError(f"Batch contains duplicate audio filename: {name}")
        audio_by_name[name] = data

    items: list[BatchItem] = []
    for name in ordered_names:
        expected = expected_by_name[name]
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_TYPES:
            items.append(
                BatchItem(
                    name=name,
                    data=None,
                    mime_type=None,
                    expected=expected,
                    preflight_error=f"Unsupported audio extension: {suffix or '(none)'}",
                )
            )
            continue

        data = audio_by_name.get(name)
        if data is None:
            items.append(
                BatchItem(
                    name=name,
                    data=None,
                    mime_type=SUPPORTED_AUDIO_TYPES[suffix],
                    expected=expected,
                    preflight_error="Audio file referenced by manifest is missing from the batch.",
                )
            )
            continue

        if len(data) > MAX_AUDIO_BYTES:
            items.append(
                BatchItem(
                    name=name,
                    data=None,
                    mime_type=SUPPORTED_AUDIO_TYPES[suffix],
                    expected=expected,
                    preflight_error=(
                        f"Audio file exceeds configured {MAX_AUDIO_BYTES // (1024 * 1024)} MB limit."
                    ),
                )
            )
            continue

        items.append(
            BatchItem(
                name=name,
                data=data,
                mime_type=SUPPORTED_AUDIO_TYPES[suffix],
                expected=expected,
            )
        )

    manifest_names = set(ordered_names)
    for extra_name in sorted(set(audio_by_name).difference(manifest_names)):
        warnings.append(
            f"Unmatched audio file {extra_name} is not listed in the manifest and was ignored."
        )

    labeled_count = sum(item.expected is not None for item in items)
    return PreparedBatch(items=items, warnings=warnings, labeled_count=labeled_count)


class GeminiAnalyzer:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=api_key)
        self.model = MODEL

    def _generate(self, audio_bytes: bytes, mime_type: str) -> tuple[AudioAnalysis, UsageMetadata]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                ANALYSIS_PROMPT,
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AudioAnalysis,
                thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
            ),
        )

        if not response.text:
            raise ValueError("Gemini returned no structured response text.")
        result = AudioAnalysis.model_validate_json(response.text)

        usage = response.usage_metadata
        usage_result = UsageMetadata(
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            total_tokens=getattr(usage, "total_token_count", None) if usage else None,
            thinking_tokens=getattr(usage, "thoughts_token_count", None) if usage else None,
        )
        return result, usage_result

    async def analyze(self, name: str, audio_bytes: bytes, mime_type: str) -> AnalysisResult:
        started = time.perf_counter()
        try:
            result, usage = await asyncio.to_thread(
                self._generate, audio_bytes, mime_type
            )
        except errors.APIError as exc:
            status = getattr(exc, "code", None)
            message = getattr(exc, "message", str(exc))
            if status in (401, 403):
                category = "Gemini authentication failure"
            elif status == 429:
                category = "Gemini quota/rate-limit failure"
            else:
                category = "Gemini API failure"
            raise RuntimeError(f"{category} ({status or 'unknown'}): {message}") from exc
        except ValidationError as exc:
            raise RuntimeError(f"Structured-output validation failure: {exc}") from exc

        return AnalysisResult(
            name=name,
            result=result,
            model=self.model,
            request_seconds=round(time.perf_counter() - started, 3),
            usage=usage,
        )


def compare_prediction(predicted: AudioAnalysis, expected: AudioAnalysis) -> dict[str, Any]:
    matches: dict[str, bool] = {}
    for field in SCORED_FIELDS:
        predicted_value = getattr(predicted, field)
        expected_value = getattr(expected, field)
        if field == "background_noise_type":
            matches[field] = str(predicted_value).strip().casefold() == str(expected_value).strip().casefold()
        else:
            matches[field] = predicted_value == expected_value

    correct = sum(matches.values())
    return {
        "field_matches": matches,
        "correct_fields": correct,
        "total_fields": len(SCORED_FIELDS),
        "field_accuracy": correct / len(SCORED_FIELDS),
        "expected": expected.model_dump(),
    }


def build_validation_summary(comparisons: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not comparisons:
        return None

    per_field: dict[str, float] = {}
    for field in SCORED_FIELDS:
        per_field[field] = sum(
            1 for comparison in comparisons if comparison["field_matches"][field]
        ) / len(comparisons)

    total_correct = sum(comparison["correct_fields"] for comparison in comparisons)
    total_fields = len(comparisons) * len(SCORED_FIELDS)

    tone_labels = ["neutral", "satisfied", "frustrated", "upset", "distressed"]
    confusion = {expected: {predicted: 0 for predicted in tone_labels} for expected in tone_labels}
    for comparison in comparisons:
        expected_tone = comparison["expected"]["emotional_tone"]
        # The predicted tone is derivable from whether each field matched only if we retain it;
        # comparison callers attach it before summary construction.
        predicted_tone = comparison.get("predicted_tone")
        if predicted_tone in tone_labels:
            confusion[expected_tone][predicted_tone] += 1

    return {
        "labeled_files": len(comparisons),
        "exact_field_accuracy": total_correct / total_fields,
        "per_field_accuracy": per_field,
        "emotional_tone_confusion_matrix": confusion,
        "confidence_note": "Model confidence is reported but is not treated as a ground-truth label match.",
    }
