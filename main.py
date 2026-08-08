import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError


MODEL = "gemini-3.6-flash"
ROOT = Path(__file__).resolve().parent
AUDIO_FILE = ROOT / "call_003.ogg"

PROMPT = """This is an automotive dealership service call.
Analyze the customer's emotional tone and emotional intensity, not the dealership representative/AI agent.
Infer speaker roles from the complete conversation.
Analyze the raw acoustic signal as well as spoken content.
Detect actual audible background noise and describe its type concisely.
Do not confuse another conversational speaker with background noise.
Speaker overlap means two or more speakers audibly speaking at the same time.
Long silence means an unusually prolonged silence/dead-air segment that would be operationally meaningful in a phone call.
Audio quality should measure whether technical/audio degradation interferes with understanding.
Confidence should represent confidence in the entire classification.
Return only data matching the provided schema."""


class AudioAnalysis(BaseModel):
    emotional_tone: Literal[
        "neutral", "satisfied", "frustrated", "upset", "distressed"
    ]
    emotional_intensity: Literal["low", "medium", "high"]
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: Literal["none", "low", "medium", "high"]
    audio_quality: Literal["clear", "slightly_impaired", "severely_impaired"]
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0, le=1)


def explain_api_error(exc: errors.APIError) -> str:
    status = getattr(exc, "code", None)
    if status in (401, 403):
        category = "Authentication failure"
    elif status == 429:
        category = "Quota/rate-limit failure"
    elif status in (400, 404):
        category = "Unsupported audio/API error"
    else:
        category = "Gemini API error"
    return f"{category} ({status or 'unknown status'}): {exc.message}"


def main() -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Missing API key: GEMINI_API_KEY is not set in the root .env.", file=sys.stderr)
        return 2

    if not AUDIO_FILE.is_file():
        print(f"Missing audio file: {AUDIO_FILE.name} was not found.", file=sys.stderr)
        return 2

    try:
        audio_bytes = AUDIO_FILE.read_bytes()
    except OSError as exc:
        print(f"Unsupported audio/API error: could not read {AUDIO_FILE.name}: {exc}", file=sys.stderr)
        return 2

    try:
        client = genai.Client(api_key=api_key)
        started_at = time.perf_counter()
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                PROMPT,
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AudioAnalysis,
            ),
        )
        request_seconds = time.perf_counter() - started_at
    except errors.APIError as exc:
        request_seconds = time.perf_counter() - started_at
        print(f"Model: {MODEL}")
        print(f"File: {AUDIO_FILE.name}")
        print(f"Request time: {request_seconds:.2f} seconds")
        print(explain_api_error(exc), file=sys.stderr)
        return 3
    except (ValueError, TypeError, OSError) as exc:
        print(f"Unsupported audio/API error: {exc}", file=sys.stderr)
        return 3

    try:
        if not response.text:
            raise ValueError("Gemini returned no structured response text.")
        result = AudioAnalysis.model_validate_json(response.text)
    except (ValidationError, ValueError) as exc:
        print(f"Structured-output validation failure: {exc}", file=sys.stderr)
        return 4

    print(f"Model: {MODEL}")
    print(f"File: {AUDIO_FILE.name}")
    print(f"Request time: {request_seconds:.2f} seconds")
    if response.usage_metadata:
        usage = response.usage_metadata
        print(
            "Token usage: "
            f"input={usage.prompt_token_count}, "
            f"output={usage.candidates_token_count}, "
            f"total={usage.total_token_count}"
        )
    print()
    print(json.dumps(result.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
