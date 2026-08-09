# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "python-dotenv>=1.2.2",
#   "websockets>=15.0",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from dotenv import load_dotenv
import websockets


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
CHANNELS = 1
CHUNK_MS = 100
CHUNK_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * CHUNK_MS // 1000
TRAILING_SILENCE_MS = 900

AGENT_PHRASES = (
    "i'm erica",
    "i am erica",
    "how can i help",
    "toyota of",
    "lexington toyota",
    "what type of service",
    "just to confirm",
    "dealership is closed",
    "did you want to transfer",
    "transferring you",
    "i can help with that",
)

WATCHLIST = (
    "Anger",
    "Annoyance",
    "Distress",
    "Disappointment",
    "Contempt",
    "Satisfaction",
    "Joy",
    "Contentment",
    "Relief",
    "Calmness",
    "Interest",
    "Excitement",
    "Sadness",
    "Surprise (positive)",
    "Surprise (negative)",
)


def _session_settings_payload() -> dict[str, Any]:
    return {
        "type": "session_settings",
        "audio": {
            "encoding": "linear16",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
        },
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _speaker_text(words: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for word in words:
        speaker = str(word.get("speaker_id") or "").strip()
        text = str(word.get("text") or "").strip()
        if speaker and text:
            grouped.setdefault(speaker, []).append(text)
    return {speaker: " ".join(parts) for speaker, parts in grouped.items()}


def _infer_customer_speaker(words: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    texts = _speaker_text(words)
    if len(texts) < 2:
        raise RuntimeError("Need at least two diarized speakers to isolate the customer.")

    phrase_hits: dict[str, list[str]] = {}
    scores: dict[str, int] = {}
    for speaker, text in texts.items():
        lowered = text.casefold()
        hits = [phrase for phrase in AGENT_PHRASES if phrase in lowered]
        phrase_hits[speaker] = hits
        scores[speaker] = len(hits)

    agent = max(scores, key=scores.get)
    if scores[agent] <= 0:
        raise RuntimeError("Could not infer dealership-agent speaker from role phrases.")

    others = [speaker for speaker in texts if speaker != agent]
    if len(others) != 1:
        raise RuntimeError(f"Expected one non-agent speaker; found {others}.")

    customer = others[0]
    return customer, {
        "agent_speaker": agent,
        "customer_speaker": customer,
        "agent_phrase_scores": scores,
        "agent_phrase_hits": phrase_hits,
        "speaker_text": texts,
    }


def _customer_segments(
    words: list[dict[str, Any]], customer_speaker: str
) -> list[dict[str, Any]]:
    timed: list[tuple[float, float, str]] = []
    for word in words:
        if str(word.get("speaker_id") or "") != customer_speaker:
            continue
        start = word.get("start")
        end = word.get("end")
        text = str(word.get("text") or "").strip()
        if start is None or end is None or not text:
            continue
        timed.append((float(start), float(end), text))

    timed.sort(key=lambda item: item[0])
    if not timed:
        raise RuntimeError("No timed customer words were available.")

    segments: list[dict[str, Any]] = []
    current_start, current_end, first_text = timed[0]
    current_words = [first_text]

    for start, end, text in timed[1:]:
        gap = start - current_end
        projected_duration = end - current_start
        if gap <= 0.8 and projected_duration <= 9.0:
            current_end = max(current_end, end)
            current_words.append(text)
        else:
            segments.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(current_words),
                }
            )
            current_start, current_end, current_words = start, end, [text]

    segments.append(
        {
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_words),
        }
    )
    return segments


def _extract_pcm(audio_path: Path, start: float, end: float) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required only for audio format conversion and is not installed. "
            "Do not install any ML packages as a fallback."
        )

    padded_start = max(0.0, start - 0.08)
    duration = max(0.1, (end - start) + 0.16)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{padded_start:.3f}",
        "-i",
        str(audio_path),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        error = result.stderr.decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"ffmpeg audio extraction failed: {error}")
    return result.stdout


def _scores_from_user_message(payload: dict[str, Any]) -> dict[str, float]:
    models = payload.get("models")
    if not isinstance(models, dict):
        return {}
    prosody = models.get("prosody")
    if not isinstance(prosody, dict):
        return {}
    raw_scores = prosody.get("scores")
    if not isinstance(raw_scores, dict):
        return {}

    scores: dict[str, float] = {}
    for name, value in raw_scores.items():
        try:
            scores[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


def _message_text(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "").strip()
    return ""


async def _wait_for_user_message(ws: Any, timeout_seconds: float) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RuntimeError("Timed out waiting for final Hume EVI user_message.")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        msg_type = payload.get("type")
        if msg_type == "error":
            raise RuntimeError(
                "Hume EVI error "
                f"{payload.get('code') or ''}: {payload.get('message') or payload}"
            )
        if msg_type == "user_message" and not payload.get("interim"):
            return payload


async def _stream_segment(
    ws: Any,
    pcm: bytes,
    *,
    realtime_factor: float,
    response_timeout_seconds: float,
) -> dict[str, Any]:
    sleep_seconds = (CHUNK_MS / 1000.0) * realtime_factor
    for offset in range(0, len(pcm), CHUNK_BYTES):
        chunk = pcm[offset : offset + CHUNK_BYTES]
        await ws.send(
            json.dumps(
                {
                    "type": "audio_input",
                    "data": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)

    silence_bytes = (
        SAMPLE_RATE
        * BYTES_PER_SAMPLE
        * CHANNELS
        * TRAILING_SILENCE_MS
        // 1000
    )
    silence = b"\x00" * silence_bytes
    for offset in range(0, len(silence), CHUNK_BYTES):
        chunk = silence[offset : offset + CHUNK_BYTES]
        await ws.send(
            json.dumps(
                {
                    "type": "audio_input",
                    "data": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)

    return await _wait_for_user_message(ws, response_timeout_seconds)


async def _analyze_call(
    *,
    api_key: str,
    config_id: str | None,
    audio_path: Path,
    words: list[dict[str, Any]],
    realtime_factor: float,
    response_timeout_seconds: float,
) -> dict[str, Any]:
    customer_speaker, role_diagnostics = _infer_customer_speaker(words)
    segments = _customer_segments(words, customer_speaker)

    query = {"api_key": api_key, "verbose_transcription": "false"}
    if config_id:
        query["config_id"] = config_id
    url = "wss://api.hume.ai/v0/evi/chat?" + urlencode(query)

    segment_results: list[dict[str, Any]] = []
    aggregate_weighted: defaultdict[str, float] = defaultdict(float)
    total_weight = 0.0

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps(_session_settings_payload()))
        # EVI keeps listening/transcribing/prosody-scoring while assistant responses
        # are paused, avoiding irrelevant generated speech during this diagnostic.
        await ws.send(json.dumps({"type": "pause_assistant_message"}))

        for segment in segments:
            start = float(segment["start"])
            end = float(segment["end"])
            duration = max(0.0, end - start)
            if duration < 0.35:
                continue

            pcm = _extract_pcm(audio_path, start, end)
            message = await _stream_segment(
                ws,
                pcm,
                realtime_factor=realtime_factor,
                response_timeout_seconds=response_timeout_seconds,
            )
            scores = _scores_from_user_message(message)
            if not scores:
                raise RuntimeError(
                    f"{audio_path.name}: Hume user_message contained no prosody scores."
                )

            for name, score in scores.items():
                aggregate_weighted[name] += score * duration
            total_weight += duration

            segment_results.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(duration, 3),
                    "scribe_text": segment["text"],
                    "hume_transcript": _message_text(message),
                    "top_expressions": sorted(
                        scores.items(), key=lambda item: item[1], reverse=True
                    )[:15],
                    "watchlist_scores": {
                        label: scores[label] for label in WATCHLIST if label in scores
                    },
                    "all_expression_scores": scores,
                }
            )

    if not segment_results or total_weight <= 0:
        raise RuntimeError(f"{audio_path.name}: no customer prosody results were produced.")

    aggregate = {
        name: score / total_weight for name, score in aggregate_weighted.items()
    }
    return {
        "name": audio_path.name,
        "customer_speaker": customer_speaker,
        "role_diagnostics": role_diagnostics,
        "customer_speech_duration_seconds": round(total_weight, 3),
        "segments": segment_results,
        "aggregate_top_expressions": sorted(
            aggregate.items(), key=lambda item: item[1], reverse=True
        )[:20],
        "aggregate_watchlist_scores": {
            label: aggregate[label] for label in WATCHLIST if label in aggregate
        },
        "all_aggregate_expression_scores": aggregate,
    }


def _load_expected(path: Path) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("name") or "").strip()
            raw = (row.get("result_json") or "").strip()
            if not name or not raw:
                continue
            payload = json.loads(raw)
            expected[name] = {
                "emotional_tone": payload.get("emotional_tone"),
                "emotional_intensity": payload.get("emotional_intensity"),
            }
    return expected


async def main_async(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        raise RuntimeError("HUME_API_KEY is not configured in .env.")
    config_id = os.getenv("HUME_EVI_CONFIG_ID") or None

    if not args.scribe_artifact.is_file():
        raise RuntimeError(
            "Scribe artifact is missing. Re-run evaluate_scribe_diagnostics.py after "
            "pulling the version that persists spoken_words."
        )
    scribe_rows = _load_json(args.scribe_artifact)
    if not isinstance(scribe_rows, list):
        raise RuntimeError("Unexpected Scribe artifact shape.")

    rows_by_name = {
        str(row.get("name") or ""): row
        for row in scribe_rows
        if isinstance(row, dict) and not row.get("error")
    }

    audio_paths = sorted(args.input_dir.glob("*.ogg"))
    if not audio_paths:
        raise RuntimeError(f"No OGG files found in {args.input_dir}")

    blind_rows: list[dict[str, Any]] = []
    for audio_path in audio_paths:
        row = rows_by_name.get(audio_path.name)
        if not row:
            raise RuntimeError(f"No Scribe diagnostics found for {audio_path.name}.")
        words = row.get("spoken_words")
        if not isinstance(words, list) or not words:
            raise RuntimeError(
                f"{audio_path.name}: Scribe artifact has no spoken_words. Re-run Scribe."
            )
        result = await _analyze_call(
            api_key=api_key,
            config_id=config_id,
            audio_path=audio_path,
            words=words,
            realtime_factor=args.realtime_factor,
            response_timeout_seconds=args.response_timeout,
        )
        blind_rows.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = args.output_dir / "hume_evi_scores_before_labels.json"
    blind_path.write_text(json.dumps(blind_rows, indent=2) + "\n", encoding="utf-8")

    # Development-set comparison begins only after all Hume EVI results are persisted.
    expected = _load_expected(args.labels)
    comparison = {
        "purpose": (
            "Development-set diagnostic only. Raw Hume EVI customer prosody scores are "
            "collected without AutoAce labels or tuned mappings."
        ),
        "comparisons": [
            {
                "name": row["name"],
                "expected": expected.get(row["name"]),
                "aggregate_top_expressions": row["aggregate_top_expressions"],
                "aggregate_watchlist_scores": row["aggregate_watchlist_scores"],
            }
            for row in blind_rows
        ],
    }
    comparison_path = args.output_dir / "comparison_after_labels.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(comparison, indent=2))
    print(f"\nRaw EVI scores: {blind_path}")
    print(f"Post-label comparison: {comparison_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only customer prosody test using Hume's active EVI WebSocket. "
            "No local ML model is installed or executed."
        )
    )
    parser.add_argument("input_dir", type=Path, nargs="?", default=Path("."))
    parser.add_argument(
        "--scribe-artifact",
        type=Path,
        default=Path("artifacts/scribe_diagnostics/scribe_diagnostics_before_labels.json"),
    )
    parser.add_argument("--labels", type=Path, default=Path("labels.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/hume_evi_prosody")
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="1.0 streams PCM at real-time speed; keep at 1.0 for the diagnostic.",
    )
    parser.add_argument("--response-timeout", type=float, default=15.0)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
