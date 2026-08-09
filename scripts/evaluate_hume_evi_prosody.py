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
import contextlib
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from array import array
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
TRAILING_SILENCE_MS = 1800

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


def _evi_query(api_key: str, config_id: str | None = None) -> dict[str, str]:
    query = {"api_key": api_key, "verbose_transcription": "true"}
    if config_id:
        query["config_id"] = config_id
    return query


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


def _speaker_turns(
    words: list[dict[str, Any]], speaker_id: str
) -> list[dict[str, Any]]:
    timed: list[tuple[float, float, str]] = []
    for word in words:
        if str(word.get("speaker_id") or "") != speaker_id:
            continue
        start = word.get("start")
        end = word.get("end")
        text = str(word.get("text") or "").strip()
        if start is None or end is None or not text:
            continue
        timed.append((float(start), float(end), text))
    timed.sort(key=lambda item: item[0])

    turns: list[dict[str, Any]] = []
    for start, end, text in timed:
        if turns and start - float(turns[-1]["end"]) <= 0.8:
            turns[-1]["end"] = max(float(turns[-1]["end"]), end)
            turns[-1]["text"] += f" {text}"
        else:
            turns.append({"start": start, "end": end, "text": text})
    return turns


def _speaker_diagnostics(
    words: list[dict[str, Any]], speaker_id: str
) -> dict[str, Any]:
    timed_words = [
        word
        for word in words
        if str(word.get("speaker_id") or "") == speaker_id
        and word.get("start") is not None
        and word.get("end") is not None
        and str(word.get("text") or "").strip()
    ]
    timed_words.sort(key=lambda word: float(word["start"]))
    if not timed_words:
        raise RuntimeError(f"Speaker {speaker_id} has no timed spoken words.")
    return {
        "speaker_id": speaker_id,
        "first_word_start": float(timed_words[0]["start"]),
        "last_word_end": float(timed_words[-1]["end"]),
        "spoken_word_count": len(timed_words),
        "total_speech_duration": round(
            sum(
                max(0.0, float(word["end"]) - float(word["start"]))
                for word in timed_words
            ),
            3,
        ),
        "transcript": " ".join(str(word["text"]).strip() for word in timed_words),
    }


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

    agent_turns = _speaker_turns(words, agent)
    if not agent_turns:
        raise RuntimeError("Dealership-agent speaker has no timed greeting turn.")
    greeting = next(
        (
            turn
            for turn in agent_turns
            if any(phrase in str(turn["text"]).casefold() for phrase in AGENT_PHRASES)
        ),
        agent_turns[0],
    )
    greeting_end = float(greeting["end"])

    others = [speaker for speaker in texts if speaker != agent]
    candidates: list[dict[str, Any]] = []
    for speaker in others:
        diagnostics = _speaker_diagnostics(words, speaker)
        response_turns = [
            turn
            for turn in _speaker_turns(words, speaker)
            if float(turn["start"]) >= greeting_end
        ]
        first_response = response_turns[0] if response_turns else None
        diagnostics["first_post_greeting_turn"] = first_response
        candidates.append(diagnostics)

    eligible = [
        candidate
        for candidate in candidates
        if candidate["first_post_greeting_turn"] is not None
    ]
    if not eligible:
        raise RuntimeError("No non-agent speaker responded after the dealership greeting.")
    selected = min(
        eligible,
        key=lambda candidate: (
            float(candidate["first_post_greeting_turn"]["start"]),
            -float(candidate["total_speech_duration"]),
            str(candidate["speaker_id"]),
        ),
    )
    customer = str(selected["speaker_id"])
    selection_reason = (
        f"Selected {customer}: earliest non-agent spoken turn after the dealership "
        f"greeting ended at {greeting_end:.3f}s; response began at "
        f"{float(selected['first_post_greeting_turn']['start']):.3f}s. Total speech "
        "duration is the tie-breaker."
    )
    return customer, {
        "agent_speaker": agent,
        "customer_speaker": customer,
        "agent_greeting_turn": greeting,
        "agent_phrase_scores": scores,
        "agent_phrase_hits": phrase_hits,
        "speaker_text": texts,
        "candidate_speakers": candidates,
        "customer_selection_reason": selection_reason,
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


def _pcm_statistics(pcm: bytes) -> dict[str, Any]:
    usable_length = len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)
    samples = array("h")
    samples.frombytes(pcm[:usable_length])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return {
            "pcm_byte_length": len(pcm),
            "sample_count": 0,
            "peak_absolute_sample_amplitude": 0,
            "rms_amplitude": 0.0,
            "rms_dbfs": None,
        }
    peak = max(abs(sample) for sample in samples)
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    rms = math.sqrt(mean_square)
    rms_dbfs = 20.0 * math.log10(rms / 32768.0) if rms > 0 else None
    return {
        "pcm_byte_length": len(pcm),
        "sample_count": len(samples),
        "peak_absolute_sample_amplitude": peak,
        "rms_amplitude": rms,
        "rms_dbfs": rms_dbfs,
    }


def _transcript_delta(previous: str, current: str) -> str | None:
    if not previous:
        return current
    if current.startswith(previous):
        return current[len(previous) :].strip()
    return None


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


def _safe_event(payload: dict[str, Any]) -> dict[str, Any]:
    event: dict[str, Any] = {"type": str(payload.get("type") or "unknown")}
    if "interim" in payload:
        event["interim"] = bool(payload.get("interim"))
    text = _message_text(payload)
    if text:
        event["transcript"] = text
    scores = _scores_from_user_message(payload)
    if scores:
        event["prosody_score_count"] = len(scores)
    for key in ("code", "message", "chat_id", "chat_group_id"):
        value = payload.get(key)
        if value is not None and not isinstance(value, (dict, list)):
            event[key] = str(value)
    return event


def _hume_error(payload: dict[str, Any]) -> RuntimeError:
    code = str(payload.get("code") or "unknown")
    message = payload.get("message")
    safe_message = str(message) if isinstance(message, (str, int, float, bool)) else "Unknown error"
    return RuntimeError(f"Hume EVI error {code}: {safe_message}")


async def _receive_events(
    ws: Any,
    queue: asyncio.Queue[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> None:
    async for raw in ws:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            trace.append({"type": "invalid_json"})
            continue
        if not isinstance(payload, dict):
            trace.append({"type": "invalid_payload"})
            continue
        trace.append(_safe_event(payload))
        await queue.put(payload)


async def _wait_for_chat_metadata(
    queue: asyncio.Queue[dict[str, Any]], timeout_seconds: float
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RuntimeError("Timed out waiting for Hume EVI chat_metadata.")
        payload = await asyncio.wait_for(queue.get(), timeout=remaining)
        msg_type = payload.get("type")
        if msg_type == "error":
            raise _hume_error(payload)
        if msg_type == "chat_metadata":
            return payload


async def _wait_for_final_user_message(
    queue: asyncio.Queue[dict[str, Any]], timeout_seconds: float
) -> tuple[dict[str, Any], int]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    interim_count = 0
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            detail = (
                "Interim user_message observed, but no final user_message"
                if interim_count
                else "No interim user_message ever observed"
            )
            raise RuntimeError(f"Timed out waiting for final Hume EVI user_message: {detail}.")
        try:
            payload = await asyncio.wait_for(queue.get(), timeout=remaining)
        except TimeoutError as exc:
            detail = (
                "Interim user_message observed, but no final user_message"
                if interim_count
                else "No interim user_message ever observed"
            )
            raise RuntimeError(
                f"Timed out waiting for final Hume EVI user_message: {detail}."
            ) from exc
        msg_type = payload.get("type")
        if msg_type == "error":
            raise _hume_error(payload)
        if msg_type != "user_message":
            continue
        if payload.get("interim"):
            interim_count += 1
            continue
        return payload, interim_count


async def _stream_segment(
    ws: Any,
    queue: asyncio.Queue[dict[str, Any]],
    pcm: bytes,
    *,
    realtime_factor: float,
    response_timeout_seconds: float,
) -> tuple[dict[str, Any], int]:
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

    # The finalized-turn timeout starts only after all speech and silence is sent.
    return await _wait_for_final_user_message(queue, response_timeout_seconds)


async def _analyze_call(
    *,
    api_key: str,
    config_id: str | None,
    audio_path: Path,
    words: list[dict[str, Any]],
    realtime_factor: float,
    response_timeout_seconds: float,
    event_trace: list[dict[str, Any]],
    diagnostic_state: dict[str, Any],
) -> dict[str, Any]:
    customer_speaker, role_diagnostics = _infer_customer_speaker(words)
    segments = _customer_segments(words, customer_speaker)

    prepared_segments: list[tuple[dict[str, Any], bytes]] = []
    segment_diagnostics: list[dict[str, Any]] = []
    total_sample_count = 0
    total_sum_squares = 0.0
    durations: list[float] = []
    maximum_peak = 0
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        duration = max(0.0, end - start)
        if duration < 0.35:
            continue
        pcm = _extract_pcm(audio_path, start, end)
        pcm_stats = _pcm_statistics(pcm)
        prepared_segments.append((segment, pcm))
        durations.append(duration)
        sample_count = int(pcm_stats["sample_count"])
        rms = float(pcm_stats["rms_amplitude"])
        total_sample_count += sample_count
        total_sum_squares += rms * rms * sample_count
        maximum_peak = max(
            maximum_peak, int(pcm_stats["peak_absolute_sample_amplitude"])
        )
        segment_diagnostics.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "scribe_text": segment["text"],
                **{
                    key: value
                    for key, value in pcm_stats.items()
                    if key != "sample_count"
                },
            }
        )

    weighted_rms = (
        math.sqrt(total_sum_squares / total_sample_count)
        if total_sample_count
        else 0.0
    )
    diagnostic_state.update(
        {
            "customer_speaker": customer_speaker,
            "role_diagnostics": role_diagnostics,
            "customer_audio_diagnostics": {
                "customer_segment_count": len(prepared_segments),
                "total_extracted_customer_duration": round(sum(durations), 3),
                "minimum_segment_duration": round(min(durations), 3)
                if durations
                else None,
                "maximum_segment_duration": round(max(durations), 3)
                if durations
                else None,
                "mean_segment_duration": round(sum(durations) / len(durations), 3)
                if durations
                else None,
                "maximum_pcm_peak": maximum_peak,
                "weighted_rms_dbfs": (
                    20.0 * math.log10(weighted_rms / 32768.0)
                    if weighted_rms > 0
                    else None
                ),
                "segments": segment_diagnostics,
            },
        }
    )

    url = "wss://api.hume.ai/v0/evi/chat?" + urlencode(
        _evi_query(api_key, config_id)
    )

    segment_results: list[dict[str, Any]] = []
    aggregate_weighted: defaultdict[str, float] = defaultdict(float)
    total_weight = 0.0
    previous_final_transcript = ""

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        receiver = asyncio.create_task(_receive_events(ws, queue, event_trace))
        try:
            await _wait_for_chat_metadata(queue, response_timeout_seconds)
            await ws.send(json.dumps(_session_settings_payload()))
            # EVI keeps listening/transcribing/prosody-scoring while assistant responses
            # are paused, avoiding irrelevant generated speech during this diagnostic.
            await ws.send(json.dumps({"type": "pause_assistant_message"}))

            for segment, pcm in prepared_segments:
                start = float(segment["start"])
                end = float(segment["end"])
                duration = max(0.0, end - start)
                message, interim_count = await _stream_segment(
                    ws,
                    queue,
                    pcm,
                    realtime_factor=realtime_factor,
                    response_timeout_seconds=response_timeout_seconds,
                )
                scores = _scores_from_user_message(message)
                if not scores:
                    raise RuntimeError(
                        f"{audio_path.name}: Final user_message observed, but no prosody scores."
                    )
                raw_transcript = _message_text(message)
                transcript_delta = _transcript_delta(
                    previous_final_transcript, raw_transcript
                )
                previous_final_transcript = raw_transcript

                for name, score in scores.items():
                    aggregate_weighted[name] += score * duration
                total_weight += duration

                segment_results.append(
                    {
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "duration": round(duration, 3),
                        "scribe_text": segment["text"],
                        "interim_user_message_count": interim_count,
                        "raw_hume_transcript": raw_transcript,
                        "transcript_delta": transcript_delta,
                        "top_expressions": sorted(
                            scores.items(), key=lambda item: item[1], reverse=True
                        )[:15],
                        "watchlist_scores": {
                            label: scores[label] for label in WATCHLIST if label in scores
                        },
                        "all_expression_scores": scores,
                    }
                )
        finally:
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver

    if not segment_results or total_weight <= 0:
        raise RuntimeError(f"{audio_path.name}: no customer prosody results were produced.")

    aggregate = {
        name: score / total_weight for name, score in aggregate_weighted.items()
    }
    return {
        "name": audio_path.name,
        **diagnostic_state,
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
    trace_rows: list[dict[str, Any]] = []
    for audio_path in audio_paths:
        event_trace: list[dict[str, Any]] = []
        diagnostic_state: dict[str, Any] = {}
        row = rows_by_name.get(audio_path.name)
        if not row:
            blind_rows.append(
                {"name": audio_path.name, "error": "No Scribe diagnostics found."}
            )
            trace_rows.append({"name": audio_path.name, "events": event_trace})
            continue
        words = row.get("spoken_words")
        if not isinstance(words, list) or not words:
            blind_rows.append(
                {
                    "name": audio_path.name,
                    "error": "Scribe artifact has no spoken_words. Re-run Scribe.",
                }
            )
            trace_rows.append({"name": audio_path.name, "events": event_trace})
            continue
        try:
            result = await _analyze_call(
                api_key=api_key,
                config_id=config_id,
                audio_path=audio_path,
                words=words,
                realtime_factor=args.realtime_factor,
                response_timeout_seconds=args.response_timeout,
                event_trace=event_trace,
                diagnostic_state=diagnostic_state,
            )
        except Exception as exc:
            result = {
                "name": audio_path.name,
                **diagnostic_state,
                "error": str(exc),
            }
        blind_rows.append(result)
        trace_rows.append({"name": audio_path.name, "events": event_trace})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = args.output_dir / "hume_evi_scores_before_labels.json"
    blind_path.write_text(json.dumps(blind_rows, indent=2) + "\n", encoding="utf-8")
    trace_path = args.output_dir / "hume_evi_event_trace_before_labels.json"
    trace_path.write_text(json.dumps(trace_rows, indent=2) + "\n", encoding="utf-8")

    # Development-set comparison begins only after all Hume EVI results and safe event
    # traces are persisted.
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
                "error": row.get("error"),
                "aggregate_top_expressions": row.get("aggregate_top_expressions"),
                "aggregate_watchlist_scores": row.get("aggregate_watchlist_scores"),
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
    print(f"Safe EVI event trace: {trace_path}")
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
    parser.add_argument("--response-timeout", type=float, default=30.0)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
