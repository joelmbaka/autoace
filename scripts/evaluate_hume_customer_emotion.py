from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoace_backend.schemas import AudioAnalysis
from autoace_backend.services import SUPPORTED_AUDIO_TYPES


HUME_BASE_URL = "https://api.hume.ai"
AGENT_PHRASES: tuple[tuple[str, float], ...] = (
    ("how can i help", 5.0),
    ("i'm erica", 5.0),
    ("i am erica", 5.0),
    ("toyota of", 3.0),
    ("lexington toyota", 3.0),
    ("what type of service", 3.0),
    ("just to confirm", 2.5),
    ("dealership is closed", 3.0),
    ("did you want to transfer", 3.0),
    ("transferring you", 3.0),
    ("i can help with that", 2.0),
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


def _mime_type(path: Path) -> str:
    return SUPPORTED_AUDIO_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _load_ground_truth(path: Path) -> dict[str, AudioAnalysis]:
    expected: dict[str, AudioAnalysis] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"name", "result_json"}.issubset(reader.fieldnames):
            raise RuntimeError("labels.csv must contain name and result_json columns.")
        for row_number, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            raw = (row.get("result_json") or "").strip()
            if not name or not raw:
                continue
            try:
                expected[name] = AudioAnalysis.model_validate(json.loads(raw))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid ground truth in labels.csv row {row_number}: {exc}"
                ) from exc
    return expected


def _job_status(details: dict[str, Any]) -> str:
    state = details.get("state")
    if isinstance(state, dict):
        value = state.get("status") or state.get("state") or state.get("name")
        return str(value or "UNKNOWN").upper()
    return str(state or details.get("status") or "UNKNOWN").upper()


def _prediction_entries(payload: Any) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    entries: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    def visit(value: Any, source: dict[str, Any] | None = None) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item, source)
            return
        if not isinstance(value, dict):
            return

        next_source = source
        if isinstance(value.get("source"), dict):
            next_source = value["source"]

        results = value.get("results")
        if isinstance(results, dict):
            predictions = results.get("predictions")
            if isinstance(predictions, list):
                for prediction in predictions:
                    if isinstance(prediction, dict):
                        entries.append((prediction, next_source))

        if isinstance(value.get("models"), dict) and (
            "file" in value or "filename" in value or not entries
        ):
            entries.append((value, next_source))

    visit(payload)
    deduped: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    seen: set[int] = set()
    for entry, source in entries:
        marker = id(entry)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append((entry, source))
    return deduped


def _entry_filename(entry: dict[str, Any], source: dict[str, Any] | None) -> str | None:
    for key in ("file", "filename", "name"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).name
    if source:
        for key in ("filename", "file", "name"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return Path(value).name
        url = source.get("url")
        if isinstance(url, str) and url.strip():
            return Path(urlparse(url).path).name or None
    return None


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _speaker_summary(entry: dict[str, Any]) -> dict[str, Any]:
    models = entry.get("models")
    if not isinstance(models, dict):
        raise RuntimeError("Hume prediction is missing models.")
    prosody = models.get("prosody")
    if not isinstance(prosody, dict):
        raise RuntimeError("Hume prediction is missing the prosody model result.")
    groups = prosody.get("grouped_predictions")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("Hume returned no grouped prosody predictions.")

    speakers: dict[str, dict[str, Any]] = {}
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        speaker_id = str(group.get("id") or f"speaker_{group_index}")
        predictions = group.get("predictions")
        if not isinstance(predictions, list):
            continue

        turns: list[dict[str, Any]] = []
        weighted_scores: defaultdict[str, float] = defaultdict(float)
        total_weight = 0.0
        total_duration = 0.0
        text_parts: list[str] = []
        first_begin: float | None = None

        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            text = str(prediction.get("text") or "").strip()
            time_value = prediction.get("time")
            begin = end = 0.0
            if isinstance(time_value, dict):
                begin = _float(time_value.get("begin"))
                end = _float(time_value.get("end"), begin)
            duration = max(0.0, end - begin)
            weight = max(duration, 0.25)
            if first_begin is None or begin < first_begin:
                first_begin = begin
            total_duration += duration
            total_weight += weight
            if text:
                text_parts.append(text)

            emotion_scores: dict[str, float] = {}
            emotions = prediction.get("emotions")
            if isinstance(emotions, list):
                for emotion in emotions:
                    if not isinstance(emotion, dict):
                        continue
                    name = str(emotion.get("name") or "").strip()
                    if not name:
                        continue
                    score = _float(emotion.get("score"))
                    emotion_scores[name] = score
                    weighted_scores[name] += score * weight

            turns.append(
                {
                    "text": text,
                    "begin": begin,
                    "end": end,
                    "duration": round(duration, 3),
                    "confidence": prediction.get("confidence"),
                    "speaker_confidence": prediction.get("speaker_confidence"),
                    "top_expressions": sorted(
                        emotion_scores.items(), key=lambda item: item[1], reverse=True
                    )[:12],
                    "all_expression_scores": emotion_scores,
                }
            )

        aggregate = {
            name: score / total_weight
            for name, score in weighted_scores.items()
            if total_weight > 0
        }
        transcript = " ".join(text_parts).strip()
        lower = transcript.casefold()
        role_hits = [phrase for phrase, _ in AGENT_PHRASES if phrase in lower]
        role_score = sum(weight for phrase, weight in AGENT_PHRASES if phrase in lower)

        speakers[speaker_id] = {
            "speaker_id": speaker_id,
            "transcript": transcript,
            "first_begin": first_begin,
            "speech_duration_seconds": round(total_duration, 3),
            "role_agent_score": role_score,
            "role_agent_phrase_hits": role_hits,
            "top_expressions": sorted(
                aggregate.items(), key=lambda item: item[1], reverse=True
            )[:20],
            "watchlist_scores": {
                label: aggregate[label] for label in WATCHLIST if label in aggregate
            },
            "all_aggregate_expression_scores": aggregate,
            "turns": turns,
        }

    if not speakers:
        raise RuntimeError("Hume returned no usable prosody speakers.")

    ordered = sorted(
        speakers.values(),
        key=lambda speaker: (
            -float(speaker["role_agent_score"]),
            float(speaker["first_begin"] or 0.0),
        ),
    )
    agent = ordered[0]
    non_agents = [speaker for speaker in speakers.values() if speaker is not agent]
    if non_agents:
        customer = max(
            non_agents, key=lambda speaker: float(speaker["speech_duration_seconds"])
        )
        role_method = "agent phrase heuristic; longest remaining speaker selected as customer"
    else:
        customer = agent
        role_method = "single speaker returned; customer/agent separation unavailable"

    return {
        "role_inference_method": role_method,
        "agent_speaker_id": agent["speaker_id"],
        "customer_speaker_id": customer["speaker_id"],
        "agent_role_phrase_hits": agent["role_agent_phrase_hits"],
        "customer_top_expressions": customer["top_expressions"],
        "customer_watchlist_scores": customer["watchlist_scores"],
        "speakers": speakers,
    }


def _submit_job(
    client: httpx.Client, api_key: str, audio_paths: list[Path]
) -> tuple[str, float]:
    configuration = {
        "models": {
            "prosody": {
                "granularity": "conversational_turn",
                "identify_speakers": True,
            }
        },
        "transcription": {
            "language": None,
            "identify_speakers": True,
            "confidence_threshold": 0.5,
        },
        "notify": False,
    }
    files = [
        ("file", (path.name, path.read_bytes(), _mime_type(path))) for path in audio_paths
    ]
    started = time.perf_counter()
    response = client.post(
        f"{HUME_BASE_URL}/v0/batch/jobs",
        headers={"X-Hume-Api-Key": api_key},
        data={"json": json.dumps(configuration, separators=(",", ":"))},
        files=files,
    )
    request_seconds = time.perf_counter() - started
    if response.is_error:
        raise RuntimeError(
            f"Hume job submission failed ({response.status_code}): {response.text[:1000]}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Hume job submission returned non-JSON data.") from exc
    job_id = payload.get("job_id") if isinstance(payload, dict) else None
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError(f"Hume job submission did not return job_id: {payload!r}")
    return job_id, round(request_seconds, 3)


def _await_job(
    client: httpx.Client,
    api_key: str,
    job_id: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    deadline = started + timeout_seconds
    last_status = "UNKNOWN"
    headers = {"X-Hume-Api-Key": api_key}

    while time.perf_counter() < deadline:
        response = client.get(
            f"{HUME_BASE_URL}/v0/batch/jobs/{job_id}", headers=headers
        )
        if response.is_error:
            raise RuntimeError(
                f"Hume job status failed ({response.status_code}): {response.text[:1000]}"
            )
        try:
            details = response.json()
        except ValueError as exc:
            raise RuntimeError("Hume job status returned non-JSON data.") from exc
        if not isinstance(details, dict):
            raise RuntimeError("Hume job status returned an unexpected response shape.")

        last_status = _job_status(details)
        if last_status == "COMPLETED":
            return details, round(time.perf_counter() - started, 3)
        if last_status == "FAILED":
            raise RuntimeError(f"Hume batch job failed: {json.dumps(details)[:2000]}")
        time.sleep(poll_seconds)

    raise RuntimeError(
        f"Timed out after {timeout_seconds:.0f}s waiting for Hume job {job_id}; "
        f"last status was {last_status}."
    )


def _get_predictions(client: httpx.Client, api_key: str, job_id: str) -> Any:
    response = client.get(
        f"{HUME_BASE_URL}/v0/batch/jobs/{job_id}/predictions",
        headers={"X-Hume-Api-Key": api_key},
    )
    if response.is_error:
        raise RuntimeError(
            f"Hume predictions failed ({response.status_code}): {response.text[:1000]}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("Hume predictions returned non-JSON data.") from exc


def evaluate(
    input_dir: Path,
    labels_path: Path,
    output_dir: Path,
    timeout_seconds: float,
    poll_seconds: float,
) -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        raise RuntimeError("HUME_API_KEY is not configured in .env.")

    audio_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_TYPES
    )
    if not audio_paths:
        raise RuntimeError(f"No supported audio files found in {input_dir}")

    http_timeout = float(os.getenv("HUME_HTTP_TIMEOUT_SECONDS", "120"))
    with httpx.Client(timeout=http_timeout) as client:
        job_id, submit_seconds = _submit_job(client, api_key, audio_paths)
        job_details, processing_seconds = _await_job(
            client,
            api_key,
            job_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        predictions_payload = _get_predictions(client, api_key, job_id)

    parsed: list[dict[str, Any]] = []
    known_names = {path.name for path in audio_paths}
    for entry, source in _prediction_entries(predictions_payload):
        filename = _entry_filename(entry, source)
        if filename not in known_names:
            continue
        try:
            summary = _speaker_summary(entry)
            parsed.append({"name": filename, **summary})
        except Exception as exc:
            parsed.append({"name": filename, "parse_error": str(exc)})

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "hume_raw_before_labels.json"
    raw_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "submit_seconds": submit_seconds,
                "processing_wait_seconds": processing_seconds,
                "job_details": job_details,
                "predictions": predictions_payload,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    parsed_path = output_dir / "customer_prosody_before_labels.json"
    parsed_path.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")

    # Development-label comparison starts only after provider output and parsed
    # customer prosody have both been persisted to disk.
    if not labels_path.is_file():
        raise RuntimeError(f"Ground-truth file not found: {labels_path}")
    expected = _load_ground_truth(labels_path)

    comparisons: list[dict[str, Any]] = []
    for row in parsed:
        name = row.get("name")
        truth = expected.get(str(name))
        if truth is None:
            continue
        comparisons.append(
            {
                "name": name,
                "expected_emotional_tone": truth.emotional_tone,
                "expected_emotional_intensity": truth.emotional_intensity,
                "customer_speaker_id": row.get("customer_speaker_id"),
                "agent_speaker_id": row.get("agent_speaker_id"),
                "agent_role_phrase_hits": row.get("agent_role_phrase_hits"),
                "customer_top_expressions": row.get("customer_top_expressions"),
                "customer_watchlist_scores": row.get("customer_watchlist_scores"),
                "parse_error": row.get("parse_error"),
            }
        )

    comparison = {
        "experiment": "Hume Expression Measurement prosody, hosted batch API",
        "note": (
            "Development diagnostic only. Hume expression scores are measurements of "
            "expressive behavior and are not mapped one-to-one to AutoAce labels."
        ),
        "job_id": job_id,
        "files_submitted": len(audio_paths),
        "files_parsed": sum(1 for row in parsed if not row.get("parse_error")),
        "submit_seconds": submit_seconds,
        "processing_wait_seconds": processing_seconds,
        "comparisons": comparisons,
    }
    comparison_path = output_dir / "comparison_after_labels.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(comparison, indent=2))
    print(f"\nRaw Hume response: {raw_path}")
    print(f"Customer prosody before labels: {parsed_path}")
    print(f"Comparison after labels: {comparison_path}")

    return 1 if any(row.get("parse_error") for row in parsed) or not parsed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a hosted Hume Expression Measurement prosody diagnostic over the "
            "assessment calls. No local ML model is installed or executed."
        )
    )
    parser.add_argument("input_dir", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--labels", type=Path, default=Path("labels.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/hume_customer_emotion")
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("HUME_JOB_TIMEOUT_SECONDS", "300")),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.getenv("HUME_POLL_SECONDS", "3")),
    )
    args = parser.parse_args()
    return evaluate(
        args.input_dir,
        args.labels,
        args.output_dir,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=max(1.0, args.poll_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
