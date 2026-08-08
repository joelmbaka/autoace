# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "av>=14.0.0",
#   "numpy>=2.0.0",
#   "torch>=2.6.0",
#   "transformers>=4.48.0",
# ]
# ///

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


MODEL_ID = "superb/wav2vec2-base-superb-er"
SAMPLE_RATE = 16_000
AGENT_PHRASES = (
    "i'm erica",
    "i am erica",
    "how can i help",
    "toyota of",
    "lexington toyota",
    "the dealership is closed",
    "did you want to transfer",
    "transferring you",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_audio(path: Path) -> np.ndarray:
    chunks: list[np.ndarray] = []
    resampler = av.audio.resampler.AudioResampler(
        format="fltp", layout="mono", rate=SAMPLE_RATE
    )
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            for output in resampler.resample(frame):
                chunks.append(output.to_ndarray().reshape(-1).astype(np.float32))
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray().reshape(-1).astype(np.float32))
    if not chunks:
        raise RuntimeError(f"Could not decode audio: {path}")
    return np.concatenate(chunks)


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

    scores: dict[str, int] = {}
    phrase_hits: dict[str, list[str]] = {}
    for speaker, text in texts.items():
        lowered = text.casefold()
        hits = [phrase for phrase in AGENT_PHRASES if phrase in lowered]
        phrase_hits[speaker] = hits
        scores[speaker] = len(hits)

    agent = max(scores, key=scores.get)
    if scores[agent] <= 0:
        raise RuntimeError(
            "Could not infer the dealership-agent speaker from transcript role cues."
        )

    others = [speaker for speaker in texts if speaker != agent]
    if len(others) != 1:
        raise RuntimeError(
            f"Expected exactly one non-agent speaker for this diagnostic; found {others}."
        )
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
) -> list[tuple[float, float, str]]:
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

    segments: list[tuple[float, float, str]] = []
    current_start, current_end, first_text = timed[0]
    current_words = [first_text]

    for start, end, text in timed[1:]:
        gap = start - current_end
        projected_duration = end - current_start
        if gap <= 0.8 and projected_duration <= 9.0:
            current_end = max(current_end, end)
            current_words.append(text)
        else:
            segments.append((current_start, current_end, " ".join(current_words)))
            current_start, current_end, current_words = start, end, [text]
    segments.append((current_start, current_end, " ".join(current_words)))
    return segments


def _normalize_label(label: str) -> str:
    lowered = label.casefold()
    aliases = {
        "neu": "neutral",
        "neutral": "neutral",
        "hap": "happy",
        "happy": "happy",
        "ang": "angry",
        "angry": "angry",
        "sad": "sad",
    }
    return aliases.get(lowered, lowered)


def _score_segments(
    waveform: np.ndarray,
    segments: list[tuple[float, float, str]],
    feature_extractor: AutoFeatureExtractor,
    model: AutoModelForAudioClassification,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    weighted: dict[str, float] = {}
    total_weight = 0.0

    for start, end, text in segments:
        padded_start = max(0.0, start - 0.10)
        padded_end = end + 0.10
        start_idx = int(padded_start * SAMPLE_RATE)
        end_idx = min(len(waveform), int(padded_end * SAMPLE_RATE))
        clip = waveform[start_idx:end_idx]
        duration = len(clip) / SAMPLE_RATE
        if duration < 0.35:
            continue

        inputs = feature_extractor(
            clip,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        scores: dict[str, float] = {}
        for index, probability in enumerate(probabilities):
            raw_label = str(model.config.id2label[index])
            label = _normalize_label(raw_label)
            scores[label] = float(probability)
            weighted[label] = weighted.get(label, 0.0) + float(probability) * duration

        total_weight += duration
        rows.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "text": text,
                "scores": scores,
                "top_label": max(scores, key=scores.get),
            }
        )

    if not rows or total_weight <= 0:
        raise RuntimeError("No usable customer segments remained for emotion inference.")

    aggregate = {
        label: score / total_weight for label, score in sorted(weighted.items())
    }
    return rows, aggregate


def _coarse_expected(autoace_tone: str) -> str | None:
    return {
        "neutral": "neutral",
        "satisfied": "happy",
        "frustrated": "angry",
        "upset": "angry",
        "distressed": "sad",
    }.get(autoace_tone)


def _load_expected_tones(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("name") or "").strip()
            raw = (row.get("result_json") or "").strip()
            if not name or not raw:
                continue
            payload = json.loads(raw)
            expected[name] = str(payload["emotional_tone"])
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only customer-speech emotion test using the Apache-2.0 "
            "SUPERB Wav2Vec2 emotion-recognition model."
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
        "--output-dir", type=Path, default=Path("artifacts/local_customer_emotion")
    )
    args = parser.parse_args()

    if not args.scribe_artifact.is_file():
        raise RuntimeError(
            "Scribe artifact is missing. Re-run evaluate_scribe_diagnostics.py after "
            "pulling the version that persists spoken_words."
        )

    scribe_rows = _load_json(args.scribe_artifact)
    if not isinstance(scribe_rows, list):
        raise RuntimeError("Unexpected Scribe diagnostics artifact shape.")

    print(f"Loading local emotion model: {MODEL_ID}")
    started = time.perf_counter()
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()
    model_load_seconds = time.perf_counter() - started

    blind_rows: list[dict[str, Any]] = []
    for row in scribe_rows:
        if not isinstance(row, dict) or row.get("error"):
            continue
        name = str(row.get("name") or "")
        words = row.get("spoken_words")
        if not isinstance(words, list) or not words:
            raise RuntimeError(
                f"{name}: Scribe artifact has no spoken_words. Re-run the Scribe diagnostic."
            )
        audio_path = args.input_dir / name
        if not audio_path.is_file():
            raise RuntimeError(f"Missing local audio: {audio_path}")

        customer_speaker, role_diagnostics = _infer_customer_speaker(words)
        segments = _customer_segments(words, customer_speaker)
        waveform = _decode_audio(audio_path)

        inference_started = time.perf_counter()
        segment_rows, aggregate = _score_segments(
            waveform, segments, feature_extractor, model
        )
        inference_seconds = time.perf_counter() - inference_started
        top_label = max(aggregate, key=aggregate.get)

        blind_rows.append(
            {
                "name": name,
                "model": MODEL_ID,
                "model_license": "Apache-2.0",
                "customer_speaker": customer_speaker,
                "role_diagnostics": role_diagnostics,
                "segments": segment_rows,
                "aggregate_scores": aggregate,
                "top_label": top_label,
                "inference_seconds": round(inference_seconds, 3),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = args.output_dir / "emotion_scores_before_labels.json"
    blind_path.write_text(json.dumps(blind_rows, indent=2) + "\n", encoding="utf-8")

    # Runtime label isolation: ground truth is opened only after every local model
    # result has been materialized. These three labels are already known from prior
    # experiments, so this is a development-set diagnostic, not a pristine holdout.
    expected_tones = _load_expected_tones(args.labels)
    comparisons: list[dict[str, Any]] = []
    for row in blind_rows:
        autoace_tone = expected_tones.get(row["name"])
        expected_coarse = _coarse_expected(autoace_tone) if autoace_tone else None
        comparisons.append(
            {
                "name": row["name"],
                "autoace_tone": autoace_tone,
                "expected_coarse_emotion": expected_coarse,
                "predicted_coarse_emotion": row["top_label"],
                "directional_match": (
                    row["top_label"] == expected_coarse if expected_coarse else None
                ),
                "aggregate_scores": row["aggregate_scores"],
            }
        )

    result = {
        "purpose": (
            "Diagnostic only: determine whether customer-only prosody contains an "
            "emotion direction that aligns with AutoAce tone labels. No direct "
            "AutoAce taxonomy mapping is proposed from this three-call set."
        ),
        "model": MODEL_ID,
        "license": "Apache-2.0",
        "model_load_seconds": round(model_load_seconds, 3),
        "comparisons": comparisons,
    }
    comparison_path = args.output_dir / "comparison_after_labels.json"
    comparison_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nRaw local scores: {blind_path}")
    print(f"Post-label comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
