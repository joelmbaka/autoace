from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from autoace_backend.hybrid_emotion import HybridEmotionAnalyzer, HybridEmotionResult
from autoace_backend.schemas import AudioAnalysis
from autoace_backend.services import SUPPORTED_AUDIO_TYPES


TONE_LABELS = ["neutral", "satisfied", "frustrated", "upset", "distressed"]
INTENSITY_LABELS = ["low", "medium", "high"]


def _serialize_result(result: HybridEmotionResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "prediction": result.classification.prediction.model_dump(),
        "transcription": {
            "model": result.transcript.model,
            "request_seconds": result.transcript.request_seconds,
            "language": result.transcript.language,
            "duration": result.transcript.duration,
            "text": result.transcript.text,
            "segments": [segment.model_dump() for segment in result.transcript.segments],
        },
        "classification": {
            "model": result.classification.model,
            "request_seconds": result.classification.request_seconds,
            "usage": result.classification.usage.model_dump(),
        },
    }


def _load_ground_truth(labels_path: Path) -> dict[str, AudioAnalysis]:
    if not labels_path.is_file():
        raise RuntimeError(f"Ground-truth file not found: {labels_path}")

    expected: dict[str, AudioAnalysis] = {}
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"name", "result_json"}.issubset(reader.fieldnames):
            raise RuntimeError("labels.csv must contain name and result_json columns.")
        for row_number, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            raw_result = (row.get("result_json") or "").strip()
            if not name or not raw_result:
                continue
            try:
                expected[name] = AudioAnalysis.model_validate(json.loads(raw_result))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid ground truth in labels.csv row {row_number}: {exc}"
                ) from exc
    return expected


def _classification_metrics(
    expected_values: list[str], predicted_values: list[str], labels: list[str]
) -> dict[str, Any]:
    confusion = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    for expected, predicted in zip(expected_values, predicted_values, strict=True):
        if expected in confusion and predicted in confusion[expected]:
            confusion[expected][predicted] += 1

    per_class: dict[str, Any] = {}
    observed_f1: list[float] = []
    all_f1: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        all_f1.append(f1)
        if support:
            observed_f1.append(f1)

    correct = sum(
        1 for expected, predicted in zip(expected_values, predicted_values, strict=True)
        if expected == predicted
    )
    return {
        "accuracy": correct / len(expected_values) if expected_values else 0.0,
        "confusion_matrix": confusion,
        "per_class": per_class,
        "macro_f1_observed_classes": (
            sum(observed_f1) / len(observed_f1) if observed_f1 else 0.0
        ),
        "macro_f1_all_classes": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
    }


async def evaluate(
    input_dir: Path,
    labels_path: Path,
    output_dir: Path,
    concurrency: int,
) -> int:
    load_dotenv()

    audio_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_TYPES
    )
    if not audio_paths:
        raise RuntimeError(f"No supported audio files found in {input_dir}")

    analyzer = HybridEmotionAnalyzer()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_audio(path: Path) -> tuple[Path, HybridEmotionResult | None, str | None]:
        mime_type = SUPPORTED_AUDIO_TYPES[path.suffix.lower()]
        async with semaphore:
            try:
                result = await asyncio.to_thread(
                    analyzer.analyze,
                    path.name,
                    path.read_bytes(),
                    mime_type,
                )
                return path, result, None
            except Exception as exc:
                return path, None, str(exc)

    # Blind phase: labels.csv is intentionally not opened until every inference task is complete.
    completed = await asyncio.gather(*(run_audio(path) for path in audio_paths))

    blind_rows: list[dict[str, Any]] = []
    successful: dict[str, HybridEmotionResult] = {}
    for path, result, error in completed:
        if result is not None:
            successful[path.name] = result
            blind_rows.append(_serialize_result(result))
        else:
            blind_rows.append({"name": path.name, "error": error})

    output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = output_dir / "blind_predictions.json"
    blind_path.write_text(json.dumps(blind_rows, indent=2) + "\n", encoding="utf-8")

    transcript_path = output_dir / "transcripts.json"
    transcript_path.write_text(
        json.dumps(
            [
                {
                    "name": name,
                    "model": result.transcript.model,
                    "request_seconds": result.transcript.request_seconds,
                    "language": result.transcript.language,
                    "duration": result.transcript.duration,
                    "text": result.transcript.text,
                    "segments": [
                        segment.model_dump() for segment in result.transcript.segments
                    ],
                }
                for name, result in sorted(successful.items())
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Evaluation phase begins only after blind predictions are fully materialized on disk.
    ground_truth = _load_ground_truth(labels_path)

    comparison_rows: list[dict[str, Any]] = []
    expected_tones: list[str] = []
    predicted_tones: list[str] = []
    expected_intensities: list[str] = []
    predicted_intensities: list[str] = []

    for name, result in sorted(successful.items()):
        expected = ground_truth.get(name)
        if expected is None:
            continue
        prediction = result.classification.prediction
        tone_match = prediction.emotional_tone == expected.emotional_tone
        intensity_match = prediction.emotional_intensity == expected.emotional_intensity
        comparison_rows.append(
            {
                "name": name,
                "expected_tone": expected.emotional_tone,
                "predicted_tone": prediction.emotional_tone,
                "tone_match": tone_match,
                "expected_intensity": expected.emotional_intensity,
                "predicted_intensity": prediction.emotional_intensity,
                "intensity_match": intensity_match,
                "confidence": prediction.confidence,
                "transcription_seconds": result.transcript.request_seconds,
                "classification_seconds": result.classification.request_seconds,
                "transcription_model": result.transcript.model,
                "classification_model": result.classification.model,
                "llm_usage": result.classification.usage.model_dump(),
            }
        )
        expected_tones.append(expected.emotional_tone)
        predicted_tones.append(prediction.emotional_tone)
        expected_intensities.append(expected.emotional_intensity)
        predicted_intensities.append(prediction.emotional_intensity)

    summary = {
        "files_discovered": len(audio_paths),
        "files_successful": len(successful),
        "files_failed": len(audio_paths) - len(successful),
        "evaluated_files": len(comparison_rows),
        "tone": _classification_metrics(expected_tones, predicted_tones, TONE_LABELS),
        "intensity": _classification_metrics(
            expected_intensities, predicted_intensities, INTENSITY_LABELS
        ),
        "comparisons": comparison_rows,
    }

    (output_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nBlind predictions: {blind_path}")
    print(f"Transcripts: {transcript_path}")
    print(f"Evaluation: {output_dir / 'evaluation.json'}")

    return 1 if len(successful) != len(audio_paths) else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Blindly evaluate Groq Whisper Large V3 + NVIDIA Nemotron on customer "
            "emotional tone/intensity without changing the production classifier."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Directory containing the assessment audio files (default: repository root)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("labels.csv"),
        help="Ground-truth CSV; it is not opened until all blind predictions finish",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hybrid_emotion"),
    )
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    return asyncio.run(
        evaluate(args.input_dir, args.labels, args.output_dir, args.concurrency)
    )


if __name__ == "__main__":
    raise SystemExit(main())
