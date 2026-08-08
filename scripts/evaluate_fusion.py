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

from autoace_backend.fusion_classifier import FusionAnalyzer, FusionResult
from autoace_backend.schemas import AudioAnalysis
from autoace_backend.services import (
    SUPPORTED_AUDIO_TYPES,
    build_validation_summary,
    compare_prediction,
)


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


def _serialize(result: FusionResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "result_json": result.analysis.result.model_dump(),
        "transcription": {
            "model": result.transcript.model,
            "request_seconds": result.transcript.request_seconds,
            "duration": result.transcript.duration,
            "language": result.transcript.language,
            "text": result.transcript.text,
            "segments": [segment.model_dump() for segment in result.transcript.segments],
        },
        "classification": {
            "model": result.analysis.model,
            "request_seconds": result.analysis.request_seconds,
            "usage": result.analysis.usage.model_dump(),
        },
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

    analyzer = FusionAnalyzer()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_audio(path: Path) -> tuple[Path, FusionResult | None, str | None]:
        mime_type = SUPPORTED_AUDIO_TYPES[path.suffix.lower()]
        async with semaphore:
            try:
                result = await analyzer.analyze(
                    path.name,
                    path.read_bytes(),
                    mime_type,
                )
                return path, result, None
            except Exception as exc:
                return path, None, str(exc)

    # Inference first: labels.csv is not opened or passed into either provider request.
    # Note that this is a development-set experiment after earlier labels were inspected,
    # so it is label-isolated at runtime rather than a claim of untouched blind tuning.
    completed = await asyncio.gather(*(run_audio(path) for path in audio_paths))

    successful: dict[str, FusionResult] = {}
    prediction_rows: list[dict[str, Any]] = []
    for path, result, error in completed:
        if result is None:
            prediction_rows.append({"name": path.name, "error": error})
        else:
            successful[path.name] = result
            prediction_rows.append(_serialize(result))

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions_before_label_comparison.json"
    predictions_path.write_text(
        json.dumps(prediction_rows, indent=2) + "\n", encoding="utf-8"
    )

    transcripts_path = output_dir / "transcripts.json"
    transcripts_path.write_text(
        json.dumps(
            [
                {
                    "name": name,
                    "model": result.transcript.model,
                    "request_seconds": result.transcript.request_seconds,
                    "duration": result.transcript.duration,
                    "language": result.transcript.language,
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

    # Ground truth is opened only after all provider inference has completed and predictions
    # have been materialized to disk.
    ground_truth = _load_ground_truth(labels_path)

    comparisons: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for name, result in sorted(successful.items()):
        expected = ground_truth.get(name)
        comparison = None
        if expected is not None:
            comparison = compare_prediction(result.analysis.result, expected)
            comparison["predicted_tone"] = result.analysis.result.emotional_tone
            comparisons.append(comparison)
        rows.append(
            {
                "name": name,
                "prediction": result.analysis.result.model_dump(),
                "expected": expected.model_dump() if expected else None,
                "validation": comparison,
                "transcription_model": result.transcript.model,
                "transcription_seconds": result.transcript.request_seconds,
                "classification_model": result.analysis.model,
                "classification_seconds": result.analysis.request_seconds,
                "usage": result.analysis.usage.model_dump(),
            }
        )

    summary = {
        "experiment": "Groq Whisper Turbo transcript + raw audio -> Gemini 3.5 Flash-Lite",
        "development_set_note": (
            "The original labels were inspected during earlier experiments. This run keeps labels "
            "out of runtime inference and compares them only afterward; it is not presented as an "
            "untouched blind benchmark."
        ),
        "files_discovered": len(audio_paths),
        "files_successful": len(successful),
        "files_failed": len(audio_paths) - len(successful),
        "validation": build_validation_summary(comparisons),
        "rows": rows,
    }

    evaluation_path = output_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nPredictions before label comparison: {predictions_path}")
    print(f"Transcripts: {transcripts_path}")
    print(f"Evaluation: {evaluation_path}")
    return 1 if len(successful) != len(audio_paths) else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate transcript+raw-audio fusion using Groq Whisper Turbo and "
            "Gemini 3.5 Flash-Lite without changing the production dashboard path."
        )
    )
    parser.add_argument("input_dir", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--labels", type=Path, default=Path("labels.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/fusion_experiment")
    )
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    return asyncio.run(
        evaluate(args.input_dir, args.labels, args.output_dir, args.concurrency)
    )


if __name__ == "__main__":
    raise SystemExit(main())
