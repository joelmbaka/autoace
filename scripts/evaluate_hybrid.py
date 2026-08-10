from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from autoace_backend.hybrid_analyzer import HybridAnalyzer
from autoace_backend.schemas import AudioAnalysis
from autoace_backend.services import SCORED_FIELDS, SUPPORTED_AUDIO_TYPES, compare_prediction


async def blind(audio_dir: Path, output: Path, only: set[str] | None = None) -> int:
    # Intentionally discover audio by extension; do not enumerate or open labels.csv.
    audio_paths = sorted(
        path for path in audio_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_AUDIO_TYPES
        and (not only or path.name in only)
    )
    if not audio_paths:
        raise RuntimeError(f"No supported audio files found in {audio_dir}")
    analyzer = HybridAnalyzer()
    rows = []
    for path in audio_paths:
        try:
            details = await analyzer.analyze_detailed(
                path.name, path.read_bytes(), SUPPORTED_AUDIO_TYPES[path.suffix.lower()]
            )
            rows.append({
                "name": path.name,
                "prediction": details.analysis.result.model_dump(),
                "semantic_evidence": details.semantic_evidence.model_dump(),
                "emotion2vec": {
                    "aggregate_scores": details.emotion.aggregate_scores,
                    "negative_affect": details.emotion.negative_affect,
                    "positive_affect": details.emotion.positive_affect,
                    "neutral_affect": details.emotion.neutral_affect,
                },
                "customer_speaker": details.customer_speaker,
                "customer_transcript": details.customer_transcript,
                "provider_latency": details.provider_latency,
                "model": details.analysis.model,
                "error": None,
            })
        except Exception as exc:
            rows.append({"name": path.name, "prediction": None, "error": str(exc)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"phase": "blind", "rows": rows}, indent=2) + "\n")
    print(json.dumps({"phase": "blind", "artifact": str(output), "rows": rows}, indent=2))
    return 1 if any(row["error"] for row in rows) else 0


def compare(blind_path: Path, labels_path: Path, output: Path) -> int:
    blind_payload = json.loads(blind_path.read_text())
    predictions = {row["name"]: row for row in blind_payload["rows"]}
    comparisons = []
    failures = []
    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            predicted = predictions.get(row["name"])
            if not predicted or not predicted.get("prediction"):
                failures.append({"name": row["name"], "error": (predicted or {}).get("error", "missing prediction")})
                continue
            expected = AudioAnalysis.model_validate_json(row["result_json"])
            comparison = compare_prediction(AudioAnalysis.model_validate(predicted["prediction"]), expected)
            comparisons.append({"name": row["name"], **comparison})

    total_correct = sum(item["correct_fields"] for item in comparisons)
    total = len(comparisons) * len(SCORED_FIELDS)
    per_field = {
        field: sum(item["field_matches"][field] for item in comparisons) / len(comparisons)
        for field in SCORED_FIELDS
    } if comparisons else {}
    summary = {
        "exact_field_correct": total_correct,
        "exact_field_total": total,
        "exact_field_accuracy": total_correct / total if total else None,
        "per_field_accuracy": per_field,
        "tone_accuracy": per_field.get("emotional_tone"),
        "intensity_accuracy": per_field.get("emotional_intensity"),
        "comparisons": comparisons,
        "failures": failures,
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    blind_parser = subparsers.add_parser("blind")
    blind_parser.add_argument("audio_dir", type=Path)
    blind_parser.add_argument("output", type=Path)
    blind_parser.add_argument("--only", action="append", default=[])
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("blind_artifact", type=Path)
    compare_parser.add_argument("labels", type=Path)
    compare_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "blind":
        return asyncio.run(blind(args.audio_dir, args.output, set(args.only)))
    return compare(args.blind_artifact, args.labels, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
