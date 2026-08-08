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

from autoace_backend.metrics import emotional_tone_metrics
from autoace_backend.services import (
    GeminiAnalyzer,
    build_validation_summary,
    compare_prediction,
    prepare_batch,
)


async def evaluate(input_path: Path, output_dir: Path, concurrency: int) -> int:
    if input_path.is_dir():
        uploaded = [
            (path.name, path.read_bytes())
            for path in input_path.iterdir()
            if path.is_file()
        ]
    elif input_path.is_file():
        uploaded = [(input_path.name, input_path.read_bytes())]
    else:
        raise SystemExit(f"Input does not exist: {input_path}")

    batch = prepare_batch(uploaded)
    analyzer = GeminiAnalyzer()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_item(item):
        if item.preflight_error:
            return item, None, item.preflight_error
        async with semaphore:
            try:
                result = await analyzer.analyze(item.name, item.data, item.mime_type)
                return item, result, None
            except Exception as exc:
                return item, None, str(exc)

    completed = await asyncio.gather(*(run_item(item) for item in batch.items))

    rows: list[dict] = []
    comparisons: list[dict] = []
    for item, analysis, error in completed:
        row = {
            "name": item.name,
            "result_json": analysis.result.model_dump() if analysis else None,
            "error": error,
            "model": analysis.model if analysis else None,
            "request_seconds": analysis.request_seconds if analysis else None,
            "usage": analysis.usage.model_dump() if analysis else None,
        }
        if analysis and item.expected:
            comparison = compare_prediction(analysis.result, item.expected)
            comparison["predicted_tone"] = analysis.result.emotional_tone
            comparisons.append(comparison)
            row["validation"] = comparison
        rows.append(row)

    summary = build_validation_summary(comparisons)
    if summary is not None:
        summary["emotional_tone_metrics"] = emotional_tone_metrics(
            summary["emotional_tone_confusion_matrix"]
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "predictions.json").write_text(
        json.dumps(
            [{"name": row["name"], "result_json": row["result_json"]} for row in rows],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "result_json"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "result_json": json.dumps(row["result_json"], separators=(",", ":"))
                    if row["result_json"]
                    else "",
                }
            )

    (output_dir / "run_details.json").write_text(
        json.dumps(
            {
                "warnings": batch.warnings,
                "rows": rows,
                "validation": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"validation": summary, "rows": rows}, indent=2))
    print(f"\nExports written to {output_dir}")
    return 1 if any(row["error"] for row in rows) else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the AutoAce trial classifier against a folder or ZIP batch."
    )
    parser.add_argument("input", type=Path, help="Evaluation folder or ZIP archive")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    return asyncio.run(evaluate(args.input, args.output_dir, args.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
