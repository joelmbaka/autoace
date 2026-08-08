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

from autoace_backend.schemas import AudioAnalysis
from autoace_backend.scribe_diagnostics import (
    ElevenLabsScribeDiagnostics,
    ScribeDiagnosticsResult,
)
from autoace_backend.services import SUPPORTED_AUDIO_TYPES


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


def _diagnostic_row(result: ScribeDiagnosticsResult) -> dict[str, Any]:
    return result.model_dump()


async def evaluate(
    input_dir: Path,
    labels_path: Path,
    output_dir: Path,
    concurrency: int,
) -> int:
    load_dotenv(ROOT / ".env")
    audio_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_TYPES
    )
    if not audio_paths:
        raise RuntimeError(f"No supported audio files found in {input_dir}")

    analyzer = ElevenLabsScribeDiagnostics()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_audio(path: Path):
        mime_type = SUPPORTED_AUDIO_TYPES[path.suffix.lower()]
        async with semaphore:
            try:
                result = await asyncio.to_thread(
                    analyzer.analyze,
                    path.name,
                    path.read_bytes(),
                    mime_type,
                )
                return path.name, result, None
            except Exception as exc:
                return path.name, None, str(exc)

    # Diagnostic inference happens before labels are opened.
    completed = await asyncio.gather(*(run_audio(path) for path in audio_paths))

    rows: list[dict[str, Any]] = []
    successful: dict[str, ScribeDiagnosticsResult] = {}
    for name, result, error in completed:
        if result is None:
            rows.append({"name": name, "error": error})
        else:
            successful[name] = result
            rows.append(_diagnostic_row(result))

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "scribe_diagnostics_before_labels.json"
    raw_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    # Labels are inspected only after the provider output is fully persisted.
    if not labels_path.is_file():
        raise RuntimeError(f"Ground-truth file not found: {labels_path}")
    expected = _load_ground_truth(labels_path)

    comparisons: list[dict[str, Any]] = []
    for name, result in sorted(successful.items()):
        truth = expected.get(name)
        if truth is None:
            continue
        comparisons.append(
            {
                "name": name,
                "expected_acoustic_labels": {
                    "background_noise_present": truth.background_noise_present,
                    "background_noise_type": truth.background_noise_type,
                    "background_noise_severity": truth.background_noise_severity,
                    "audio_quality": truth.audio_quality,
                    "speaker_overlap_present": truth.speaker_overlap_present,
                    "long_silence_present": truth.long_silence_present,
                },
                "scribe_observations": {
                    "audio_events": [event.model_dump() for event in result.audio_events],
                    "speaker_ids": result.speaker_ids,
                    "overlap_intervals": [
                        interval.model_dump() for interval in result.overlap_intervals
                    ],
                    "total_overlap_seconds": result.total_overlap_seconds,
                    "max_interword_gap_seconds": result.max_interword_gap_seconds,
                },
            }
        )

    evaluation = {
        "purpose": (
            "Diagnostic only: determine whether a specialist STT model exposes the "
            "noise/diarization/timing evidence missing from general multimodal models."
        ),
        "files_discovered": len(audio_paths),
        "files_successful": len(successful),
        "files_failed": len(audio_paths) - len(successful),
        "comparisons": comparisons,
    }
    evaluation_path = output_dir / "diagnostic_comparison.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(evaluation, indent=2))
    print(f"\nRaw Scribe output: {raw_path}")
    print(f"Diagnostic comparison: {evaluation_path}")
    return 1 if len(successful) != len(audio_paths) else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect ElevenLabs Scribe v2 acoustic event and diarization signals."
    )
    parser.add_argument("input_dir", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--labels", type=Path, default=Path("labels.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/scribe_diagnostics")
    )
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    return asyncio.run(
        evaluate(args.input_dir, args.labels, args.output_dir, args.concurrency)
    )


if __name__ == "__main__":
    raise SystemExit(main())
