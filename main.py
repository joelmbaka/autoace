from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from autoace_backend.services import GeminiAnalyzer, SUPPORTED_AUDIO_TYPES


async def run(audio_path: Path) -> int:
    if not audio_path.is_file():
        raise SystemExit(f"Audio file not found: {audio_path}")

    suffix = audio_path.suffix.lower()
    mime_type = SUPPORTED_AUDIO_TYPES.get(suffix)
    if mime_type is None:
        raise SystemExit(f"Unsupported audio extension: {suffix or '(none)'}")

    analyzer = GeminiAnalyzer()
    result = await analyzer.analyze(audio_path.name, audio_path.read_bytes(), mime_type)
    print(json.dumps(result.model_dump(), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze one AutoAce call audio file.")
    parser.add_argument("audio", type=Path, help="Path to an audio file")
    args = parser.parse_args()
    return asyncio.run(run(args.audio))


if __name__ == "__main__":
    raise SystemExit(main())
