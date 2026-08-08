import io
import json
import zipfile

import pytest

from autoace_backend.schemas import AudioAnalysis
from autoace_backend.services import prepare_batch


def manifest(rows: list[tuple[str, dict | None]]) -> bytes:
    lines = ["name,result_json"]
    for name, result in rows:
        payload = "" if result is None else json.dumps(result, separators=(",", ":"))
        escaped = payload.replace('"', '""')
        lines.append(f'{name},"{escaped}"')
    return ("\n".join(lines) + "\n").encode()


def expected() -> dict:
    return AudioAnalysis(
        emotional_tone="neutral",
        emotional_intensity="low",
        background_noise_present=False,
        background_noise_type="",
        background_noise_severity="none",
        audio_quality="clear",
        speaker_overlap_present=False,
        long_silence_present=False,
        confidence=0.8,
    ).model_dump()


def test_folder_batch_matches_manifest_and_keeps_labels_separate() -> None:
    batch = prepare_batch(
        [
            ("labels.csv", manifest([("call_001.ogg", expected())])),
            ("call_001.ogg", b"fake-ogg"),
        ]
    )

    assert len(batch.items) == 1
    assert batch.items[0].name == "call_001.ogg"
    assert batch.items[0].data == b"fake-ogg"
    assert batch.items[0].expected is not None
    assert batch.items[0].preflight_error is None
    assert batch.labeled_count == 1


def test_missing_manifest_audio_is_isolated_as_file_error() -> None:
    batch = prepare_batch(
        [("labels.csv", manifest([("missing.ogg", None), ("present.ogg", None)])), ("present.ogg", b"audio")]
    )

    assert batch.items[0].preflight_error is not None
    assert batch.items[1].preflight_error is None


def test_unmatched_audio_is_reported_and_ignored() -> None:
    batch = prepare_batch(
        [
            ("labels.csv", manifest([("listed.ogg", None)])),
            ("listed.ogg", b"audio"),
            ("extra.ogg", b"audio"),
        ]
    )

    assert len(batch.items) == 1
    assert any("extra.ogg" in warning for warning in batch.warnings)


def test_zip_batch_accepts_single_wrapper_folder() -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("evaluation_batch/labels.csv", manifest([("call_001.ogg", None)]))
        zipped.writestr("evaluation_batch/call_001.ogg", b"audio")

    batch = prepare_batch([("evaluation_batch.zip", archive.getvalue())])
    assert [item.name for item in batch.items] == ["call_001.ogg"]
    assert batch.items[0].preflight_error is None


def test_manifest_requires_result_json_column() -> None:
    with pytest.raises(ValueError, match="result_json"):
        prepare_batch([("labels.csv", b"name\ncall_001.ogg\n"), ("call_001.ogg", b"audio")])
