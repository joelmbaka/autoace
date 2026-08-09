import asyncio
import math
import struct

import pytest

from scripts.evaluate_hume_evi_prosody import (
    SegmentWaitError,
    TRAILING_SILENCE_MS,
    _evi_query,
    _infer_customer_speaker,
    _pcm_statistics,
    _scores_from_user_message,
    _select_audio_paths,
    _session_settings_payload,
    _transcript_delta,
    _wait_for_final_user_message,
)


def _words_for_speaker(
    speaker: str, start: float, text: str
) -> list[dict[str, object]]:
    return [
        {
            "speaker_id": speaker,
            "start": start + index * 0.2,
            "end": start + index * 0.2 + 0.1,
            "text": word,
        }
        for index, word in enumerate(text.split())
    ]


def test_session_settings_uses_current_hume_audio_schema() -> None:
    audio = _session_settings_payload()["audio"]

    assert audio == {
        "encoding": "linear16",
        "sample_rate": 16_000,
        "channels": 1,
    }
    assert "format" not in audio


def test_evi_query_enables_verbose_transcription() -> None:
    assert _evi_query("test-key")["verbose_transcription"] == "true"


def test_trailing_silence_allows_turn_finalization_margin() -> None:
    assert TRAILING_SILENCE_MS >= 1500


def test_customer_selection_with_one_agent_and_one_customer() -> None:
    words = [
        *_words_for_speaker("agent", 1.0, "Hi I'm Erica how can I help"),
        *_words_for_speaker("customer", 4.0, "I need an appointment"),
    ]

    customer, diagnostics = _infer_customer_speaker(words)

    assert customer == "customer"
    assert diagnostics["customer_speaker"] == "customer"
    assert len(diagnostics["candidate_speakers"]) == 1


def test_customer_selection_prefers_earliest_post_greeting_responder() -> None:
    words = [
        *_words_for_speaker("background", 0.0, "Tonight on television"),
        *_words_for_speaker("agent", 2.0, "Hi I'm Erica how can I help"),
        *_words_for_speaker("customer", 5.0, "I need service please"),
        *_words_for_speaker("background", 8.0, "More television speech continues here"),
    ]

    customer, diagnostics = _infer_customer_speaker(words)

    assert customer == "customer"
    assert {candidate["speaker_id"] for candidate in diagnostics["candidate_speakers"]} == {
        "background",
        "customer",
    }
    assert "earliest non-agent spoken turn" in diagnostics["customer_selection_reason"]
    background = next(
        candidate
        for candidate in diagnostics["candidate_speakers"]
        if candidate["speaker_id"] == "background"
    )
    assert background["first_word_start"] == 0.0
    assert background["first_post_greeting_turn"]["start"] == 8.0
    assert background["transcript"] == (
        "Tonight on television More television speech continues here"
    )


def test_pcm_statistics_for_known_linear16_samples() -> None:
    stats = _pcm_statistics(struct.pack("<hhh", 0, 1000, -1000))

    expected_rms = math.sqrt((0**2 + 1000**2 + (-1000) ** 2) / 3)
    assert stats["pcm_byte_length"] == 6
    assert stats["peak_absolute_sample_amplitude"] == 1000
    assert stats["rms_amplitude"] == pytest.approx(expected_rms)
    assert stats["rms_dbfs"] == pytest.approx(
        20 * math.log10(expected_rms / 32768)
    )


def test_cumulative_hume_transcript_delta() -> None:
    assert _transcript_delta("", "First turn") == "First turn"
    assert _transcript_delta("First turn", "First turn Second turn") == "Second turn"
    assert _transcript_delta("First turn", "Unrelated transcript") is None


def test_final_user_message_ignores_interim_messages() -> None:
    async def exercise() -> None:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        await queue.put(
            {"type": "user_message", "interim": True, "message": {"content": "hi"}}
        )
        final = {
            "type": "user_message",
            "interim": False,
            "message": {"content": "hi there"},
        }
        await queue.put(final)

        message, interim_count, final_count = await _wait_for_final_user_message(
            queue, 0.1
        )

        assert message == final
        assert interim_count == 1
        assert final_count == 1

    asyncio.run(exercise())


def test_later_segment_succeeds_after_timeout_without_stale_contamination() -> None:
    async def exercise() -> None:
        first_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        await first_queue.put(
            {
                "_event_sequence": 1,
                "type": "user_message",
                "interim": True,
                "message": {"content": "old segment"},
            }
        )
        with pytest.raises(SegmentWaitError) as captured:
            await _wait_for_final_user_message(first_queue, 0.001)
        assert captured.value.status == "interim_without_final"
        assert captured.value.interim_count == 1

        second_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        await second_queue.put(
            {
                "_event_sequence": 2,
                "type": "user_message",
                "interim": False,
                "message": {"content": "old segment"},
                "models": {"prosody": {"scores": {"Anger": 0.9}}},
            }
        )
        await second_queue.put(
            {
                "_event_sequence": 3,
                "type": "user_message",
                "interim": True,
                "message": {"content": "new segment"},
            }
        )
        later_final = {
            "_event_sequence": 4,
            "type": "user_message",
            "interim": False,
            "message": {"content": "new segment"},
            "models": {"prosody": {"scores": {"Calmness": 0.8}}},
        }
        await second_queue.put(later_final)

        message, interim_count, final_count = await _wait_for_final_user_message(
            second_queue,
            0.1,
            after_sequence=1,
            unresolved_interim_transcripts=captured.value.interim_transcripts,
        )

        assert message == later_final
        assert interim_count == 1
        assert final_count == 2
        assert _scores_from_user_message(message) == {"Calmness": 0.8}

    asyncio.run(exercise())


def test_no_speech_timeout_retains_segment_status() -> None:
    async def exercise() -> None:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        with pytest.raises(SegmentWaitError) as captured:
            await _wait_for_final_user_message(queue, 0.001)
        assert captured.value.status == "no_speech_detected"
        assert captured.value.interim_count == 0
        assert captured.value.final_count == 0

    asyncio.run(exercise())


def test_only_filters_input_files(tmp_path) -> None:
    for name in ("call_001.ogg", "call_002.ogg"):
        (tmp_path / name).touch()

    assert [path.name for path in _select_audio_paths(tmp_path, "call_001.ogg")] == [
        "call_001.ogg"
    ]
    assert [path.name for path in _select_audio_paths(tmp_path, None)] == [
        "call_001.ogg",
        "call_002.ogg",
    ]


def test_hume_error_event_raises_sanitized_runtime_error() -> None:
    async def exercise() -> None:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        await queue.put(
            {
                "type": "error",
                "code": "E1234",
                "message": "Safe provider message",
                "api_key": "must-not-appear",
            }
        )

        with pytest.raises(
            RuntimeError, match="^Hume EVI error E1234: Safe provider message$"
        ) as captured:
            await _wait_for_final_user_message(queue, 0.1)
        assert "must-not-appear" not in str(captured.value)

    asyncio.run(exercise())
