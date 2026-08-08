from autoace_backend.scribe_diagnostics import (
    ScribeToken,
    _find_overlap_intervals,
    _max_interword_gap,
    _parse_tokens,
)


def test_parse_tokens_preserves_audio_events() -> None:
    tokens = _parse_tokens(
        [
            {
                "text": "Hello",
                "type": "word",
                "start": 0.0,
                "end": 0.4,
                "speaker_id": "speaker_0",
            },
            {
                "text": "(static)",
                "type": "audio_event",
                "start": 0.4,
                "end": 1.0,
            },
        ]
    )
    assert len(tokens) == 2
    assert tokens[1].type == "audio_event"
    assert tokens[1].text == "(static)"


def test_overlap_requires_different_speakers() -> None:
    words = [
        ScribeToken(
            text="hello",
            type="word",
            start=0.0,
            end=0.8,
            speaker_id="speaker_0",
        ),
        ScribeToken(
            text="yes",
            type="word",
            start=0.5,
            end=1.0,
            speaker_id="speaker_1",
        ),
    ]
    overlaps = _find_overlap_intervals(words, min_seconds=0.05)
    assert len(overlaps) == 1
    assert overlaps[0].duration == 0.3


def test_max_interword_gap() -> None:
    words = [
        ScribeToken(text="a", type="word", start=0.0, end=0.5),
        ScribeToken(text="b", type="word", start=1.0, end=1.2),
        ScribeToken(text="c", type="word", start=4.0, end=4.2),
    ]
    assert _max_interword_gap(words) == 2.8
