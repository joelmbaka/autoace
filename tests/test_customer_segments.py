from autoace_backend.customer_segments import (
    build_customer_segments,
    infer_customer_speaker,
)
from autoace_backend.scribe_diagnostics import ScribeToken


def word(
    text: str,
    start: float,
    end: float,
    speaker: str,
) -> ScribeToken:
    return ScribeToken(
        text=text,
        type="word",
        start=start,
        end=end,
        speaker_id=speaker,
    )


def test_selects_customer_after_dealership_greeting():
    words = [
        word("I'm", 4.4, 4.7, "speaker_0"),
        word("Erica", 4.7, 5.1, "speaker_0"),
        word("from", 5.1, 5.3, "speaker_0"),
        word("Toyota", 5.3, 5.8, "speaker_0"),
        word("How", 6.0, 6.2, "speaker_0"),
        word("can", 6.2, 6.4, "speaker_0"),
        word("I", 6.4, 6.5, "speaker_0"),
        word("help", 6.5, 6.8, "speaker_0"),
        word("Spanish,", 9.88, 10.5, "speaker_1"),
        word("please.", 10.5, 11.779, "speaker_1"),
    ]

    customer, diagnostics = infer_customer_speaker(words)

    assert customer == "speaker_1"
    assert diagnostics["agent_speaker"] == "speaker_0"


def test_earliest_non_agent_response_wins_over_later_background_speaker():
    words = [
        word("I'm", 4.4, 4.7, "speaker_0"),
        word("Erica", 4.7, 5.1, "speaker_0"),
        word("How", 6.0, 6.2, "speaker_0"),
        word("can", 6.2, 6.4, "speaker_0"),
        word("I", 6.4, 6.5, "speaker_0"),
        word("help", 6.5, 7.0, "speaker_0"),

        word("Spanish,", 9.88, 10.5, "speaker_1"),
        word("please.", 10.5, 11.779, "speaker_1"),

        # Simulates the later TV/background speech seen in call_002.
        word("Las", 28.939, 29.2, "speaker_2"),
        word("huellas", 29.2, 29.8, "speaker_2"),
        word("del", 29.8, 30.0, "speaker_2"),
        word("objetivo", 30.0, 30.7, "speaker_2"),
    ]

    customer, diagnostics = infer_customer_speaker(words)

    assert customer == "speaker_1"
    assert len(diagnostics["candidate_speakers"]) == 2
    assert "earliest non-agent" in diagnostics["customer_selection_reason"]


def test_customer_words_merge_within_gap_and_split_after_gap():
    words = [
        word("Hello", 1.0, 1.3, "customer"),
        word("there", 1.5, 1.8, "customer"),
        word("again", 3.0, 3.4, "customer"),
        word("please", 3.6, 4.0, "customer"),
    ]

    segments = build_customer_segments(words, "customer")

    assert segments == [
        {
            "start": 1.0,
            "end": 1.8,
            "text": "Hello there",
        },
        {
            "start": 3.0,
            "end": 4.0,
            "text": "again please",
        },
    ]


def test_segment_never_grows_beyond_nine_seconds():
    words = [
        word("first", 0.0, 4.8, "customer"),
        word("second", 5.0, 9.2, "customer"),
    ]

    segments = build_customer_segments(words, "customer")

    assert len(segments) == 2
