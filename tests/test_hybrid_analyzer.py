import pytest

from autoace_backend.hybrid_analyzer import (
    SemanticEvidence,
    fuse_acoustic_fields,
    fuse_tone,
    stabilize_semantic_evidence,
)
from autoace_backend.schemas import AudioAnalysis
from autoace_backend.scribe_diagnostics import ScribeDiagnosticsResult, ScribeToken


def evidence(**overrides):
    values = {
        "request_resolved": False,
        "positive_ending": False,
        "complaint_or_blockage": False,
        "repeated_failed_contact": False,
        "distress_or_panic": False,
        "strong_anger_or_agitation": False,
        "semantic_intensity": "low",
    }
    values.update(overrides)
    return SemanticEvidence(**values)


@pytest.mark.parametrize("negative,semantic,expected", [
    (0.45, {"strong_anger_or_agitation": True}, ("upset", "high")),
    (0.16, {"complaint_or_blockage": True, "semantic_intensity": "medium"}, ("frustrated", "medium")),
    (0.05, {"request_resolved": True, "positive_ending": True}, ("satisfied", "medium")),
    (0.05, {}, ("neutral", "medium")),
    (0.05, {"distress_or_panic": True}, ("distressed", "high")),
])
def test_tone_fusion(negative, semantic, expected):
    assert fuse_tone(negative, evidence(**semantic)) == expected


def baseline(**overrides):
    values = {
        "emotional_tone": "neutral", "emotional_intensity": "low",
        "background_noise_present": True, "background_noise_type": "hum",
        "background_noise_severity": "medium", "audio_quality": "clear",
        "speaker_overlap_present": False, "long_silence_present": False,
        "confidence": 0.8,
    }
    values.update(overrides)
    return AudioAnalysis(**values)


def token(text, token_type, start=None, end=None, speaker=None):
    return ScribeToken(text=text, type=token_type, start=start, end=end, speaker_id=speaker)


def scribe(*, speakers=("agent", "customer"), events=(), words=()):
    return ScribeDiagnosticsResult(
        name="call.ogg", model="scribe", request_seconds=1, transcript="hello",
        speaker_ids=list(speakers),
        spoken_words=list(words),
        audio_events=list(events),
    )


def test_two_speakers_and_no_events_enforces_noise_consistency():
    result = fuse_acoustic_fields(baseline(), scribe(), agent_speaker="agent", customer_speaker="customer")
    assert result["background_noise_present"] is False
    assert result["background_noise_type"] == ""
    assert result["background_noise_severity"] == "none"


def test_static_event_maps_to_sharp_static():
    result = fuse_acoustic_fields(baseline(), scribe(events=(token("[static]", "audio_event", 1, 1.5),)), agent_speaker="agent", customer_speaker="customer")
    assert result["background_noise_present"] is True
    assert result["background_noise_type"] == "sharp static"


def test_extra_nonconversational_speaker_maps_to_tv():
    result = fuse_acoustic_fields(baseline(), scribe(speakers=("agent", "customer", "background")), agent_speaker="agent", customer_speaker="customer")
    assert result["background_noise_type"] == "TV"


def test_low_negative_language_preference_is_neutral_medium():
    assert fuse_tone(0.05, evidence(semantic_intensity="medium")) == ("neutral", "medium")


@pytest.mark.parametrize("context", ["operational correction", "profanity in isolation"])
def test_keyword_context_without_acoustic_negative_is_not_frustrated(context):
    # The semantic prompt owns the textual distinction; fusion still gates even if it errs.
    assert fuse_tone(0.05, evidence(complaint_or_blockage=True, semantic_intensity="medium")) == ("neutral", "medium")


def test_persistent_static_is_medium_severity():
    diagnostic = scribe(events=(token("[static]", "audio_event", 1, 3.1),))
    result = fuse_acoustic_fields(baseline(), diagnostic, agent_speaker="agent", customer_speaker="customer")
    assert result["background_noise_severity"] == "medium"


def test_persistent_background_speech_is_medium_without_forcing_overlap():
    words = (
        token("broadcast", "word", 1, 2.2, "background"),
        token("continues", "word", 2.2, 3.2, "background"),
    )
    diagnostic = scribe(speakers=("agent", "customer", "background"), words=words)
    result = fuse_acoustic_fields(baseline(), diagnostic, agent_speaker="agent", customer_speaker="customer")
    assert result["background_noise_severity"] == "medium"
    assert result["speaker_overlap_present"] is False


def test_low_semantic_intensity_with_neutral_affect_defaults_to_medium():
    assert fuse_tone(0.02, evidence(semantic_intensity="low")) == ("neutral", "medium")


@pytest.mark.parametrize(
    "description",
    ["TV", "television", "background speech", "background talk", "radio speech", "broadcast speech"],
)
def test_broadcast_descriptions_canonicalize_to_tv(description):
    diagnostic = scribe(events=(token("[beep]", "audio_event", 0, 0.2),))
    result = fuse_acoustic_fields(
        baseline(background_noise_type=description, background_noise_severity="low"),
        diagnostic,
        agent_speaker="agent",
        customer_speaker="customer",
    )
    assert result["background_noise_type"] == "TV"


def test_sustained_broadcast_description_has_medium_severity():
    diagnostic = scribe(events=(token("[beep]", "audio_event", 0, 0.2),))
    result = fuse_acoustic_fields(
        baseline(
            background_noise_type="continuous background speech",
            background_noise_severity="low",
        ),
        diagnostic,
        agent_speaker="agent",
        customer_speaker="customer",
    )
    assert result["background_noise_type"] == "TV"
    assert result["background_noise_severity"] == "medium"


def test_intelligible_call_is_not_downgraded_by_one_quality_sample():
    words = (
        token("hello", "word", 1, 1.3, "agent"),
        token("thanks", "word", 2, 2.3, "customer"),
    )
    diagnostic = scribe(words=words)
    result = fuse_acoustic_fields(
        baseline(audio_quality="slightly_impaired"),
        diagnostic,
        agent_speaker="agent",
        customer_speaker="customer",
    )
    assert result["audio_quality"] == "clear"


def test_accepted_next_step_and_appreciative_ending_can_be_satisfied():
    stabilized = stabilize_semantic_evidence(
        0.01,
        evidence(complaint_or_blockage=True),
        transcript="I am transferring you to an advisor now. Thank you.",
        customer_transcript="Okay. Yes, please do. Thank you.",
    )
    assert stabilized.request_resolved is True
    assert stabilized.positive_ending is True
    assert fuse_tone(0.01, stabilized) == ("satisfied", "medium")


def test_polite_ending_without_agreed_outcome_remains_neutral():
    stabilized = stabilize_semantic_evidence(
        0.01,
        evidence(),
        transcript="Could you check that for me? Thank you.",
        customer_transcript="Thank you.",
    )
    assert stabilized.request_resolved is False
    assert stabilized.positive_ending is False
    assert fuse_tone(0.01, stabilized) == ("neutral", "medium")


def test_overlap_fusion_behavior_is_unchanged():
    words = (
        token("hello", "word", 1, 2, "agent"),
        token("yes", "word", 1.5, 2.2, "customer"),
    )
    diagnostic = scribe(words=words)
    diagnostic.overlap_intervals = [
        {
            "start": 1.5,
            "end": 2.0,
            "duration": 0.5,
            "speaker_a": "agent",
            "speaker_b": "customer",
            "text_a": "hello",
            "text_b": "yes",
        }
    ]
    result = fuse_acoustic_fields(
        baseline(speaker_overlap_present=False),
        diagnostic,
        agent_speaker="agent",
        customer_speaker="customer",
    )
    assert result["speaker_overlap_present"] is True
