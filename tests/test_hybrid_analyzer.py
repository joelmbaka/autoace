import pytest

from autoace_backend.hybrid_analyzer import SemanticEvidence, fuse_acoustic_fields, fuse_tone
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
    (0.05, {}, ("neutral", "low")),
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
