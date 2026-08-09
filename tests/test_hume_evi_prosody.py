from scripts.evaluate_hume_evi_prosody import _session_settings_payload


def test_session_settings_uses_current_hume_audio_schema() -> None:
    audio = _session_settings_payload()["audio"]

    assert audio == {
        "encoding": "linear16",
        "sample_rate": 16_000,
        "channels": 1,
    }
    assert "format" not in audio
