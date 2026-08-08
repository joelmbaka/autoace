from scripts.evaluate_hume_customer_emotion import _job_status, _speaker_summary


def test_job_status_reads_nested_state() -> None:
    assert _job_status({"state": {"status": "COMPLETED"}}) == "COMPLETED"
    assert _job_status({"state": {"status": "FAILED"}}) == "FAILED"


def test_speaker_summary_identifies_agent_and_customer() -> None:
    entry = {
        "models": {
            "prosody": {
                "grouped_predictions": [
                    {
                        "id": "speaker_agent",
                        "predictions": [
                            {
                                "text": "Hi, I'm Erica from Toyota of Braintree. How can I help?",
                                "time": {"begin": 0.0, "end": 3.0},
                                "emotions": [
                                    {"name": "Calmness", "score": 0.7},
                                    {"name": "Anger", "score": 0.05},
                                ],
                            }
                        ],
                    },
                    {
                        "id": "speaker_customer",
                        "predictions": [
                            {
                                "text": "Are you a real person? Hello?",
                                "time": {"begin": 3.1, "end": 7.0},
                                "emotions": [
                                    {"name": "Anger", "score": 0.8},
                                    {"name": "Calmness", "score": 0.1},
                                ],
                            }
                        ],
                    },
                ]
            }
        }
    }

    result = _speaker_summary(entry)
    assert result["agent_speaker_id"] == "speaker_agent"
    assert result["customer_speaker_id"] == "speaker_customer"
    assert "how can i help" in result["agent_role_phrase_hits"]
    assert result["customer_top_expressions"][0][0] == "Anger"
