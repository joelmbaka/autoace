from autoace_backend.hybrid_emotion import EmotionPrediction, _extract_json_object
from scripts.evaluate_hybrid_emotion import _classification_metrics


def test_extract_json_object_accepts_plain_json():
    payload = _extract_json_object(
        '{"emotional_tone":"neutral","emotional_intensity":"low","confidence":0.7}'
    )
    result = EmotionPrediction.model_validate(payload)
    assert result.emotional_tone == "neutral"
    assert result.emotional_intensity == "low"
    assert result.confidence == 0.7


def test_extract_json_object_accepts_fenced_json():
    payload = _extract_json_object(
        """```json
        {"emotional_tone":"satisfied","emotional_intensity":"medium","confidence":0.8}
        ```"""
    )
    result = EmotionPrediction.model_validate(payload)
    assert result.emotional_tone == "satisfied"
    assert result.emotional_intensity == "medium"


def test_extract_json_object_recovers_embedded_object():
    payload = _extract_json_object(
        'Result: {"emotional_tone":"upset","emotional_intensity":"high","confidence":0.6}'
    )
    result = EmotionPrediction.model_validate(payload)
    assert result.emotional_tone == "upset"
    assert result.emotional_intensity == "high"


def test_classification_metrics_are_correct():
    metrics = _classification_metrics(
        ["neutral", "satisfied", "upset"],
        ["neutral", "neutral", "upset"],
        ["neutral", "satisfied", "upset"],
    )
    assert metrics["accuracy"] == 2 / 3
    assert metrics["confusion_matrix"]["neutral"]["neutral"] == 1
    assert metrics["confusion_matrix"]["satisfied"]["neutral"] == 1
    assert metrics["confusion_matrix"]["upset"]["upset"] == 1
    assert metrics["per_class"]["upset"]["f1"] == 1.0
