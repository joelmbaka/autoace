import httpx
import pytest

from autoace_backend.emotion_client import EmotionServiceClient


def response(payload, status=200):
    return httpx.Response(status, json=payload, request=httpx.Request("POST", "https://emotion.test"))


def valid_payload():
    return {
        "mode": "segments",
        "segment_count": 1,
        "customer_speech_seconds": 1.0,
        "segments": [{
            "index": 1, "start": 0.0, "end": 1.0, "duration": 1.0,
            "text": "hello", "scores": {"neutral": 0.8}, "predicted": "neutral",
            "inference_seconds": 0.1,
        }],
        "aggregate_scores": {"neutral": 0.8},
        "aggregate_predicted": "neutral",
        "negative_affect": 0.1,
        "positive_affect": 0.1,
        "neutral_affect": 0.8,
        "inference_seconds": 0.1,
    }


def test_parses_valid_segment_response():
    transport = httpx.MockTransport(lambda request: response(valid_payload()))
    with httpx.Client(transport=transport) as http:
        result = EmotionServiceClient(client=http).analyze(
            b"audio", [{"start": 0, "end": 1, "text": "hello"}]
        )
    assert result.negative_affect == 0.1
    assert result.segment_count == 1


@pytest.mark.parametrize("payload,match", [
    ({"error": "decode failed"}, "provider error"),
    ({"mode": "segments"}, "validation failed"),
])
def test_clear_provider_and_shape_errors(payload, match):
    transport = httpx.MockTransport(lambda request: response(payload))
    with httpx.Client(transport=transport) as http:
        with pytest.raises(RuntimeError, match=match):
            EmotionServiceClient(client=http).analyze(
                b"audio", [{"start": 0, "end": 1, "text": "hello"}]
            )


def test_http_error_is_clear():
    transport = httpx.MockTransport(lambda request: response({"detail": "down"}, 503))
    with httpx.Client(transport=transport) as http:
        with pytest.raises(RuntimeError, match="503"):
            EmotionServiceClient(client=http).analyze(
                b"audio", [{"start": 0, "end": 1, "text": "hello"}]
            )
