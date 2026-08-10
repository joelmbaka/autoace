import modal

app = modal.App("autoace-emotion-service")

volume = modal.Volume.from_name("autoace-models")

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi[standard]",
        "funasr",
        "modelscope",
        "torch",
        "torchaudio",
        "soundfile",
        "numpy",
    )
)

MODEL_PATH = "/models/emotion2vec-plus-base"
SAMPLE_RATE = 16000


@app.cls(
    image=image,
    volumes={"/models": volume},
    cpu=2,
    memory=4096,
    scaledown_window=300,
)
class EmotionModel:
    @modal.enter()
    def load(self):
        from funasr import AutoModel

        self.model = AutoModel(
            model=MODEL_PATH,
            disable_update=True,
        )

    def _normalize_response(self, response):
        result = response[0]
        scores = {}

        for label, score in zip(result["labels"], result["scores"]):
            english = label.split("/")[-1]
            scores[english] = float(score)

        return scores

    @modal.fastapi_endpoint(method="POST", docs=True)
    def predict(self, item: dict):
        import base64
        import subprocess
        import tempfile
        import time

        audio_b64 = item.get("audio_base64")
        if not audio_b64:
            return {"error": "audio_base64 is required"}

        try:
            audio_bytes = base64.b64decode(audio_b64, validate=True)
        except Exception:
            return {"error": "audio_base64 is invalid"}

        raw_segments = item.get("segments")

        if raw_segments is not None and not isinstance(raw_segments, list):
            return {"error": "segments must be a list"}

        started = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".input") as source:
            source.write(audio_bytes)
            source.flush()

            with tempfile.NamedTemporaryFile(suffix=".wav") as decoded:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel", "error",
                        "-y",
                        "-i", source.name,
                        "-ac", "1",
                        "-ar", str(SAMPLE_RATE),
                        "-c:a", "pcm_s16le",
                        decoded.name,
                    ],
                    check=True,
                )

                # Backward-compatible whole-audio inference when no segments
                # are supplied. Production AutoAce analysis supplies customer
                # speech segments.
                if not raw_segments:
                    response = self.model.generate(
                        decoded.name,
                        granularity="utterance",
                        extract_embedding=False,
                    )

                    scores = self._normalize_response(response)

                    return {
                        "mode": "whole_audio",
                        "scores": scores,
                        "predicted": max(scores, key=scores.get),
                        "negative_affect": (
                            scores.get("angry", 0.0)
                            + scores.get("disgusted", 0.0)
                            + scores.get("fearful", 0.0)
                            + scores.get("sad", 0.0)
                        ),
                        "positive_affect": scores.get("happy", 0.0),
                        "inference_seconds": time.perf_counter() - started,
                    }

                validated_segments = []

                for index, raw in enumerate(raw_segments, start=1):
                    if not isinstance(raw, dict):
                        return {"error": f"segment {index} must be an object"}

                    try:
                        start = float(raw["start"])
                        end = float(raw["end"])
                    except (KeyError, TypeError, ValueError):
                        return {
                            "error": f"segment {index} requires numeric start and end"
                        }

                    if start < 0 or end <= start:
                        return {
                            "error": f"segment {index} has invalid timestamps"
                        }

                    validated_segments.append(
                        {
                            "start": start,
                            "end": end,
                            "text": str(raw.get("text") or ""),
                        }
                    )

                segment_results = []
                weighted_scores = {}
                total_duration = 0.0

                for index, segment in enumerate(validated_segments, start=1):
                    duration = segment["end"] - segment["start"]

                    with tempfile.NamedTemporaryFile(suffix=".wav") as clip:
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-hide_banner",
                                "-loglevel", "error",
                                "-y",
                                "-ss", str(segment["start"]),
                                "-t", str(duration),
                                "-i", decoded.name,
                                "-ac", "1",
                                "-ar", str(SAMPLE_RATE),
                                "-c:a", "pcm_s16le",
                                clip.name,
                            ],
                            check=True,
                        )

                        segment_started = time.perf_counter()

                        response = self.model.generate(
                            clip.name,
                            granularity="utterance",
                            extract_embedding=False,
                        )

                        segment_elapsed = time.perf_counter() - segment_started

                    scores = self._normalize_response(response)

                    for label, score in scores.items():
                        weighted_scores[label] = (
                            weighted_scores.get(label, 0.0)
                            + score * duration
                        )

                    total_duration += duration

                    segment_results.append(
                        {
                            "index": index,
                            "start": segment["start"],
                            "end": segment["end"],
                            "duration": duration,
                            "text": segment["text"],
                            "scores": scores,
                            "predicted": max(scores, key=scores.get),
                            "inference_seconds": segment_elapsed,
                        }
                    )

                aggregate_scores = {
                    label: value / total_duration
                    for label, value in weighted_scores.items()
                }

                negative_affect = sum(
                    aggregate_scores.get(label, 0.0)
                    for label in ("angry", "disgusted", "fearful", "sad")
                )

                return {
                    "mode": "segments",
                    "segment_count": len(segment_results),
                    "customer_speech_seconds": total_duration,
                    "segments": segment_results,
                    "aggregate_scores": aggregate_scores,
                    "aggregate_predicted": max(
                        aggregate_scores,
                        key=aggregate_scores.get,
                    ),
                    "negative_affect": negative_affect,
                    "positive_affect": aggregate_scores.get("happy", 0.0),
                    "neutral_affect": aggregate_scores.get("neutral", 0.0),
                    "inference_seconds": time.perf_counter() - started,
                }
