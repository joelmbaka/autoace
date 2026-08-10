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

        started = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".input") as source:
            source.write(audio_bytes)
            source.flush()

            with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        source.name,
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-c:a",
                        "pcm_s16le",
                        wav.name,
                    ],
                    check=True,
                )

                response = self.model.generate(
                    wav.name,
                    granularity="utterance",
                    extract_embedding=False,
                )

        result = response[0]

        labels = result["labels"]
        scores = result["scores"]

        normalized = {}
        for label, score in zip(labels, scores):
            english = label.split("/")[-1]
            normalized[english] = float(score)

        elapsed = time.perf_counter() - started

        return {
            "scores": normalized,
            "predicted": max(normalized, key=normalized.get),
            "inference_seconds": elapsed,
        }
