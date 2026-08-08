# Hybrid emotion experiment

This experiment is intentionally isolated from the working Gemini batch classifier. It tests whether a specialized speech-to-text stage plus a strong text-reasoning model improves the two weakest Gemini fields: customer emotional tone and intensity.

## Pipeline

```text
raw call audio
  -> Groq Whisper Large V3 (transcript + segment timestamps)
  -> NVIDIA Llama-3.3-Nemotron-Super-49B-v1.5
  -> emotional_tone + emotional_intensity + confidence
```

The experiment does **not** ask the text model to infer acoustic fields such as background-noise type, audio quality, speaker overlap, or long silence.

## Credentials

Add these to the local `.env` only:

```text
GROQ_API_KEY=...
NVIDIA_API_KEY=...
```

Optional overrides:

```text
GROQ_WHISPER_MODEL=whisper-large-v3
NVIDIA_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
GROQ_TIMEOUT_SECONDS=180
NVIDIA_TIMEOUT_SECONDS=180
```

## Blind evaluation

With the three supplied `.ogg` files and `labels.csv` present locally in the repository root:

```bash
uv sync --group dev
uv run python scripts/evaluate_hybrid_emotion.py . \
  --labels labels.csv \
  --output-dir artifacts/hybrid_emotion \
  --concurrency 1
```

The script deliberately performs inference for every audio file before opening `labels.csv`. It writes the blind predictions and transcripts to disk first, then begins the comparison phase.

Generated local-only artifacts:

- `artifacts/hybrid_emotion/blind_predictions.json`
- `artifacts/hybrid_emotion/transcripts.json`
- `artifacts/hybrid_emotion/evaluation.json`

`artifacts/`, `labels.csv`, and new `.ogg` files are ignored by Git.

## Evaluation decision

Do not integrate this pipeline into the dashboard merely because it runs successfully. Compare its tone/intensity metrics against the existing Gemini baseline first. Only change the production inference path if the labeled-set improvement is material and the eventual total production cost remains inside the trial ceiling.
