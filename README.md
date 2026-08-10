# AutoAce Voice Tone & Background Noise Trial

Hosted batch-analysis implementation for the AutoAce AI technical trial. The system accepts the required evaluation folder/ZIP, validates `labels.csv`, analyzes each valid audio file independently, streams progress to a browser dashboard, and exports predictions as CSV or JSON using the required schema.

## Architecture

```text
Evaluator browser (Next.js)
        |
        | login + multipart batch
        v
FastAPI / Vercel (`HybridAnalyzer`)
        |
        |-- validate ZIP/folder + CSV manifest
        |-- isolate malformed/missing files
        |-- bounded concurrent analysis (default 3)
        |-- Scribe transcript, diarization, events, timing
        |-- customer-role inference and customer-only segments
        |-- Modal emotion2vec customer acoustic affect
        |-- Gemini semantic and selected raw-audio evidence
        |-- deterministic tone and acoustic fusion
        v
Pydantic-validated AutoAce JSON
        |
        |-- streamed per-file results
        |-- optional post-prediction comparison to result_json
        v
Results table + CSV/JSON downloads
```

`result_json` is used only after a prediction exists for optional validation. Labels are never used by customer-role inference, provider prompts, thresholds, or any other inference step. `GeminiAnalyzer` remains available as the documented baseline.

## Required output

Each successful file returns exactly:

- `emotional_tone`: `neutral | satisfied | frustrated | upset | distressed`
- `emotional_intensity`: `low | medium | high`
- `background_noise_present`: boolean
- `background_noise_type`: string (empty when no noise is present)
- `background_noise_severity`: `none | low | medium | high`
- `audio_quality`: `clear | slightly_impaired | severely_impaired`
- `speaker_overlap_present`: boolean
- `long_silence_present`: boolean
- `confidence`: `0.0..1.0`

## Batch shape

The main workflow follows the trial contract exactly:

```text
evaluation_batch/
  call_001.ogg
  call_002.ogg
  call_003.ogg
  labels.csv
```

The manifest must contain `name` and `result_json`. `result_json` may be blank for hidden/unlabeled evaluation data. Missing files, unsupported extensions, and unmatched files are reported without failing valid calls in the same batch.

Supported audio extensions: `.ogg`, `.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.webm`.

## Local setup

### Backend

Python 3.12+ and `uv` are expected.

```bash
uv sync --group dev
```

Keep the existing local `.env` if it already contains the Gemini key, or start from:

```bash
cp .env.example .env
```

Required for inference:

```text
GEMINI_API_KEY=...
ELEVENLABS_API_KEY=...
EMOTION_SERVICE_URL=https://mbakajoe26--autoace-emotion-service-emotionmodel-predict.modal.run
```

Local login defaults are `evaluator` / `autoace-local` only when the corresponding environment variables are omitted. Deployed environments must set strong values for:

```text
DASHBOARD_USERNAME
DASHBOARD_PASSWORD
AUTH_SECRET
```

Start FastAPI:

```bash
uv run uvicorn autoace_backend.app:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Analyze one file from the CLI:

```bash
uv run python main.py call_003.ogg
```

Run backend tests:

```bash
uv run pytest
```

### Dashboard

```bash
cd webapp
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000` and sign in. `NEXT_PUBLIC_API_URL` defaults to `http://localhost:8000` when omitted.

Before deployment/build verification:

```bash
npm run lint
npm run build
```

## Blind hybrid evaluation

Prediction and label comparison are deliberately separate. Write blind predictions first:

```bash
uv run python scripts/evaluate_hybrid.py blind /path/to/audio_folder \
  artifacts/hybrid/blind_predictions.json
```

Only after that artifact exists, compare it with a manifest:

```bash
uv run python scripts/evaluate_hybrid.py compare \
  artifacts/hybrid/blind_predictions.json /path/to/labels.csv \
  artifacts/hybrid/labeled_comparison.json
```

Local recordings, labels, `.env`, and generated artifacts are ignored by Git. Sanitized aggregate results are published under `docs/results/`.

## Baseline evaluation

The retained Gemini-only baseline can be evaluated against a folder or ZIP:

```bash
uv run python scripts/evaluate_batch.py /path/to/evaluation_batch \
  --output-dir artifacts/evaluation \
  --concurrency 3
```

This writes:

- `artifacts/evaluation/predictions.json`
- `artifacts/evaluation/predictions.csv`
- `artifacts/evaluation/run_details.json`

When labels are present, `run_details.json` includes per-field accuracy and an emotional-tone confusion matrix. Model confidence is reported but is not counted as a ground-truth exact-field match.

## Hybrid and Modal architecture

Scribe supplies the full transcript, word timestamps, speaker IDs, audio events, overlap intervals, and maximum interword gap. Role-based selection isolates the customer without labels. Customer timestamps and audio are sent to the configured Modal endpoint, where a preloaded emotion2vec model returns duration-weighted negative, positive, and neutral affect. Gemini returns structured customer-semantic evidence and a raw-audio baseline. Named deterministic rules fuse these signals, preventing one general-purpose model from controlling every output field.

The Modal service is defined in `modal_service/emotion_service.py`. It decodes audio with ffmpeg, clips only supplied customer segments, runs the volume-mounted emotion2vec model, and aggregates scores by speech duration. Deploy it separately with Modal and configure its HTTPS endpoint through `EMOTION_SERVICE_URL`.

## Model configuration

Defaults:

```text
GEMINI_MODEL=gemini-3.6-flash
GEMINI_THINKING_LEVEL=low
BATCH_CONCURRENCY=3
```

The production endpoint uses `HybridAnalyzer`. Affect, persistent-noise, and abnormal-dead-air thresholds are named in `autoace_backend/hybrid_analyzer.py` and covered by unit tests.

## Cost constraint

The trial ceiling is **$0.003 per audio minute**. Do not report the free-tier invoice (`$0`) as production cost. Use measured token usage from `run_details.json` with the provider's paid production rates and audio duration.

For Gemini 3.6 Flash, the hosted interactive API and the lower-cost Batch/Flex consumption modes have different token prices. If 3.6 Flash is retained after accuracy testing, the production cost analysis should use the batch-oriented consumption mode appropriate to this offline workload and verify that total input + output/thinking cost remains below the ceiling. A lower-cost audio-capable model can be selected with `GEMINI_MODEL` if validation shows an acceptable accuracy/cost tradeoff.

## Data handling

Production-call audio is confidential. The application does not persist uploaded audio, transcripts, or labels to a database; request data is held only for the request lifetime. Audio is sent to configured Gemini, ElevenLabs, and Modal services, so production must use AutoAce-approved accounts, regions, tiers, and retention/training controls. Local recordings, manifests, credentials, and raw artifacts are excluded from Git.

## Vercel deployment

### Backend project

Create a Vercel project with repository root `/`. `api/index.py` exposes the FastAPI app and `vercel.json` rewrites requests to it.

Set:

```text
GEMINI_API_KEY
GEMINI_MODEL
GEMINI_THINKING_LEVEL
ELEVENLABS_API_KEY
ELEVENLABS_SCRIBE_MODEL
EMOTION_SERVICE_URL
DASHBOARD_USERNAME
DASHBOARD_PASSWORD
AUTH_SECRET
ALLOWED_ORIGINS=https://<dashboard-domain>
BATCH_CONCURRENCY=3
```

### Dashboard project

Create a second Vercel project with Root Directory `webapp` and set:

```text
NEXT_PUBLIC_API_URL=https://<backend-domain>
```

After the dashboard domain is known, add it to backend `ALLOWED_ORIGINS` and redeploy the backend.

### Upload-size note

The implementation validates both folder selections and ZIP archives at the API boundary. Vercel imposes request-body limits on Functions; if the evaluation batch exceeds the active plan's request limit, the production hardening path is direct object-storage upload followed by server-side analysis. The classifier and batch-validation layers are intentionally isolated so that storage transport can be swapped without changing inference logic.

## API

- `GET /health` - public health/config summary
- `POST /api/v1/auth/login` - evaluator login
- `POST /api/v1/batches/analyze` - authenticated ZIP/folder batch analysis; streams NDJSON events

The batch endpoint emits `batch_started`, `file_completed`, `file_failed`, and `batch_completed` events. Each file is an independent pipeline unit: malformed input or a provider failure is reported for that file without aborting the rest of the batch.
