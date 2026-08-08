# AutoAce Voice Tone & Background Noise Trial

Hosted batch-analysis implementation for the AutoAce AI technical trial. The system accepts the required evaluation folder/ZIP, validates `labels.csv`, analyzes each valid audio file independently, streams progress to a browser dashboard, and exports predictions as CSV or JSON using the required schema.

## Architecture

```text
Evaluator browser (Next.js)
        |
        | login + multipart batch
        v
FastAPI / Vercel
        |
        |-- validate ZIP/folder + CSV manifest
        |-- isolate malformed/missing files
        |-- bounded concurrent analysis (default 3)
        v
Gemini raw-audio classifier
        |
        v
Pydantic-validated AutoAce JSON
        |
        |-- streamed per-file results
        |-- optional post-prediction comparison to result_json
        v
Results table + CSV/JSON downloads
```

`result_json` is parsed only after batch ingestion for validation. It is never included in the Gemini prompt or model request.

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

## Reproducible labeled-set evaluation

Run the exact same classifier used by the API against a folder or ZIP:

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

## Model configuration

Defaults:

```text
GEMINI_MODEL=gemini-3.6-flash
GEMINI_THINKING_LEVEL=low
BATCH_CONCURRENCY=3
```

The prompt contains the trial's operational definitions and explicitly separates customer emotion, non-speech background noise, technical audio quality, overlap, and dead air. Model and thinking level are environment-configurable so candidate models can be compared without changing application code.

## Cost constraint

The trial ceiling is **$0.003 per audio minute**. Do not report the free-tier invoice (`$0`) as production cost. Use measured token usage from `run_details.json` with the provider's paid production rates and audio duration.

For Gemini 3.6 Flash, the hosted interactive API and the lower-cost Batch/Flex consumption modes have different token prices. If 3.6 Flash is retained after accuracy testing, the production cost analysis should use the batch-oriented consumption mode appropriate to this offline workload and verify that total input + output/thinking cost remains below the ceiling. A lower-cost audio-capable model can be selected with `GEMINI_MODEL` if validation shows an acceptable accuracy/cost tradeoff.

## Data handling

Production-call audio is confidential. The service does not persist uploaded audio or labels to a database; data is held only for the request lifetime. The current classifier sends raw audio to the configured Gemini API, so the final memo/deployment must disclose the provider and its retention/training policy and use an AutoAce-approved processing tier/provider.

## Vercel deployment

### Backend project

Create a Vercel project with repository root `/`. `api/index.py` exposes the FastAPI app and `vercel.json` rewrites requests to it.

Set:

```text
GEMINI_API_KEY
GEMINI_MODEL
GEMINI_THINKING_LEVEL
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

The batch endpoint emits `batch_started`, `file_completed`, `file_failed`, and `batch_completed` events. A single file failure is isolated from the rest of the batch.
