# AutoAce Voice Tone & Background Noise Trial — Technical Memo

## Executive summary

This submission treats the problem as **offline production-call classification**, not a live telephony problem. The hosted dashboard accepts the required ZIP/folder batch and CSV manifest, validates the batch, analyzes valid recordings independently with bounded concurrency, streams progress, and exports the required `name,result_json` results as CSV or JSON.

The inference layer is intentionally isolated from transport/UI concerns so the same classifier can later be called from an internal AutoAce workflow using a call ID or recording URL rather than manual upload. The final complete authenticated production-dashboard run scored **21/24 exact discrete fields (87.5%)**, compared with the historical label-isolated local hybrid benchmark of **20/24 (83.3%)** and the Gemini-only baseline of **10/24 (41.7%)**.

## Classification contract

The implementation follows the assignment taxonomy directly and emits a Pydantic-validated schema containing:

- customer emotional tone and intensity;
- meaningful non-speech background-noise presence/type/severity;
- technical audio quality;
- meaningful simultaneous speaker overlap;
- operationally unusual dead air;
- overall confidence.

The prompt explicitly keeps background noise and technical audio quality independent and does not infer emotion from loudness alone.

## Approaches evaluated

### A. General raw-audio multimodal baseline

The first feasibility approach sent the original OGG audio directly to Gemini 3.6 Flash with a concise structured-output prompt. This established that raw OGG input, structured output, free-tier development inference, and acceptable batch-oriented latency were technically feasible.

The initial blind development-set checks exposed important weaknesses: the general prompt over-detected long silence and background conditions and did not reliably identify the customer's emotional class. This baseline was therefore **not accepted as the final prompt**.

Observed baseline examples before reading the supplied labels:

| Development clip | Audio duration | Request latency | Exact scored fields correct |
| --- | ---: | ---: | ---: |
| call_001 | ~30.95 s | 9.78 s | 0 / 8 |
| call_003 | ~171.93 s | 25.61 s | 3 / 8 |

These results are retained as an experiment rather than hidden: they motivated tighter operational definitions and demonstrate why validation is necessary before treating a general multimodal model as a production classifier.

### B. Final hybrid classifier

The production candidate separates specialist evidence instead of allowing one general-purpose model to decide every field:

- ElevenLabs Scribe supplies diarization, transcript, word timestamps, audio events, overlap timing, and silence structure.
- Role-based selection isolates customer speech without using supplied labels.
- Modal-hosted emotion2vec supplies duration-weighted acoustic affect for customer-only segments.
- Gemini supplies structured customer-semantic evidence and selected raw-audio evidence.
- Named deterministic fusion rules combine tone, intensity, persistent noise, audio quality, overlap, and dead-air evidence.

The earlier verified label-isolated local blind run produced:

| Field | Correct | Accuracy |
| --- | ---: | ---: |
| Emotional tone | 2 / 3 | 66.7% |
| Emotional intensity | 2 / 3 | 66.7% |
| Background-noise presence | 3 / 3 | 100% |
| Background-noise type | 3 / 3 | 100% |
| Background-noise severity | 3 / 3 | 100% |
| Audio quality | 3 / 3 | 100% |
| Speaker overlap | 1 / 3 | 33.3% |
| Long silence | 3 / 3 | 100% |
| **All discrete fields** | **20 / 24** | **83.3%** |

Per-call exact results were 8/8, 7/8, and 5/8. The emotional-tone confusion matrix was:

| Expected / Predicted | neutral | satisfied | upset |
| --- | ---: | ---: | ---: |
| neutral | 1 | 0 | 0 |
| satisfied | 1 | 0 | 0 |
| upset | 0 | 0 | 1 |

Reproduce the label-isolated workflow with:

```bash
uv run python scripts/evaluate_hybrid.py blind /path/to/audio_folder artifacts/hybrid/blind_predictions.json
uv run python scripts/evaluate_hybrid.py compare artifacts/hybrid/blind_predictions.json /path/to/labels.csv artifacts/hybrid/labeled_comparison.json
```

The blind artifact is written before `labels.csv` is opened. Labels are never used during customer identification, provider inference, or deterministic fusion. The three supplied calls are too small for statistically strong generalization claims and do not cover every emotional-tone class.

One semantic-prompt clarification was made after that historical run to define an accepted workable next step as resolution. Gemini's free-tier daily quota was exhausted before that local three-call run could be repeated, so the clarification is not included in the historical 20/24 metric.

### Final production-dashboard result

A newer complete authenticated dashboard run verified the frozen production implementation on all three supplied calls: **3 completed, 0 failed**, with model string `hybrid:gemini-3.6-flash+scribe+emotion2vec`.

| Field | Correct | Accuracy |
| --- | ---: | ---: |
| Emotional tone | 3 / 3 | 100% |
| Emotional intensity | 3 / 3 | 100% |
| Background-noise presence | 3 / 3 | 100% |
| Background-noise type | 3 / 3 | 100% |
| Background-noise severity | 2 / 3 | 66.7% |
| Audio quality | 3 / 3 | 100% |
| Speaker overlap | 1 / 3 | 33.3% |
| Long silence | 3 / 3 | 100% |
| **All discrete fields** | **21 / 24** | **87.5%** |

Per-call scores were `call_001` 8/8, `call_002` 6/8, and `call_003` 7/8. The misses were `call_002` background-noise severity (`low` predicted, `medium` expected), `call_002` overlap (`false` predicted, `true` expected), and `call_003` overlap (`false` predicted, `true` expected). The dashboard rounds the overall percentage to 88%.

The emotional-tone confusion matrix is perfect only for the classes represented in this three-call set:

| Expected / Predicted | neutral | satisfied | upset |
| --- | ---: | ---: | ---: |
| neutral | 1 | 0 | 0 |
| satisfied | 0 | 1 | 0 |
| upset | 0 | 0 | 1 |

There are no frustrated or distressed examples, so no recall is measured for those classes. This is a supplied labeled development set with `n=3`, not statistically meaningful evidence of production-wide generalization.

The final predictions and observed request latency were:

| Call | Tone / intensity | Noise | Quality | Overlap | Long silence | Confidence | Latency |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| call_001 | upset / high | none | clear | false | false | 0.90 | 37.134 s |
| call_002 | neutral / medium | TV, low | clear | false | false | 0.90 | 35.128 s |
| call_003 | satisfied / medium | sharp static, medium | clear | false | false | 0.95 | 47.333 s |

Total observed request time was 119.595 seconds; mean per-file latency was 39.865 seconds. Total evaluation audio was approximately 237.8 seconds (3.963 minutes). The sanitized final result is stored in `docs/results/final_production_results.json`.

## Why a hybrid rather than transcription-only classifier

Several required outputs are acoustic, not textual: background noise, technical degradation, overlap, and dead air. A transcript-only pipeline necessarily discards evidence needed for those labels. The hybrid preserves acoustic evidence while using transcription for customer-role isolation and semantic context. Deterministic fusion makes threshold behavior testable and prevents one model's mistaken keyword interpretation from automatically controlling the final label.

## Batch and failure handling

The batch contract requires exactly one CSV manifest with `name` and `result_json` columns. The implementation:

- accepts a single ZIP or a browser folder/file selection;
- requires exactly one manifest;
- matches manifest rows to audio by exact basename;
- identifies missing, unsupported, oversized, duplicate, and unmatched inputs;
- treats each valid call as an independent inference unit;
- limits concurrent model requests (default 3);
- reports one call failure without failing the remainder of the batch;
- streams per-file completion to the dashboard.

Crucially, `result_json` is never sent to the model. When present, it is used only **after a prediction exists** to calculate development-set validation metrics.

An earlier production smoke run encountered transient Gemini 503 “high demand” errors on individual files. The application surfaced them as independent per-file failures without crashing the batch or corrupting successful rows. That smoke run is not the accuracy result above; the subsequent final run completed all 3/3 files successfully.

## Cost analysis

The assignment ceiling is **$0.003 per audio minute** for total paid production inference. The selected accuracy-first production pipeline **does not meet it**.

Google documents Gemini audio tokenization at **32 tokens/second = 1,920 audio tokens/minute**. Current Gemini 3.6 Flash Standard pricing is **$1.50 / 1M input tokens** and **$7.50 / 1M output tokens including thinking**, while Batch/Flex pricing is **$0.75 / 1M input** and **$3.75 / 1M output**.

Therefore raw-audio input alone is approximately:

- Standard: `1,920 × $1.50 / 1,000,000 = $0.00288/min` before prompt/output/thinking;
- Batch/Flex: `1,920 × $0.75 / 1,000,000 = $0.00144/min` before prompt/output/thinking.

ElevenLabs lists Scribe v2 at **$0.22/hour**, or approximately **$0.003667/audio-minute**. Scribe alone therefore exceeds the complete trial ceiling.

The final 3.963-minute run exposed semantic-call usage of 2,842 input tokens and 243 output tokens. At Gemini Standard rates that is `$0.004263 + $0.0018225 = $0.0060855`, or approximately **$0.001535/audio-minute**. Combining Scribe, raw-audio Gemini input only, and the measured semantic call gives a defensible known Standard-rate lower bound:

| Known component | Cost per audio minute |
| --- | ---: |
| ElevenLabs Scribe v2 | $0.003667 |
| Gemini raw-audio input only | $0.002880 |
| Gemini semantic call | $0.001535 |
| **Known lower bound** | **~$0.00808** |

This intentionally excludes raw-audio prompt text, raw-audio output/thinking tokens, and Modal emotion2vec compute, so actual production inference cost is somewhat higher. The raw-audio Gemini request's usage is not surfaced in the final hybrid response; an exact total would therefore be misleading. Even Batch/Flex Gemini pricing cannot make this architecture meet $0.003/min because Scribe alone is approximately $0.003667/min. The submitted architecture makes an explicit accuracy/cost trade-off.

The free-tier development invoice is **not** reported as production cost.

Official pricing and audio-tokenization references:

- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/docs/audio
- https://elevenlabs.io/pricing/api?price.section=speech_to_text
- https://modal.com/pricing

### Unshipped cost-optimization experiment

A separate prototype used one Gemini 3.5 Flash-Lite raw-audio structured call and the existing Modal emotion2vec service, removing Scribe and the second Gemini semantic call. It was not shipped or deployed.

The final cost-only `call_002` measurement used 1,387 Gemini input and 187 output tokens. Gemini cost was approximately $0.001516/audio-minute. Modal's full HTTP duration gave a conservative upper bound of approximately $0.001548/audio-minute, for **$0.003065/audio-minute combined**. The service itself reported 1.007 seconds of compute; using that measured compute gives approximately $0.001577/audio-minute combined at Modal's published CPU and memory rates.

The strict conservative gate was missed, compression changed acoustic predictions, and the resulting `call_002` prediction scored 4/8 fields. A full three-call labeled evaluation was intentionally not run, and the prototype was abandoned instead of tuning against labels. It is a plausible lower-cost direction requiring more validation, not evidence that the trial requirement was achieved.

## Latency analysis

The initial baseline calls measured:

- ~9.78 s inference for ~30.95 s of audio;
- ~25.61 s inference for ~171.93 s of audio.

That is faster than real time in both cases. In the final production-dashboard run, individual request times were 37.134, 35.128, and 47.333 seconds: 119.595 seconds total and 39.865 seconds mean per file. Bounded concurrency reduces batch wall-clock time while remaining capped to limit provider spikes.

## Privacy and data handling

The application does not persist uploaded audio or labels in an application database. Batch data is kept only for the request lifetime.

The current development classifier sends raw audio to Gemini. Google's current pricing documentation states that free-tier content may be used to improve products, while paid-tier content is not. Because the assignment identifies production-call audio as confidential, **free-tier inference is suitable only for an explicitly approved development trial**. A production deployment must use an AutoAce-approved provider/tier and document retention/training settings.

The repository never requires API keys in source control. Secrets and evaluator credentials are environment variables.

No supplied audio, `labels.csv`, or raw evaluation artifact is tracked. Uploaded data is not persisted by this application to a database. Provider-side handling remains subject to the configured provider account, tier, retention, and training settings.

## Deployment record

- Authenticated dashboard: https://autoace-sigma.vercel.app
- FastAPI backend: https://autoace-api.vercel.app

The backend is deployed through the Vercel CLI. Its Vercel project is not GitHub-connected, so repository pushes do not automatically deploy it.

## Production hardening / scale path

The assessment dashboard intentionally remains simple, but the inference service boundary supports a more production-native path:

1. call ends in AutoAce's telephony pipeline;
2. recording is stored in approved object storage;
3. internal job includes call ID + recording reference;
4. classifier runs asynchronously;
5. structured result is stored against the call;
6. QA/analytics systems consume the labels.

For much larger batches, direct object-storage uploads should replace multipart function uploads so request-body limits do not constrain the dashboard. At sustained high throughput, a queue/workers architecture would provide retries, backpressure, idempotency, and rate-limit control.

## Known limitations and likely failure modes

- Customer-role inference may be ambiguous in unusual multi-party calls.
- Emotion classes near decision boundaries (`frustrated` vs `upset`, `neutral` vs mildly `satisfied`) require more labeled examples for calibration.
- Open-ended background-noise type creates semantic-equivalence issues (`television` vs `TV`); evaluation should define whether exact string or semantic equivalence is expected.
- Very short cross-talk can be interpreted differently from operationally meaningful overlap.
- Dead-air threshold is semantic in the assignment rather than a fixed duration, which can create boundary disagreement.
- A model-generated confidence score is not inherently calibrated; calibration requires a larger held-out labeled set.
- The three supplied labels are insufficient to estimate robust generalization across all five emotional classes.

## If given a larger labeled production set

With hundreds or thousands of labeled calls, the next steps would be:

- stratified train/validation/test splits by dealership/call type;
- prompt/model selection against macro F1 and per-field metrics;
- confidence calibration and abstention/escalation thresholds;
- taxonomy normalization for open-ended noise labels;
- error slices by audio duration, dealership, phone quality, accent, call direction, and speaker count;
- comparison of single multimodal model vs specialist acoustic + semantic ensemble;
- cost/latency benchmarking at production concurrency;
- drift monitoring after deployment.
