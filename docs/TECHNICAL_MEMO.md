# AutoAce Voice Tone & Background Noise Trial — Technical Memo

## Executive summary

This submission treats the problem as **offline production-call classification**, not a live telephony problem. The hosted dashboard accepts the required ZIP/folder batch and CSV manifest, validates the batch, analyzes valid recordings independently with bounded concurrency, streams progress, and exports the required `name,result_json` results as CSV or JSON.

The inference layer is intentionally isolated from transport/UI concerns so the same classifier can later be called from an internal AutoAce workflow using a call ID or recording URL rather than manual upload.

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

### B. Taxonomy-constrained raw-audio classifier

The implemented candidate uses the same raw-audio modality but grounds every field in the assignment's operational definitions. The model is instructed to infer the **customer** role, distinguish normal turn-taking from meaningful overlap, distinguish phone artifacts from meaningful non-speech noise, treat audio quality separately from noise, and reserve `long_silence_present` for operationally unusual dead air.

The model and thinking level are environment-configurable so the same harness can compare Gemini 3.6 Flash with lower-cost audio-capable models without changing application code.

Final labeled-set metrics should be generated reproducibly with:

```bash
uv run python scripts/evaluate_batch.py /path/to/evaluation_batch --output-dir artifacts/evaluation
```

The generated `run_details.json` contains exact field accuracy, per-field accuracy, emotional-tone confusion matrix, per-class precision/recall/F1, and macro F1. The three supplied calls are too small to support statistically strong claims; in particular, not every emotional-tone class has development-set support. Hidden-set performance should therefore be treated as the primary evaluation.

## Why raw audio rather than transcription-only classification

Several required outputs are acoustic, not textual: background noise, technical degradation, overlap, and dead air. A transcript-only pipeline necessarily discards evidence needed for those labels. Direct audio understanding keeps semantic speech content and acoustic context in one inference request.

A future ensemble could combine a specialized acoustic/event model with a separate transcript/semantic classifier if a larger labeled corpus shows a measurable improvement that justifies the extra cost and operational complexity.

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

## Cost analysis

The assignment ceiling is **$0.003 per audio minute** for total paid production inference.

Google documents Gemini audio tokenization at **32 tokens/second = 1,920 audio tokens/minute**. Current Gemini 3.6 Flash Standard pricing is **$1.50 / 1M input tokens** and **$7.50 / 1M output tokens including thinking**, while Batch/Flex pricing is **$0.75 / 1M input** and **$3.75 / 1M output**.

Therefore audio input alone is approximately:

- Standard: `1,920 × $1.50 / 1,000,000 = $0.00288/min` before prompt/output/thinking;
- Batch/Flex: `1,920 × $0.75 / 1,000,000 = $0.00144/min` before prompt/output/thinking.

This means Standard Gemini 3.6 Flash is too close to the ceiling to claim compliance once prompt and output/thinking tokens are included. If 3.6 Flash is selected for accuracy, the production design must use an appropriate lower-cost batch/flex consumption mode and measured token counts must confirm the full cost. Alternatively, Gemini 3.5 Flash-Lite can be evaluated: its current Standard input rate is substantially lower, but it should only replace 3.6 Flash if validation accuracy remains acceptable.

The free-tier development invoice is **not** reported as production cost.

Official pricing and audio-tokenization references:

- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/docs/audio

## Latency analysis

The initial baseline calls measured:

- ~9.78 s inference for ~30.95 s of audio;
- ~25.61 s inference for ~171.93 s of audio.

That is faster than real time in both cases. Batch latency is further reduced by bounded concurrency, while concurrency remains capped to avoid rate-limit spikes. The evaluation harness records `request_seconds` per file so final wall-clock and per-audio-minute latency can be calculated from the complete labeled run.

## Privacy and data handling

The application does not persist uploaded audio or labels in an application database. Batch data is kept only for the request lifetime.

The current development classifier sends raw audio to Gemini. Google's current pricing documentation states that free-tier content may be used to improve products, while paid-tier content is not. Because the assignment identifies production-call audio as confidential, **free-tier inference is suitable only for an explicitly approved development trial**. A production deployment must use an AutoAce-approved provider/tier and document retention/training settings.

The repository never requires API keys in source control. Secrets and evaluator credentials are environment variables.

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
