# Matched transcript-evidence decision baselines v1

Registered on 2026-11-10 before text-decision inference. This fills the ASR-to-
text and reference-text controls in the three-stage research protocol. It does
not reproduce an external paper score or replace the AudioJev reproduction.

Use the existing shared official Qwen/Qwen3-0.6B checkpoint, source revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Eight local files, including weights,
tokenizers and license, match official source hashes; see
[model receipt](../docs/engineering/text-baseline-model-receipt.json). This
small text model is a fixed control, not a claimed strongest text baseline.
No text-model fitting, prompt search or selection on test labels is performed.

Both conditions use identical model weights, task schema, JSON-only instruction,
greedy decoding, non-thinking chat template, seed 17, CPU float32/eager, four
threads and 64 new tokens. The sole evidence change is the transcript string:

- ASR-text: verified saved Qwen3-ASR predictions, matched by example and audio
  hash. Only the transcription field enters the prompt; the reference field in
  ASR records is ignored. Empty ASR text stays empty.
- Reference-text: original transcript_reference metadata, explicitly labeled
  as a reference-text diagnostic. Missing references cause failure. This is
  neither an audio model result nor a guaranteed performance upper bound.

Use the unchanged frozen FSC development selection (31 recordings, six views,
186 decisions) and SLURP transfer selection (24 recordings, seven views, 168
decisions). Test remains unused. All derived views of a recording receive the
same transcript. No candidate remapping, answer repair, constrained decoding,
truncation or forced null fallback is allowed. Save raw outputs, token IDs,
prompt/transcript hashes, generation time and length-limit flags.

Primary comparisons are reference-text minus ASR-text within each
partition/variant/target-presence stratum, with the common strict scorer and
paired speaker bootstrap (2,000 draws, seed 17). Report format/schema validity,
coverage/risk and paired both-correct alongside semantic accuracy. Reuse the
existing null/first-candidate controls. Do not treat correlated views as
independent trials or average unlike tasks into a headline result.

Text-stage costs and upstream ASR costs are separate; four dataset/condition
runs do not represent four independent ASR passes. The upstream ASR cache
contains 31 FSC and 24 SLURP recordings. Report combined pipeline costs only
when both stages have measured durations, without counting repeated view
records as repeated ASR inference. A difference between the conditions is a
development diagnostic of evidence sensitivity, not an RL gain or causal
proof about individual acoustic errors.

Entry point: `tools/jev/evaluate_text_policy.py`. Configurations are
`configs/experiments/text-{fsc,slurp}-{asr-text,reference-text}-dev.json`.
Runs use the immutable NGA-aligned image and a copied source snapshot, no NPU
devices, read-only assets and separate experiment outputs. Check the real
Docker state and terminal receipts before resuming; an old heartbeat is not
proof of termination.
