# Qwen3-ASR chronological SFC adapter v1

This is an explicit adaptation of the ASR backbone to the official SFC inputs,
not reproduction of the official Qwen2.5-Omni-7B inference backend. Preserve the
official data release and supplied assistant history from the external evaluation
protocol. Do not tune on Test-ID/OOD.

## Prompt and waveform contract

The stock HF Qwen3-ASR chat template gathers every audio into one user turn and
then emits assistant messages, dropping user text. It therefore does not preserve
the chronology of audio 1 → assistant clarification → audio 2.
`sfc_audio.render_prompt` writes explicit Qwen role delimiters in that order,
moves the first schema/environment text to system (as the official SFC inference
route does), removes its literal `<audio>` marker, and retains subsequent text.
The project adds an English language hint. Record this prompt adaptation in all
results; it has not been shown to improve instruction following.

The model sees only the `input` mapping; passing a whole supervision record fails.
The original expected function calls never enter the request. Two ordered audio
recordings remain two audio regions, not a concatenated synthetic utterance.
Audio is converted explicitly to mono by channel mean and resampled to 16 kHz
with `scipy.signal.resample_poly`; original rate/channels/sample count and converted
sample count are recorded in each inference output. Original files are unchanged.

HF's processor requires one text per audio. Process placeholder-only text/audio
pairs together, then insert their expanded audio token runs into the chronological
text. Retain the jointly extracted features in the same order. vLLM receives the
unexpanded prompt and a list of `(waveform, 16000)` items; its engine owns expansion.
Its configured limit is two audio items per prompt.

## Integration check

`sfc_audio_smoke.py` selects the lexicographically first two-audio TRAIN record,
checks source audio hashes, verifies feature order under waveform reversal,
compares HF token IDs and valid mel frames with the direct vLLM-bundled processor,
and runs the actual HF model forward. No generated output or task accuracy is
claimed. Numerical thresholds are atol=1e-5, rtol=1e-5 for features; token IDs and
valid frame counts must match exactly.

Retained diagnostic runs:

- `sfc-audio-smoke-v1`: failed because the source audio was not mono 16 kHz;
  fixed by explicit conversion rather than changing the stored dataset.
- `v2` and diagnostic `v3`: token IDs matched, but separate HF extraction gave
  the shorter clip 139 valid frames versus 140 with joint vLLM extraction.
  Joint waveform padding changes the last STFT frame. Separate audio processing
  is not an equivalent actor contract, even if token counts match.
- `v4`: joint processing reached the waveform-reversal check; exact floating-point
  equality failed on 8 of 435,200 values, max error 1.192e-7. Reversal uses the
  same declared feature tolerance (1e-5) in v5; masks remain exactly equal.
- `v5`: passed, exit 0. Two original 22,050 Hz recordings explicitly resampled
  to 16 kHz; prompt 1,046 tokens; valid mel frame counts 1,606 and 140.
  Direct HF/vLLM processor token IDs are identical and both valid-feature max
  errors are zero. Actual HF model forward is finite. Waveform-reversal feature
  error is 1.192e-7. See [receipt](../docs/engineering/sfc-audio-smoke.json).

All 8,895 release records also passed prompt chronology/placeholder checks.
This is metadata/input validation only; no test output was generated. vLLM model
source advertises `audio: None` (no hard item count limit), but registry and full
engine behavior still require device-backed verification.

## Real engine entrypoint

On an allocated NPU, using the pinned image plus current source snapshot:

```bash
python3 tools/jev/infer_sfc_vllm.py \
  --config configs/experiments/sfc-vllm-smoke.json \
  --output /path/to/new-run
```

The checked-in configuration selects the same TRAIN-only two-turn case, greedy
256-token decoding, FP32 and a single engine sequence. This is an engine check,
not the final paper replication precision/budget. The runner hashes model,
processor, manifest and source, records raw text/token IDs, supports per-example
resume, and writes terminal status. It has no HF generation fallback.
Full external evaluation must use all official test rows with the frozen policy;
the separate scorer retains missing predictions as failures.

Remaining gates: real vLLM registry/engine execution on NPU; actor/rollout log-prob
parity for these multi-audio prompts; multi-audio verl collation and optimization;
generation quality and complete ID/OOD evaluation. Existing single-audio GRPO
collation does not yet support this multi-audio contract.

## CPU engine validation

An independent CPU vLLM run is now launched using the unchanged registered
TRAIN-only two-audio config, base Qwen3-ASR-0.6B-hf, chronological prompt,
greedy max_tokens256 and max_model_len8192. It uses the pinned official ARM CPU
vLLM image fdd4a28..., four cores,16GiB and1GiB KV cache, with no NPU allocation.
See [launch](../docs/engineering/sfc-vllm-cpu-engine-v1-launch.json).
This can validate actual engine execution independently of the occupied NPUs;
it does not satisfy the separate NPU gate. Preserve any engine failure without
silently switching to HF, concatenating audio or reducing the prompt/history.
No test-ID/OOD rows are selected and no external transfer claim follows from
this TRAIN integration check. The official Omni reproduction is also separate.

CPU v1 exited1 during engine initialization: the8192 context requires1.75GiB
KV cache, exceeding the configured1GiB (0.97GiB usable). No generation occurred.
Retain this failure. CPU v2 changes only engine.kv_cache_memory_bytes to3GiB;
source, inputs, context8192, container16GiB limit and decoding remain unchanged.
See [failure and new launch](../docs/engineering/sfc-vllm-cpu-v1-failure-v2-launch.json).
The full-pool reward diagnosis v1 queue stopped on this dependency failure;
a [new v2 queue](../docs/engineering/pair-signal-full-pool-v2-queued.json) observes
the exact v2 SFC container and retains identical diagnosis configuration.

CPU v2 also exited1 before generation: actual allocation remained1GiB despite
the engine parameter. This CPU build's VLLM_CPU_KVCACHE_SPACE environment setting
overrode it. The executor now permits an explicit rollout.kv_cache_gib (positive
integer below container memory), persists it in launch identity, and rejects
changes on an existing stage. CPU v3 sets both environment and engine to3GiB;
all prompt/audio/context/decoding inputs remain unchanged. See
[new launch](../docs/engineering/sfc-vllm-cpu-v3-launch.json).
The old full-pool queue stopped, and the observation-only two-step launcher was
stopped before any schedule submission. New separately named
[dependency queues](../docs/engineering/sfc-v3-dependent-queues.json) observe v3;
no training container was stopped or automatically resubmitted.

CPU v3 completed: exact container8187173d... exited0, application receipts and
one prediction hash verified. The engine logged3GiB KV allocation and13,952
cache tokens. Both source audio hashes and their ordered conversion records
were verified. The base ASR generated69 tokens, stopped normally and took39.62s
for generation; its output was transcription, failing strict tool-call format.
See [result](../docs/engineering/sfc-vllm-cpu-engine-v3-result.json).
This proves execution of this registered two-audio input through the CPU engine,
not that both clips causally affected the answer, nor actor/rollout probability
parity, tool-call quality, full SFC evaluation or interactive task success.
The full-pool TRAIN reward diagnosis subsequently began its own fresh engine.
