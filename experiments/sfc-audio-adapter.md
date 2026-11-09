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
