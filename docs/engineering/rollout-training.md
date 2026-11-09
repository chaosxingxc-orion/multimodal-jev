# vLLM rollout and verl audio update v1

`rollout.VllmAudioRollout` calls actual `vllm.LLM.generate` with raw prompt plus
waveform, requests chosen-token log-probs and keeps every sampled token including
EOS. It has no alternative backend fallback. Replicate IDs, termination reason,
checkpoint identity and log-probs are persisted. Invalid or incomplete groups fail.

The explicit actor_features mode preserves HF actor log-mel inputs through
vLLM's feature interface. It uses a narrowly scoped local processor workaround,
with real registry parity verified on two audio lengths. Default waveform mode
is unchanged; each rollout records its mode in config. Full processor inspection
found hop-boundary padding absent from the earlier direct-processor check; see
the CPU integration protocol for both failures and the successful feature check.

`tools/jev/sample_audio_policy.py` is the runnable sampling entrypoint. Example
configuration: `configs/experiments/vllm-audio-rollout-smoke.json` (TRAIN only).
Run only on allocated NPU resources using the pinned runtime plus this source
snapshot. Existing full NPU workloads have not been stopped or modified.
The checkpoint identity hashes weights plus tokenizer/processor/config files.
Recreate the engine after actor export; this first implementation does not claim
in-process weight synchronization or distributed verl worker support.

`rl_update.update_actor` consumes these records into real verl DataProto and
uses verl GRPO advantage and PPO loss. Before a weight update it checks policy
identity, prompt/replicate completeness, and raw actor-versus-rollout response
log-probs (default max absolute error .02). Any mismatch rejects the update.
Equal-reward groups have zero advantage and are reported as no-op. Dropout is
disabled during probability computation and differentiation. No imagined reward
variance or fabricated successful updates.

Version 1 requires temperature 1, top_p 1, top_k -1 and no probability-changing
penalties. Other sampling distributions require the same transformations on
actor logits and are rejected until implemented. Response padding has no loss.
The helper supports ordinary GRPO and `reward_mode='pair-grpo'`. Pair bonuses
are composed per rollout replicate across complete six-view source groups,
then GRPO centers by example ID (same schema), never by recording ID.
Optional fixed-reference KL has a passed controlled Qwen/verl numerical check;
see [its protocol](../../experiments/reference-kl.md). Existing schedules omit it.
The new persisted iteration entrypoints and
sequential schedule are described in [the iteration protocol](../../experiments/audio-rl-iteration.md);
the full real-vLLM schedule is not yet verified.

## Verification state

Real HF batched actor, audio collation and verl loss components passed earlier
CPU checks. New rollout parsing and stale-policy handling have CPU unit tests.
Actual ARM CPU `LLM.generate` now completed six TRAIN prompts / 12 responses;
the actor probability gate and update are running separately. Ascend devices
remain occupied; no NPU execution or complete two-iteration result is claimed.

An isolated official vLLM 0.30 ARM CPU image is now verified and a real engine
sampling completed under the [CPU integration protocol](../../experiments/vllm-cpu-integration.md).
The [sampling receipt](vllm-cpu-rollout-v1-result.json) records complete artifacts. Its backend
dependency differences are recorded and retain the same numerical parity gate.

Full vLLM ModelConfig/registry inspection in the NPU-built image tries to query
Triton Ascend device architecture even for CPU-only preprocessing. This fails
without an assigned NPU (run `vllm-processor-parity-v1`, terminal exit 1).
Direct bundled processor inspection is narrower and is tracked separately.

Direct processor check completed ([receipt](vllm-tokenizer-parity.json)):
HF actor and the vLLM-bundled Qwen ASR processor produce exactly the same token
IDs and valid log-mel frames for two audio lengths. HF pads frames to 200/100;
vLLM processor returns 185/92 frames. Both valid frame counts agree (185/92),
and maximum difference on valid features is zero. v1's whole-padded-mask equality
assertion was invalid and failed; v2 explicitly compares the effective frames.
The full vLLM registry/NPU engine failure remains unresolved by this narrower check.

## Pretrained-policy diagnosis and initialization

The TRAIN-only [HF sampling probe](pretrained-policy-probe.json) returned ASR
transcriptions in all four samples (0 JSON objects, reward -1 throughout).
This is a real sampled negative result, not an optimized policy. SFT initialization
has been launched according to [its protocol](../../experiments/fsc-audio-sft-init.md).
The full FSC closed-label control is also running under
[the full-split protocol](../../experiments/fsc-frozen-audio-full.md).

The reproducible ASR-only development baselines are defined by
`tools/jev/transcribe_manifest.py` with configs
`asr-transcript-fsc-dev.json` and `asr-transcript-slurp-dev.json`. They save raw
transcriptions, token IDs, per-record hashes and corpus WER against the frozen
reference transcript metadata. These runs measure ASR error propagation only;
they do not feed references into decisions or claim an ASR-to-text Jev score.
Completed receipts and measurements are in
[asr-baseline-v1-receipt.json](asr-baseline-v1-receipt.json): FSC WER 14.56%
and SLURP WER 17.90% on the fixed development manifests.

Long-run handles:

- `npu` container `multimodal-jev-fsc-frozen-audio-full-v1`, output
  `/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-frozen-audio-full-v1/run`.
- `npu` container `multimodal-jev-fsc-audio-sft-init-v1`, output
  `/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-audio-sft-init-v1/run`.

Check Docker live/terminal state and per-run progress before resuming; do not
restart merely because a heartbeat file is old. Source snapshots are copied into
each run. They do not update the previously built image; model/data mounts are
read-only, outputs separate. Checkpoint/optimizer state is retained for active runs.

## SFT export and fixed development evaluation

`export_audio_policy.py` gates on completed SFT receipts, verifies adapter versus
merged/reloaded response log-probs on two TRAIN examples, and publishes only a
successful content-addressed checkpoint under shared `models/multimodal-jev`.
`after_sft.py` observes the actual dependency container and launches export then
development evaluation. It does not restart the dependency or reset existing jobs.
The live `post-sft-v1` supervisor uses an earlier copied snapshot; later lock and
failed-terminal improvements in the repository do not modify that running process.

`policy_evaluation.py` preserves raw generation and per-example resumable output.
The fixed [development protocol](../../experiments/fsc-schema-evaluation.md) has
31 official valid source recordings, six views each, no SFT training overlap.
Malformed output is never null. Reward and evaluation use a strict shared parser.
Scorer controls and numerical pair-reward checks are tracked separately from model
quality; neither is evidence of RL improvement.

SFT initialization and export have now completed; see
[fsc-sft-init-v1-receipt.json](fsc-sft-init-v1-receipt.json). FSC/SLURP inference
has completed on SLURP (format learning, no correct non-null targets) and remains
running on FSC. `finalize_policy_comparisons.py` observes exact existing
container IDs over SSH and applies the common scorer and preregistered paired
comparison after exit-zero and complete prediction/hash checks. Original
inference metrics are retained. Observation failures are retried; they never
cause restarts or imply completion.

Detached finalizer root:
`/mnt/cephfs/xingchao/experiments/multimodal-jev/sft-development-comparison-v1`.
Its `launch.json` records the local PID and source/config hashes. Check that
process and the actual evaluation containers before any recovery.
