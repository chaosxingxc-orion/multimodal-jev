# vLLM rollout and verl audio update v1

`rollout.VllmAudioRollout` calls actual `vllm.LLM.generate` with raw prompt plus
waveform, requests chosen-token log-probs and keeps every sampled token including
EOS. It has no alternative backend fallback. Replicate IDs, termination reason,
checkpoint identity and log-probs are persisted. Invalid or incomplete groups fail.

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
The helper currently supports ordinary GRPO single-decision rewards; paired
reward composition, reference KL and multi-iteration experiment scheduling remain.

## Verification state

Real HF batched actor, audio collation and verl loss components passed earlier
CPU checks. New rollout parsing and stale-policy handling have CPU unit tests.
No claim of actual `LLM.generate` execution or `update_actor` end-to-end success
is made yet: the available Ascend devices are occupied.

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

Long-run handles:

- `npu` container `multimodal-jev-fsc-frozen-audio-full-v1`, output
  `/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-frozen-audio-full-v1/run`.
- `npu` container `multimodal-jev-fsc-audio-sft-init-v1`, output
  `/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-audio-sft-init-v1/run`.

Check Docker live/terminal state and per-run progress before resuming; do not
restart merely because a heartbeat file is old. Source snapshots are copied into
each run. They do not update the previously built image; model/data mounts are
read-only, outputs separate. Checkpoint/optimizer state is retained for active runs.
