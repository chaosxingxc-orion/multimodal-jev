# Mixed-length audio integration v1

Register before execution. Select the shortest recording from the fixed 31-record
SFT training pool, and the longest recording from a different speaker. Break ties
lexically by source_group. Selection uses only TRAIN metadata/audio duration,
never predictions or development/test performance. Both selected recordings have
all six predefined views. See the selection receipt and
configs/experiments/vllm-mixed-length-smoke.json.

The selected waveforms have 20,805 and 57,344 samples at 16 kHz. This adds distinct
audio lengths and speakers to the one-recording engine probe; it is not a model
evaluation or a representative benchmark sample. Do not replace the separate
two-iteration single-recording chain with this experiment.

Use the original merged SFT initializer, official ARM CPU vLLM 0.30 image,
actor_features (including valid feature masks), seed17, n2, temperature1,
top_p1, top_k-1 and max_tokens32. Keep four CPU cores, 16 GiB memory and one
scheduled sequence. Persist all 24 responses and their sampled log probabilities.

After successful real sampling, diagnose merged HF single-response probabilities
and then validate a full-batch actor update with the original base/SFT adapter,
pair-GRPO, learning rate 1e-5 and fixed 0.02 parity gate. The full-batch actor
introduces padding across both audio lengths; single-response parity is not a
substitute for that check. Use a distinct actor output and optimizer state; never
continue the one-recording actor trajectory with this diagnostic batch implicitly.
Launch the larger actor only after the first one-recording actor finishes, with
an explicit memory budget; do not preempt a running job. No microbatch fallback
is authorized by this protocol while its parameter-equivalence gate is unresolved.

Retain failures without relaxing tolerances. Zero reward variance or zero update
is an observed result. Successful mixed-length execution still does not prove
distributed verl workers, NPU execution, learning benefit or generalization.

Launched local rollout container
63ea8751bd44db77139933cf6066f5e60140ad7f8e3f61b65b39dbdfa796eac0
under vllm-cpu-mixed-rollout-v1. A detached supervisor under
vllm-mixed-probability-audit-v1 validates that exact upstream identity and all
24 outputs before launching the read-only HF audit on npu CPU. The full-batch
mixed actor is not launched yet. See
[launch receipt](../docs/engineering/vllm-mixed-length-v1-launch.json).

The mixed-length engine exited 0 with all 12 prompts, 24 responses and 153
response tokens. Model/manifest/audio/response identities and complete groups
passed validation. See [sampling receipt](../docs/engineering/vllm-mixed-rollout-v1-result.json).
The read-only HF audit has independently launched on npu CPU as container
f681f64839b333fcd82d1840c552509b6b6ef18f9f1f8b3dfb120a4bd2b5e31b.
No mixed-length actor update has run yet, and sampling completion does not
establish probability parity across these recordings.
