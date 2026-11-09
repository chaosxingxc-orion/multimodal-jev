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

The HF audit has now exited 0. All 24 responses pass the unchanged 0.02 gate;
per-response errors, artifact hashes and the observed Docker terminal are in
[audit receipt](../docs/engineering/vllm-mixed-probability-v1-result.json).
This confirms single-response probability agreement for both selected speakers
and audio lengths; the combined full-batch backward path is not yet verified.

A queued full-batch mixed actor now waits for the exact single-recording v5
container to exit; it does not preempt it. It rechecks successful mixed audit and
rollout identities, then uses the original SFT adapter, pair-GRPO, .02 gate,
four CPU threads and 256GiB memory. Response-only projection remains disabled
for this independently registered test. See
[queue receipt](../docs/engineering/vllm-mixed-actor-v1-queued.json).

## Response-only mixed actor

Register an independent response-only full-batch actor on the same 24 saved
responses, original base and SFT adapter, seed17, pair reward, LR1e-5 and .02
probability gate. It waits for exact successful exit of the single-recording
response-only actor 5b5c40e5f2b0f5583f944ed4fd1142e3ffbded3b762ea52e0f7cd42bb5064ee2
and verifies its real update and saved optimizer/adapter hashes before launch.
Revalidate the mixed rollout and independent HF audit. The only actor-config
change relative to the registered mixed actor is response_logits_only=true;
use four CPU cores and a 64GiB container limit. Full projection remains an
independent queued experiment. No microbatching, normalization change or
additional rollout selection is introduced.

Record the mixed full-batch probability gate, actual update count, finite
gradient norm, response-token ledger and saved optimizer/adapter integrity.
This extends executable training coverage to different recordings and lengths,
but remains an integration experiment; learning and generalization require the
separately registered multi-seed comparisons.

The response-only supervisor is registered in the
[queue receipt](../docs/engineering/vllm-mixed-response-only-v1-queued.json).

The v1 response-only queue stopped before launching its actor because its exact
single-recording dependency was OOMKilled at32GiB. Register a new v2 queue against
the 128GiB single-recording retry, retaining the same success/artifact gates and
byte-identical mixed actor source/config. Raise the planned mixed container
limit to256GiB given the observed activation-memory failure. This is a resource
revision before launch, not evidence of memory savings. See
[v2 queue receipt](../docs/engineering/vllm-mixed-response-only-v2-queued.json).
