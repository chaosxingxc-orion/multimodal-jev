# Native update_policy real-model gate

Register before execution. Wait for the exact successful Docker and application
terminal of `native-verl-actor-update-v1`; preserve any failed dependency without
launching this gate. That run saves both reference and native-microbatch tensors.
Verify its comparison-file hash and numerical parity before consuming it.

Use the same twelve actual saved initializer responses, same original Qwen3-ASR
0.6B model and exact initial LoRA adapter, rank4/alpha8, seed17, CPU float32,
LR1e-5, AdamW weight_decay0, foreach/fused disabled, gradient clip1. The response
and adapter identities must match the dependency. These are matching-policy
numerical replays, not newly sampled on-policy continuation trajectories.

Invoke the pinned `DataParallelPPOActor.compute_log_prob` and `update_policy`
methods directly, with all twelve responses in one minibatch and one microbatch,
one PPO epoch, vanilla token-mean loss, no entropy or reference KL, no packing,
no fused kernels, no sequence parallelism and `use_rollout_log_probs=True`.
Set temperature1 explicitly. Full-batch execution deliberately isolates native
loop semantics before adding the separately required variable-length microbatch
weighting. Do not claim it validates multi-microbatch or distributed execution.

Two explicit scoped CPU adapters are enabled: log_softmax/gather dispatch for
CPU logits, and `dp_actor.get_device_id` returning a CPU device after checking
the actor's parameters/buffers. Restore both imports on exit; installed sources
and images are unchanged. No accelerator devices are mounted. This is an
adapted CPU native-method run, not an unmodified worker or NPU success.

Require zero dropout probabilities in both model modules and config, since
native update_policy changes the model to train mode while the reference actor
uses eval. Check finite rollout log probabilities at the established .02 gate
before optimization. Capture autograd gradients before native clipping; compare
every trainable parameter, preclip gradient and AdamW moment against the saved
reference using atol2e-6/rtol2e-4. Require exactly one optimizer step per parameter
and at least one changed parameter. Save tensors and all comparison failures
before raising. Never change tolerances after observing results.

One Gloo rank supports the native actor constructor only. The gate does not
cover zero-advantage/momentum no-op behavior, variable microbatches, Ray/FSDP,
two-rank model execution, asynchronous vLLM audio integration or learning efficacy.
It does not export a policy for development evaluation. Those remain subsequent
requirements before switching the registered comparison executor.

The [queue receipt](../../../../docs/engineering/native-verl-update-policy-v1-queued.json)
pins immutable source4e40cd7 and the exact dependency container. The detached
supervisor observes that handle and launches only on successful Docker and
application terminals with matching reference hashes. It does not restart
dependencies or infer failure from old heartbeats. Scoped CPU-device adapter
tests verify restoration after exceptions and rejection of non-CPU tensors;
all five CPU dispatcher/device tests pass in the system torch environment.
