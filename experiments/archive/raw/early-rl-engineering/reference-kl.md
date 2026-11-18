# Fixed reference KL implementation check v1

Registered 2026-11-10 before execution. Add optional actor-loss regularization
using the pinned verl k2 or k3 sampled-token estimator and token-mean aggregation.
The intended research reference is the fixed merged SFT initializer, separately
loaded and frozen. It never advances with the actor. Checkpoint content hashes,
coefficient and estimator must remain identical throughout a trajectory; verify
actor and reference processor tensors on every selected TRAIN recording.
Existing schedules omit reference regularization and retain their objective.

This check uses CPU FP32, four threads, seed 17, one FSC TRAIN Noul prompt and
two identical controlled correct responses. Initialize a q/v LoRA actor (rank 4,
alpha 8), perturb its B matrices with Normal(0, 0.01), and keep the separate base
reference unchanged. Equal task rewards produce zero GRPO advantage, but a
nonzero reference penalty must still produce an update. AdamW lr=1e-5, zero
weight decay, KL coefficient 0.1, k3; these are numerical test constants, not
research hyperparameter choices. No inference on validation/test labels.

Required gates: actual verl k3 loss gradient matches the analytic expression
(atol=1e-7, rtol=1e-6), ignored nonfinite padding has zero gradient, unequal
response lengths preserve token weighting, trainable reference tensors fail,
Qwen pre-update actor/controlled-response error <=3e-4, policy loss <1e-7,
positive KL, changed LoRA tensors, no reference gradients or parameter-version
changes, and exactly identical reference outputs before/after the update.
Real response fixtures are not vLLM trajectories and cannot establish RL gains.

For k3 the pinned verl implementation clamps log ratios to [-20,20] and penalty
values to [-10,10]; this is its sampled, clipped estimator, not full vocabulary
KL. k2 is also supported explicitly, without silently changing the estimator.
Reference inference uses no_grad; only response-token log probabilities are
stored. Equal rewards skip updates only without reference regularization; with
regularization a zero total gradient still skips AdamW and its moment changes.

Entry point: tools/jev/reference_kl_smoke.py. Full-batch updates remain the
default; this check does not resolve the existing microbatch equivalence gate.

## Completed numerical check

Container multimodal-jev-reference-kl-smoke-v1 exited 0. Actual verl k3 analytic
gradient maximum error was 2.98023e-8. The real Qwen update had equal rewards
[1,1], policy loss 0, reference penalty 0.42178139, weighted loss 0.04217814,
gradient norm 4.453949 and 112 changed LoRA tensors. Reference parameter versions
and outputs stayed identical, with no reference gradients. See the
[receipt](../../../../docs/engineering/reference-kl-smoke-v1-receipt.json).
This directly validates the k3 loss/update path; the persisted runner's reference
loading/continuation and k2 numerical behavior still require runtime checks.

To enable a future separately registered schedule, add an actor.reference
object with checkpoint (the fixed merged SFT directory), coefficient (a positive
finite number), and estimator (k2 or k3). Omit the entire object to retain the
original objective. The current two-iteration smoke schedule remains unchanged.
