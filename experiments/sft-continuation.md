# Shared-initializer SFT continuation

The SFT comparison must begin from the exact initializer used by GRPO and
pair-GRPO. The initialization run alone is not the budget-matched SFT arm.
The continuation entrypoint now requires both initial_adapter and
initial_checkpoint, verifies original base/adapter/export hashes, and loads the
existing adapter as trainable instead of creating another randomly initialized
LoRA. Rank and alpha must match. Each arm starts a fresh AdamW optimizer, as the
first RL actor does; same-run SFT checkpoint resume still restores its optimizer.

Before a comparison launch, validate this path with one fixed TRAIN recording:
the shortest recording in the previously frozen 31-record SFT pool, native view,
as recorded in configs/experiments/sft-continuation-smoke.json. Compare the loaded
adapter's teacher-forced probabilities against its immutable merged export
(atol 3e-4, rtol 1e-5), then execute one real cross-entropy step at learning rate
1e-5. Require nonzero finite gradients, changed LoRA tensors, fresh optimizer
state before the step and unchanged source checkpoint/adapter hashes afterward.
No development/test predictions affect this probe.

The probe exercises the shared loader and supervised objective. It does not
validate the full SFT training loop, establish matched training budgets, or
demonstrate improvement. Full comparison still needs frozen per-arm schedules,
seeds 17/29/43, explicit sample/token/update accounting and independent evaluation.
Variable generated response lengths make exact token matching a separate design
constraint; matching update counts alone is insufficient.

The CPU probe has been launched on npu as container
8b18b501a153faaf7e368e70c1759798f28f9f05d382e396f2cb10b013ae7ada,
with immutable source, original assets mounted read-only, four CPU cores and
16 GiB memory. Evidence root: sft-continuation-smoke-v1 under the shared
multimodal-jev experiments directory. See
[launch receipt](../docs/engineering/sft-continuation-v1-launch.json).
The static identity check matches the RL initializer policy
e28da9d0b70dc639fec442395eec4133154904726564b48dcd19001348f6541c;
actual forward/backward results are pending.

The probe has now exited 0. Initial merged-versus-adapter maximum log-probability
error is 0.00008928775787353516, within the registered gate. A real supervised
step changed LoRA tensors with finite nonzero gradients; the original initializer
hashes remained unchanged. See [result receipt](../docs/engineering/sft-continuation-v1-result.json).
This establishes loading and one-step execution, not a complete continuation
training run or SFT-versus-RL efficacy comparison.

## Explicit schedule and full-runner probe

SFT now accepts explicit batches with the same example-ID/update-boundary format
as the RL scheduler. It retains the supplied order, permits repeated exposures
across updates, rejects duplicates within a batch and admits only TRAIN rows.
Do not simultaneously set legacy accumulate/groups_per_class options.
Legacy initialization keeps its original shuffled class-balanced selection.

Register a two-update full-runner probe before execution: the same shortest
TRAIN recording's native view, then its noul.False view, one example per update,
seed17, unchanged initializer and LR 1e-5. The configuration is
configs/experiments/sft-continuation-runner-smoke.json. Require successful terminal,
two history entries, two optimizer steps, effective-token ledger persisted both
in result and resume checkpoint, saved optimizer state, and changed adapter hash.
This is full-runner/persistence validation, not a matched SFT/RL experiment or an
interrupted-process resume test. Existing live jobs keep their original snapshots.
