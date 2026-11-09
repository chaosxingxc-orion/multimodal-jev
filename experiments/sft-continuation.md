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
