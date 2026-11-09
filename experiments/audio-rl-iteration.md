# Persisted audio GRPO iteration protocol v1

The runnable chain is `sample_audio_policy.py` → `update_audio_actor.py` →
`export_audio_policy.py`. `run_audio_rl.py` schedules these as sequential
subprocesses. This is the local audio actor path using real verl DataProto,
GRPO advantages and PPO loss; it is not the distributed verl Ray worker stack.

## Identity, objective and state

Both sampler and actor use the same checkpoint identity function, covering weights,
tokenizer, processor and configuration files. The update requires successful
rollout terminal/result receipts, matching response/manifest/audio hashes,
complete per-prompt replicate groups, and TRAIN-only inputs. Reward text must
equal decoded response tokens. Nonfinite log-probs and nonfinite parity errors
fail before any optimizer step. Actor/rollout parity tolerance remains .02 for
the intended engine integration; the controlled HF check used 3e-4.

The original base stays frozen across iterations. LoRA parameters retain their
names and optimizer coordinates; save/reload AdamW moments and CPU RNG state.
Actor dropout is disabled. A merged exported model is used only by the next
vLLM subprocess. Parent adapter hashes, optimizer hash/configuration and export
lineage must agree before continuing. Reward mode and pair bonus cannot change
within a trajectory. Initial SFT adapter must match the sampling export receipt.
The current schedule uses a CPU actor; device RNG persistence for stochastic
NPU actor operations has not been implemented or verified.

Every iteration closes its sampler process before updating and exporting. No
in-process weight replacement is assumed. The next engine loads the new verified
export. Equal reward groups without KL produce no optimizer step; iteration count and actual
update count are reported separately. No-op exports may be weight-identical, so
iteration-specific export roots preserve lineage without overwriting receipts.

Successful stages are reused. Failed or incomplete stages stop for process/state
inspection; a missing heartbeat/terminal is not evidence a child is dead. Active
checkpoints and optimizer state remain outside Git and are not cleanup targets.

## Executable integration schedule

`configs/experiments/rl-schedule-smoke.json` selects one complete FSC TRAIN source
group, six views, two sampled responses per prompt and two iterations. It starts
from the unadapted base to test the chain, not to measure RL quality. Uniform
malformed responses may legitimately yield zero updates. Final matched SFT/GRPO/
pair-GRPO experiments must instead start from the same completed SFT policy and
freeze their larger sample/token/update budgets before inference.

```bash
python3 tools/jev/run_audio_rl.py \
  --config configs/experiments/rl-schedule-smoke.json \
  --output /mnt/cephfs/xingchao/experiments/multimodal-jev/rl-schedule-smoke-v1
```

Run in the pinned runtime with the current immutable source snapshot, shared
assets mounted, and explicitly allocated NPU. The scheduler does not allocate
devices, preempt other jobs, or use an HF fallback for rollout. This schedule has
not yet been run: current devices are occupied.

### SFT-initialized integration configuration

`configs/experiments/rl-schedule-sft-smoke.json` pins the completed 47-update SFT
export and its matching unmerged adapter. The actor retains the original Qwen
base; the merged checkpoint is used for sampling. It preserves the base smoke
protocol: two iterations, one TRAIN recording, six views and two responses per
prompt (24 scheduled responses). This is an integration check, not the final
three-seed algorithm comparison. The separate export root ends in
`rl-schedule-sft-smoke-v2`.

Run the following before allocating an engine (choose a new output filename):

```bash
PYTHONPATH=src .venv/bin/python tools/jev/preflight_audio_rl.py --config configs/experiments/rl-schedule-sft-smoke.json --output /path/to/new-preflight-receipt.json
```

[Actual asset preflight](../docs/engineering/rl-sft-preflight-v2-receipt.json)
passed for the real export, adapter, manifest and audio. The first attempt
failed on unreadable adapter weights; its failure receipt remains outside Git.
Permissions were repaired without changing contents. Preflight does not start
an engine or prove an optimizer update or improvement.

The schedule stores both base and initial sampling identities. Actor continuation
checks export file hashes, adapter identity and the exact parent training-run
path. Two-iteration unit tests cover base/SFT starts, a fresh exported checkpoint
for the second rollout, the fixed actor base, parent-state handoff, and separate
no-op/update counts. Test doubles verify scheduling only; real optimizer
continuation and engine execution require their own numerical checks.

## Verification and limits

[Real actor update check](../docs/engineering/rl-actor-update-smoke.json) passed
in the pinned image: two controlled responses to one real FSC TRAIN prompt,
exact pre-update HF log-prob equality, actual verl objective and LoRA optimizer
step, gradient norm 2.34265 and 112 changed parameter tensors. A deliberately
mismatched reward text/token record is rejected. Fixture responses and their
HF log-probs were constructed for verification, not sampled by vLLM; no vLLM
rollout receipt is emitted. Zero scalar PPO loss at ratio=1 reflects centered
advantages, not zero gradient. The standalone check uses AdamW's default weight
decay; the experiment runner explicitly uses zero weight decay.

Receipt/identity rejection and scheduler stage handling have unit tests.
The [optimizer continuation check](../docs/engineering/actor-resume-smoke.json)
now verifies actual Qwen3-ASR LoRA restoration: 112 trainable tensors after a
second teacher-forcing step exactly match the uninterrupted run, AdamW moments
match, and all optimizer step counters advance from 1 to 2. Both losses are
10.9858503342, maximum parameter error zero. It starts from the previous controlled
fixture actor, not a vLLM rollout. The shared `actor_state.py` persistence helper
is used by the experiment runner and validates names, shapes and parameter order.

The [export check](../docs/engineering/actor-export-smoke.json) also passed on
that controlled-fixture adapter: loaded-adapter versus merged and reloaded
weights both have maximum response log-prob error 1.24931e-4, within the frozen
atol=3e-4 / rtol=1e-5 gate. The artifact remains in the experiment directory,
explicitly labeled as a fixture, not the SFT initializer. The first export attempt
failed because adapter files were unreadable from the host; its failed container
is retained. Successful exports now set readable shared-directory permissions.

Full two-iteration engine execution and actual on-policy gains remain unverified.
Optional fixed-reference KL is now implemented; see the
[numerical protocol](reference-kl.md). Existing schedules omit it. A separate
frozen checkpoint, coefficient and estimator are recorded in actor identity and
must remain unchanged across continuation. Actor/reference tokenizer semantics
and processor tensors must agree. Controlled numerical validation is separate
from actual engine execution and does not establish an appropriate KL strength.
Distributed workers, multi-audio optimization, matched three-seed comparisons and heldout
task evaluation remain required research work.

## Actor microbatching

The actor accepts optional `micro_batch_size`; the smoke schedule retains the
default full batch until numerical validation passes. Rewards and
GRPO normalization are computed on the complete response group before splitting
model forwards. Each microbatch loss is weighted by its count of unmasked
response tokens divided by the complete batch count, preserving verl's token-mean
objective even with unequal response lengths or an incomplete final microbatch.
All microbatches pass old-policy log-probability checks before any backward or
optimizer step. Gradient clipping and AdamW step happen once per complete batch.

`tools/jev/actor_microbatch_smoke.py` compares a full three-response batch to a
2+1 split on actual Qwen3-ASR/verl, including gradients, AdamW first moments and
updated parameters. It also injects a log-probability error only in the final
microbatch and requires no parameter or optimizer state change. Responses are
controlled TRAIN fixtures with unequal lengths; this is numerical verification,
not on-policy vLLM inference. Run handles:
`npu:multimodal-jev-actor-microbatch-smoke-v1` and `-v2`, source/results under
`/mnt/cephfs/xingchao/experiments/multimodal-jev/actor-microbatch-smoke-v1` and
`-v2`.
Version 1 failed its parameter comparison gate: one reported element differed
by 2.43674e-6, exceeding atol=2e-6 / rtol=2e-4. A second diagnostic run
preserved the same update code and tolerances, recorded all
gradient/parameter/moment errors before asserting, and executed the late-parity
rejection check. Its terminal result is exit 1 because the exact parameter gate
failed; do not infer equivalence from unit tests or relax tolerances to hide this
failure.
The completed v2 evidence found gradient max error `4.65661e-8`, AdamW
first-moment max error `4.65661e-9`, and parameter max error `2.82067e-6`
(two isolated elements). The late parity failure left zero optimizer states and
unchanged parameters. This is a near-equivalence diagnostic, not a passed exact
equivalence gate; formal runs retain full-batch updates until a numerical policy
for these accumulation differences is approved and separately validated.

This reduces device activation memory. CPU-side preprocessing still prepares the
complete batch, and the base model remains resident. It does not implement
distributed workers, packed sequences or an unlimited-size dataset loader.

## First actual sampled update completed

The 128GiB full-projection v5 actor exited0 after consuming all12 responses from
the verified real vLLM feature rollout. Its probability error is5.66244e-5,
gradient norm3.05773 and optimizer update count1. Independent saved-artifact
inspection confirms all112 LoRA tensors changed and all112 AdamW states have
step1 with finite moments. Hashes match the successful result. See
[receipt](../docs/engineering/vllm-feature-actor-v5-result.json).
All pair bonuses on this batch are zero, so this validates the actual sampled
training path without distinguishing GRPO from pair-GRPO or showing efficacy.
This older immutable snapshot does not contain the later token-budget ledger.

The continuation supervisor stopped before export because safetensors created
root-owned mode0600 weights. A separate container verified hashes before and
after adding read permissions, then checked saved optimizer/adapter contents.
New runner code makes saved adapters readable. The failed chain is retained;
the [v3 supervisor](../docs/engineering/vllm-cpu-chain-v3-launch.json) reuses the
completed first actor, with the prior source snapshot plus the permission fix,
and requires export parity, fresh second vLLM sampling and optimizer restoration.
No first optimizer step is repeated. The complete two-iteration chain remains
pending until all downstream artifacts pass.

The updated-policy export container exited0. Merge and reload both have maximum
response log-prob error4.4465065002441406e-5, within the unchanged3e-4/1e-5 gate.
Adapter/base/export hashes and lineage independently revalidate. The policy ID
changed from e28da9d0... to119cf54b...; see
[export receipt](../docs/engineering/vllm-updated-policy-export-v1-result.json).
The supervisor launched a fresh local second vLLM container
b3f78cfa53b12febc85fee228ce517fc4da2bb2a17305dff26fdbd33f87a596a
using that checkpoint and sampling seed18. Sampling and the restored-optimizer
second actor remain pending; successful export alone does not complete the chain.

The fresh second engine exited0 with6 prompts,12 responses and82 response tokens
from the verified updated policy, seed18. Full rollout/manifest/audio identities
pass again. The [second sampling receipt](../docs/engineering/vllm-second-rollout-v1-result.json)
records the launched second actor and its explicit original parent optimizer
path. Numerical parity and the second actual optimizer step remain pending.
