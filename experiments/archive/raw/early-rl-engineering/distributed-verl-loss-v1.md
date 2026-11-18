# Two-rank verl loss normalization probe

Use all48 saved real responses from `vllm-pair-signal-n4-v1/run`, validate their
policy/manifest/audio hashes, and compute pair rewards plus standardized GRPO
advantages on complete prompt groups before splitting. This reuses completed
rollouts solely for a numerical check, never as fresh on-policy training data.

Run two actual CPU Gloo ranks with a three-parameter deterministic log-prob
perturbation fixture. Split the first third of responses to rank0 and the
remaining responses to rank1. Compare DDP-averaged gradients of the pinned
verl PPO loss to a single-process full-batch token mean. Pass global token
count and dp_size2 to verl; do not additionally divide by microbatch count.
Also accumulate chunks of at most3 responses per rank under no_sync and
explicitly sum/average the two resulting gradients. Unequal chunk counts
must not trigger mismatched DDP collective calls.

Fix atol2e-6/rtol2e-5 for both gradient comparisons. A deliberately wrong
rank-local mean followed by rank averaging must differ by more than1e-5,
otherwise the negative control is nondiscriminating and the probe fails.
Pin actual core_algos.py SHA256. Retain every rank's gradient, token counts,
failure receipts, exact Docker ID and application terminal.

Allocate four CPU cores and8GiB in the existing pinned NGA image, no NPU
devices. No ASR model is loaded or trained: this verifies a collective-loss
prerequisite, not distributed audio workers, optimizer equivalence, model
quality or the full desired multi-rank training system. Existing snapshots
and formal algorithm comparison configurations are unchanged.

## Completed numerical result

The [exact-container receipt](../../../../docs/engineering/distributed-verl-loss-v1-result.json)
records exit0 and both rank artifacts. Rank0 processed16 responses/103 tokens;
rank1 processed32 responses/196 tokens. The complete batch contains299 tokens.
Global-token-normalized DDP gradient differs from the single-process reference
by at most5.58794e-9; accumulating6 versus11 microbatches gives7.45058e-9.
The deliberately incorrect local-mean control differs by0.0260895, above the
registered1e-5 discrimination threshold. Host-side checks independently
recomputed saved gradient errors and the299-token count from saved rollouts.

This establishes the loss-scaling prerequisite using real collectives and a
controlled three-parameter perturbation fixture. No ASR model, optimizer,
native actor update_policy, Ray/FSDP worker or accelerator was exercised.
