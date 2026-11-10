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
