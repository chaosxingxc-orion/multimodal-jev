# Effective-token budget control, revision 1

This protocol is a separate comparison from the full six-epoch opportunity
schedule. It does not alter completed or running snapshots. Its estimand is
quality under the same upper bound on effective training input tokens, with
generation reported separately. It does not establish equal FLOPs, equal wall
time, equal audio encoder work or equal total inference-plus-training compute.

Use the frozen 31-recording TRAIN pool, shared SFT initializer, original base,
six-view batches and seeds 17/29/43 from algorithm-comparison-v1. Keep the same
fixed six-epoch ordered schedule as the maximum horizon. No skipped groups,
partial groups, length-based packing or resampling after a budget rejection.
Compare SFT, GRPO standard, pair-GRPO standard, GRPO center-only and pair-GRPO
center-only. Retain raw-policy n4 decoding, max_tokens32, pair weight .2,
rank4/alpha8, float32 full projection, no reference KL and gradient clip1.

Before training, fix the training cap at **765,594 effective input tokens**
per arm/seed, the measured full six-epoch SFT total. Count each unpadded
prompt-plus-response token once per gradient forward and once per parity or
reference forward. This accounting assigns equal units to these different
passes; it is deliberately not a FLOPs conversion. Fix the separate generated
token cap at **142,848**, the n4 six-epoch upper bound. SFT generates zero tokens;
report that difference, rather than calling total costs matched. Generation
prefill, engine load, export, padded work and failed-stage costs remain separate.

Admission uses the measured processor lengths in a hash-pinned TRAIN profile
plus the configured response cap. Reserve one gradient pass and all configured
no-grad passes for a complete batch. Stop before that batch if either cap
would be exceeded; do not skip it to search for a cheaper one. After success,
charge actual tokens and release unused reservation. An RL zero-advantage batch
charges parity and generation, but no gradient pass or optimizer update.
Reloading a trajectory reconstructs the ledger from completed receipts. Missing
accounting, changed profiles, changed audio or processor contracts are errors.
Actual processor lengths are checked before sampling/model forwards, and
observed generation and actor budgets must agree. Failed stages are retained
and are not automatically retried. A successful budget stop is not a full
schedule completion and can contain zero updates; report the stop reason.

Report used cap, remaining cap, next rejected reservation, scheduled/completed
groups, actual optimizer updates, gold/sampled response tokens and generated
tokens for every trajectory. The maximum single-batch reservation here is
34,840 training token units; conservative admission can leave material slack.
If the maximum horizon is reached first, report schedule exhaustion, not budget
exhaustion. Do not extrapolate quality to unspent tokens or substitute a
performance-selected checkpoint for the final admitted export.

Keep LR1e-5 as the primary registered setting. Cross all five arms with
LR3e-6 and LR3e-5 as sensitivity controls, retaining all three seeds at each
setting. Do not derive an LR multiplier from input-token ratios or choose a
normalization from the existing small reward diagnosis. Current SFT uses
example-mean loss and RL response-token-mean loss; report this objective
difference. A loss-reduction ablation would require a separate registration.

Evaluation uses the frozen common FSC/SLURP strict development protocols,
speaker-cluster bootstrap, paired both-correct, non-null correctness, missing
evidence/null handling and constant-output controls. Report per seed, arm, LR
and task without a single blended success score. No further prompt tuning on
these development examples. Final independent task and test evaluation still
require frozen choices; this protocol alone does not prove cross-task gains.

The optional runtime config block is:

```json
{
  "budget_control": {
    "profile_path": "/absolute/shared/path/to/profile/result.json",
    "profile_sha256": "actual_sha256_from_successful_profile_run",
    "limits": {"training_tokens": 765594, "generated_tokens": 142848}
  }
}
```

Both SFT and container RL runners accept this block. The profile must include
processor-contract hashes from the current profiler; the original v1 profile
remains immutable and is not silently upgraded. Unit tests exercise caps,
no-ops, replay and whole-group stopping. Real budgeted SFT and multi-step RL
execution must additionally pass before launching this matrix. Resource
allocation and scale sensitivity execution remain pending; this document is a
registration, not a launch receipt or an efficacy result.
