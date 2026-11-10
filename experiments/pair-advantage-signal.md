# Pair reward advantage signal diagnosis

Actual n2 mixed-audio rollouts produced three nonzero pair bonuses (0.1333,
0.2,0.2), but pinned verl group-standard-deviation normalization yielded maximum
advantage difference 1.1920928955078125e-7 and no sign changes versus GRPO.
The [receipt](../docs/engineering/pair-advantage-v1-result.json) preserves all
per-response advantages. This batch does not meaningfully distinguish algorithms.

For two unequal rewards with difference d, sample-standard-deviation normalized
advantages are approximately +/-1/sqrt(2), independent of magnitude. A bonus
that preserves ranking mainly changes the epsilon correction. This diagnosis
does not say all pair objectives or larger groups are equivalent.

Register n4 sampling before execution: use exactly the same two TRAIN recordings,
12 views, SFT checkpoint, feature processing, seed17 and other decoding options;
change only n from2 to4, as already proposed for formal comparisons. Persist all
48 real responses. Compare GRPO and pair-GRPO advantages on these same outputs
using the pinned verl function, recording nonzero bonus counts, differences and
direction changes. No selection by generated correctness and no fabricated
responses. A weak or zero signal is retained as a negative diagnosis.

This is a reward/normalization diagnostic, not a training comparison. Four samples
alone do not guarantee a different signal: repeated two-level reward patterns
can still normalize alike. If needed, separately preregister alternatives such
as centering without per-group scaling or separately centered pair advantages,
with scale controls and matched ablations. Existing actor objectives stay fixed.

## Center-only control implementation

The actor now exposes norm_adv_by_std (default true), forwarded unchanged to
the pinned verl advantage function. Explicit false centers rewards per prompt
without dividing by their sample standard deviation. Reject non-boolean values
and any change within a persisted actor trajectory. Existing source snapshots
and configs keep standard normalization.

A new read-only n2 audit measures the center-only control on the same saved
responses; the n4 audit also reports both choices. This is an objective/scale
diagnosis, not a selected winning algorithm. Any subsequent training study must
cross reward composition (base/pair) with normalization (standardized/center-only),
hold initializer/data/schedule fixed, report gradient norms and effective token
budgets, and separately assess learning-rate/scale sensitivity. Neither changing
normalization nor increasing n is itself evidence of semantic improvement.

The queued n4 audit waits for exact successful rollout container
0cae1e866ea83c5c5aa3cbfc57ee3b03d12d7a3fa2db1cecba1779c29fb16633,
validates all 48 responses, and launches independently in the pinned verl image.
See [queue receipt](../docs/engineering/pair-advantage-n4-audit-v1-queued.json).

## Completed n2 center-only diagnosis

The [v2 receipt](../docs/engineering/pair-advantage-v2-result.json) records a
successful pinned-image audit of the identical 24 responses. All standardized
results match v1 exactly. Centering without standard deviation gives a maximum
base-versus-pair advantage difference of 0.10000002384185791, compared with
1.1920928955078125e-7 under standard normalization. The three nonzero bonuses
therefore survive as magnitude differences under centering. This establishes
an objective-scale effect on this batch; no actor was updated by this audit and
no semantic improvement or preferred normalization follows from it. The n4
audit and matched training comparisons remain separate experiments.

## Completed n4 diagnosis

The [n4 receipt](../docs/engineering/pair-advantage-n4-v1-result.json) verifies
successful sampler and audit container exits, all 48 real responses, and their
content hash. Three responses receive nonzero pair bonuses. Standardized
advantages still differ by at most 3.5762786865234375e-7 with zero sign changes;
the center-only maximum difference is 0.1499999761581421. Increasing n alone did
not provide a material standardized algorithm contrast on these recordings.
Retain this negative finding. It does not establish equivalence across other
recordings or policies, and the center-only scale difference does not establish
learning or cross-task transfer. No training arm is selected from this audit.

## Full initializer pool diagnosis

Register one broader TRAIN-only diagnosis before its sampling: all31 recordings
and186 views already selected by the SFT initializer, sorted by example_id,
unchanged SFT exported policy, seed17, n4, max_tokens32, raw-policy decoding and
actor_features. Keep the registered max_model_len2048; if a prompt exceeds it,
retain the failure and explicitly revise before any retry. No selection by
duration, sampled correctness, development scores or reward magnitude.

Generate all744 responses in one fixed-policy sampling job. It may queue behind
the exact live SFC CPU engine check to avoid competing for the same cores.
After successful exact Docker/application terminal checks, run the existing
pinned-verl advantage audit across the complete outputs. Preserve all responses,
pair bonuses, standardized advantages and center-only controls. Report per-view
and per-recording nonzero bonus/advantage coverage and maximum differences, with
1e-5 as a descriptive material-difference threshold, not an efficacy criterion.
This expands the scope of the two-recording mechanism diagnosis; it neither
updates weights nor selects a winning normalization. No post-divergence training
trajectory may reuse these initializer rollouts as on-policy data.

## Coverage reporting implementation

`tools/jev/summarize_advantage_signal.py` now checks successful application
terminals, rollout/manifest hashes, policy identity and exact response order
before summarizing per-view and per-recording coverage. It rejects duplicate
replicates, incomplete arrays, nonfinite advantages and non-TRAIN examples.
Material differences use strictly greater than the registered1e-5 threshold.
Sign changes are response counts, explicitly distinct from the older audit's
token-level sign-change count.

The [existing n4 coverage check](../docs/engineering/pair-advantage-n4-coverage-v1.json)
reproduces the earlier two-recording diagnosis:3 nonzero bonuses among48
responses, no material standardized differences, and12 center-only response
differences across3 prompts in1 recording. Centering spreads a bonus's change
over the prompt's four responses, so this count is not the bonus count. This
validates reporting on completed evidence; it is not the full31-record result
and does not select a preferred normalization.
