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
