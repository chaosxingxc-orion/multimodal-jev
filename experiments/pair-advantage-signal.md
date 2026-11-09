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
