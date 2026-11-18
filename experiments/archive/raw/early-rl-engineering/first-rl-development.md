# First sampled update development diagnostic

Register before evaluating this actor. Use the first successfully completed
actual-vLLM actor v5, without selecting among checkpoints by evaluation outcome.
It used one FSC TRAIN recording, six views,12 responses, one full-batch update
at LR1e-5, seed17, with original SFT initializer. All pair bonuses were zero.

Evaluate its saved unmerged adapter on the unchanged FSC186-view development
manifest and SLURP168-view transfer manifest, using the same original base,
greedy64-token decoding, CPU float32 and seed17 as the completed SFT evaluations.
Reuse those immutable SFT predictions after artifact verification. Compare
updated actor minus SFT, with strict common scoring and2,000 speaker bootstrap
draws, seed17. Report all variants, format/schema validity, false refusals,
non-null correctness and paired both-correct rates; retain negative deltas.

This fixed one-update diagnostic checks whether the executable update changes
heldout decisions. It is not a matched-budget SFT/GRPO/pair-GRPO comparison,
does not isolate pair reward, and cannot establish multi-seed learning or final
generalization. No test inference, checkpoint selection, early stopping or
training-parameter adjustment is driven by these results. The separately
registered continuation chains and mixed-audio integration runs remain fixed.

Use immutable evaluation snapshots and read-only adapter/model/data mounts,
four CPU cores and16GiB per evaluation. Verify the completed actor terminal,
Docker exit and saved adapter hashes before launching. The two existing SFT
comparators must have successful exact container IDs and complete matching
predictions before finalization. Do not alter the historical SFT outputs.

## Completed SLURP diagnostic

The168-view SLURP candidate and unchanged SFT comparator both have verified
exact Docker exit0 and complete predictions. Recomputing the comparison with
the current common scorer reproduces the saved2,000-draw speaker bootstrap.
See [result](../../../../docs/engineering/first-rl-slurp-development-v1-result.json).

| Measure | SFT initializer | One FSC RL update |
| --- | ---: | ---: |
| Strict format |168/168|168/168|
| Schema valid |159/168|158/168|
| Correct non-null targets |0/81|0/81|
| Correct null targets |71/87|59/87|
| False refusals on non-null targets |72/81|69/81|
| Native intent |0/24|0/24|
| Permutation/rebind both-correct |0/24 each|0/24 each|

Date/time/person/place slot correctness changes14/18/20/19 to11/13/18/17
(each out of24), entirely from absent-slot null predictions. Answers increase
from25 to40 without a correct non-null answer. The all-null control obtains87
correct rows, exceeding both policies' mixed-view totals; those totals must not
be presented as semantic transfer. Most nonzero per-variant bootstrap intervals
include zero; the valid-partition time-slot interval is negative, but these are
descriptive multiple development comparisons, not corrected confirmatory tests.

This fixed one-update policy shows no non-null transfer gain and loses correct
abstentions. It is neither a matched-budget algorithm comparison nor evidence
that all RL methods fail. The initializer, continuation checks and full-pool
TRAIN signal diagnosis stay fixed; no checkpoint/learning-rate/prompt selection
is made from this diagnostic. FSC completion is reported below.

## Completed FSC diagnostic

Both exact FSC containers exited0 and contain all186 predictions. The current
common scorer reproduces the saved comparison exactly after excluding its
Docker observation metadata; all2,000-draw speaker bootstrap results agree.
See [receipt](../../../../docs/engineering/first-rl-fsc-development-v1-result.json).

| Measure | SFT initializer | One FSC RL update |
| --- | ---: | ---: |
| Strict format |186/186|186/186|
| Correct non-null targets |50/155|68/155|
| False refusals on non-null targets |57/155|34/155|
| Native intent |5/31|9/31|
| Permutation / rebind correct |4/31,7/31|9/31,15/31|
| Correct missing-candidate null |30/31|27/31|
| Permutation / rebind both-correct |2/31,4/31|6/31,9/31|
| Noul True / False correct |8/31,26/31|9/31,26/31|

Noul balanced accuracy changes54.84% to56.45%; paired both-correct changes3/31
to4/31. Answering more improves some FSC decisions while losing some correct
abstentions. Native and paired bootstrap intervals include zero at their lower
boundary; the dataset has only8 speaker clusters. All statistics remain
descriptive development findings without multiple-comparison correction.

This one-recording, one-update, one-seed result has zero pair bonuses during
training. It cannot isolate pair reward or replace the matched budget controls.
The SLURP non-null result remains0/81 for both policies, so the FSC changes do
not demonstrate cross-task transfer. Preserve both positive and negative
findings and the originally frozen training/evaluation configurations.
