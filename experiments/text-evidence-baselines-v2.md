# Transcript controls with the TRAIN-validated v2 prompt

Conditional registration before v2 development inference: launch only after both
TRAIN prompt-probe containers exit0, complete13 records, and independent strict
scoring verifies v2 format validity13/13 and semantic correct count at least v1.
Keep the original v1 development controls and their raw malformed outputs.

Use prompt_version=v2, as implemented and frozen for the TRAIN probe. No further
prompt changes or answer repair. The model, official revision, evidence sources,
greedy non-thinking CPU float32 decoding, seed17 and64-token limit remain exactly
as in text-evidence-baselines.md. Both ASR-text and reference-text conditions
use v2. Reuse existing ASR caches after hash/manifest validation. Run all186 FSC
and168 SLURP decisions per condition, with new immutable output roots.

Report reference minus ASR within every registered variant/presence stratum,
strict format/schema validity and semantic correctness, refusals, paired
both-correct rates and2,000 speaker-bootstrap draws. Model/source/prompt-version
equality is required. These runs follow a development-driven format diagnosis
and a small TRAIN gate, so remain exploratory development controls, not final
test evidence or an unbiased claim of prompt superiority. No acoustic or RL
improvement follows from text formatting success. Do not choose between v1/v2
by development accuracy; report the two protocols separately.

Four CPU containers, four cores and12GiB each, readonly shared assets. Finalizer
requires exact successful Docker IDs and complete matching prediction artifacts.
Do not stop or overwrite the still-running original FSC v1 controls.

The TRAIN gate passed and all four exact v2 containers are running; see
[launch receipt](../docs/engineering/text-development-v2-launch.json).
The final SLURP reference launch returned an SSH observation timeout; the same
named container was subsequently inspected running with the expected image and
job mount, and was not relaunched. The detached finalizer observes the recorded
IDs and performs no inference or restarts.

## Completed SLURP v2 controls

Both exact SLURP containers exited0 with168 complete predictions. The current
common scorer reproduces the saved2,000-draw speaker-bootstrap comparison and
checks identical model/source/prompt version and decoding. See
[receipt](../docs/engineering/slurp-text-development-v2-result.json).

| Measure | ASR transcript | Reference transcript |
| --- | ---: | ---: |
| Strict format valid |168/168|168/168|
| Schema valid |116/168|123/168|
| Native intent correct |4/24|4/24|
| Permutation correct |3/24|5/24|
| Rebind correct |4/24|5/24|
| Present-slot correct |0/9|1/9|
| Correct absent-slot null |0/87|0/87|

Every absent-slot output is a string rather than JSON null; all remain scored
as generated, including the literal string `NULL`. Answers contain extracted
words, copied text and occasional instruction text rather than reliable slot
decisions. JSON formatting success is therefore insufficient. Reference text
does not solve schema adherence or abstention, and is not an acoustic upper
bound. Preserve these results without post-hoc answer repair or another prompt
selection on the same development set. The original fenced-output v1 protocol
remains separately reported. FSC v1/v2 controls continue independently.
