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
