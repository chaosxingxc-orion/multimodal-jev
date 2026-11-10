# Frozen FSC-Jev baseline and native-prompt transfer

2026-11-10. Freeze before observing six-view inference.

Use release e046b9e58357e26343b628adce1bcbe178472fa2de0290b6480d489c52fc527c,
with labels/provenance separate from model requests. First engineering probe:
the same 64 TRAIN learning source groups previously locked for native FSC,
all six views per recording (384 requests). This is a probe, not held-out
transfer evidence: the native teacher already saw feedback from these groups.
No new teacher dispatch is authorized by this protocol.

Compare original instruction with native-selected p1 unchanged. Both use the
same ASR evidence, Qwen3-ASR-0.6B-hf and Qwen3-0.6B, HF eager BF16, seed17,
greedy max128, context4096, frozen weights. Physical175 cards4–7 only.
Each probe has a one-hour application budget. Preserve all outputs and failures.

First run original384, export pinned ASR evidence, then run p1 with that exact
evidence. IDs (including truth-bearing view suffixes), labels and provenance
must never enter model prompts. Do not repair output or weaken schema scoring.

Report each of native, permutation, rebind, missing_candidate, noul.True and
noul.False separately. Pair native with permutation/rebind/missing_candidate,
and positive with negative Noul: both-correct is the principal paired measure.
Also report all-six-correct by source recording; no sixfold independent sample
claim. Invalid JSON including malformed null is incorrect. Incomplete or
duplicate views/predictions invalidate the result. Constant ID, constant true,
constant null and gold-output fixtures must distinguish failure from success.

Next: same unchanged prompts on all official TEST recordings (22758 derived
requests/3793 recordings), separate from native tests. This measures transfer
between task definitions within FSC, not cross-dataset generalization.
An actual six-view teacher-learning loop must have its own predeclared budget,
TRAIN-only feedback, unchanged selection groups, and test isolation.
SLURP and other dataset families are required for cross-dataset conclusions.
