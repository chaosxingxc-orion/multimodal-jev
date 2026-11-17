# Frozen direct audio: full FSC-Jev held-out evaluation v1

Registered before reading v3 candidate selection results or any new TEST
prediction. Selection follows the already locked 64-learning/256-selection
TRAIN procedure. Freeze its winner; no more teacher calls in this protocol.
Evaluate the parent and selected instruction on all 3793 official FSC TEST
recordings, six correlated views each (22758 requests per distinct prompt).
If parent wins, run it once and record that no distinct optimized prompt exists.
Do not evaluate losing candidates on TEST or reselect from TEST outcomes.

Same Qwen3-ASR-0.6B checkpoint and pinned NPU image, BF16/eager, seed17,
actual audio, fixed `language English<asr_text>` assistant header and unchanged
finite-label next-token readout. No generated/reference transcription, no
online second model, no weight updates. New runner adds only full TEST
population/prompt/model locking; it does not change forward or label decoding.
Recompute both TEST conditions with the same new source snapshot. Historical
JSON generation results are a separate interface, never the paired control.

Physical cards 5 and 6 are authorized and will be checked before launch. Each
prompt has a 10800s application / 11100s outer cap, detached container, immutable
source and durable terminal. No automatic restart or retries. A failed run is
not a partial accuracy result. All requested IDs must complete, audio/request/
prompt/model hashes must match, weights must remain unchanged, and container
and application exits must both be zero before scoring. Actor never reads gold.

Report native31-way intent accuracy and all six per-view scores, four paired
both-correct scores, all-six-correct, paired fixes/regressions and descriptive
McNemar statistics. TEST is final measurement, never optimization feedback.
Do not pool six views as independent recordings; all-six is the primary derived
metric. Native FSC and derived Jev scores remain separate. Format validity is
forced by label readout and is not evidence of task understanding. No monetary
cost is invented from token counts; prior unknown teacher usage stays unknown.
