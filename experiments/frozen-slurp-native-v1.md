# Frozen native SLURP implementation contract

**User route correction:** the ASR→text probe registered below is cancelled.
Retain its immutable schedule as history; do not resume it. Current execution
must feed audio plus taxonomy directly into one speech multimodal actor and
emit the full native composite result. Dataset export and official scoring
remain applicable. A new direct-audio protocol/config must precede dispatch.

2026-11-10. Complements the native academic lock; no completed inference claim.

Retain all official real recordings: TRAIN50628, devel8690, TEST13078. Exclude
synthetic only from this explicitly real-only condition; never merge its
results with real+synthetic trained academic models. Preserve utterance IDs,
recording filenames, speakers and reference token spans in scoring provenance.

2026-11-10 duplicate audit correction: TRAIN has 50628 recording entries but
50627 unique filenames. One audio appears under two utterance IDs with conflicting
person fillers. Match the pinned official loader's last-occurrence-per-filename
semantics, preserving both annotations in a conflict audit. Merge both utterance
components (and all their recordings) into one source group before learning or
selection allocation. No duplicate audio is counted as independent evidence.
Devel8690 and TEST13078 are already unique. Old release v1 failed closed at the
duplicate; keep its terminal and partial stage. Publish corrected release v2
under a new content hash, never overwrite the original metadata.
No transcript-based removal or held-out-domain filtering in the native release.

Input is audio or actual ASR evidence plus a TRAIN-defined taxonomy. Reference
transcription is allowed only in a separately named oracle control. Prediction
must provide scenario, action and an entities array with every entity occurrence
represented as type/filler. Repeated types and repeated identical fillers retain
their multiplicity. Joint intent scoring uses scenario/action together; retain
the original intent field separately. The source has 162/30/37 utterances in
train/devel/test whose raw intent string is not scenario + "_" + action.
Do not interpret those raw aliases as unseen semantic classes or silently
rewrite the original annotations. Verify the official scorer's contract.
This composite native output is separate from the current single-value Jev
consumer contract; do not encode all native entities as a first-span query.

No gold, span indexes, sample IDs or reference transcript in audio-route model
prompts. Exact original token-span surface words populate offline gold fillers.
The metadata audit validates spans and counts repeated entity types; it does not
prove audio integrity, model inference or scorer parity.

Before running full inference, pin the published SLURP scorer at
8eb16545762be97ace75334109d73824217311f1 and verify fixtures for repeated
entities, wrong type/right filler, right type/wrong filler, empty entities and
malformed outputs. Enforce all recording IDs exactly once before invoking the
official scorer: its own missing/duplicate handling is insufficient.
Report scenario/action/joint intent, exact entity F1, Word-F1, Char-F1 and
official SLU-F1 separately. Do not average Word/Char F1 to invent SLU-F1.

Learning and selection populations must be grouped by original slurp_id, keeping
all recordings of one annotated utterance in one role. Freeze group IDs before
any teacher feedback. The previous domain-held-out scenario/first-slot release
remains a separate Jev transfer task, not this native benchmark.

Outstanding gates: label-free composite release, full-entity output parser,
official scorer fixtures, pinned no-training model/decoding config and TRAIN
engineering probe. No teacher call or new weight training is scheduled here.

Before any native inference, lock malformed-output policy: retain each recording
in the denominator with invalid scenario/action sentinels and zero predicted
entities. Report malformed count separately. This produces official false
negatives for its gold entities; empty-empty contributes no entity counts.
Well-formed wrong or out-of-taxonomy predictions retain their content and accrue
official errors. Never repair JSON, normalize predicted fillers, drop malformed
records, or invent a false-positive entity. Missing/duplicate prediction IDs
invalidate the run instead of silently dropping samples.

## Registered first native TRAIN probe

Use the first16 TRAIN slurp_id groups in ascending SHA256("slurp:"+slurp_id)
order, with all real recordings of each group. Record the group list before
dispatch; do not choose by entity content, model correctness or audio length.
The first64 groups and following256 groups are reserved as disjoint future
learning/selection populations, but this probe launches no teacher request.

Qwen3-ASR-0.6B-hf revision7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c and
Qwen3-0.6B revisionc1899de289a04d12100db370d81485cdf75e47ca are frozen.
HF eager BF16, seed17, greedy n1, context4096, ASR max128, semantic output
max256 (complete entity lists can exceed FSC's max128); no truncation/repair.
Use configs/tasks/frozen-slurp-native-prompt-v1.txt unchanged. One hour per
probe, fixed existing NPU image, physical175 card4 after the exact FSC-Jev
baseline container exits. Pin all model assets by hash before/after inference.

Wait for the complete native data release and its successful terminal receipt;
never consume a partial .building directory. Record request/taxonomy/group
hashes and container identity, then score using the separate official scorer
environment. This proves only a TRAIN engineering path. Full native TEST is
not dispatched by this probe, and no test results influence it.
