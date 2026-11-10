# Frozen native SLURP implementation contract

2026-11-10. Complements the native academic lock; no completed inference claim.

Retain all official real recordings: TRAIN50628, devel8690, TEST13078. Exclude
synthetic only from this explicitly real-only condition; never merge its
results with real+synthetic trained academic models. Preserve utterance IDs,
recording filenames, speakers and reference token spans in scoring provenance.
No transcript-based removal or held-out-domain filtering in the native release.

Input is audio or actual ASR evidence plus a TRAIN-defined taxonomy. Reference
transcription is allowed only in a separately named oracle control. Prediction
must provide scenario, action and an entities array with every entity occurrence
represented as type/filler. Repeated types and repeated identical fillers retain
their multiplicity. Intent is the official scenario/action combination.
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
