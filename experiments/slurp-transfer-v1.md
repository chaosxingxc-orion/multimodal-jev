# SLURP Jev transfer protocol v1

Registered before building/inference. No model score was used to choose domains.
Hold out calendar, email, and transport from all training utterances; these cover
scheduling, communication and travel. Retain official train/dev/test membership
and every real recording of admitted utterances. Derived views never cross splits.

Exclude train utterances whose NFKC/casefold/whitespace-normalized transcript is
present in official dev/test or any heldout domain. This uses text only to remove
contamination, never to construct candidate values or tune model parameters.
Speaker identity comes from the native metadata; report overlaps rather than
claiming speaker-disjoint generalization. Source IDs/paths/audio hashes are checked.

Produce five partitions: train, valid, test, heldout-valid, heldout-test.
heldout-valid can support development of the generalization procedure, so only
heldout-test is final domain-transfer evidence. For a strictly FSC-only transfer
experiment, no SLURP partition may be used in fitting or checkpoint selection.

Each recording: scenario Choice, candidate permutation, definition/id rebind,
and four open-slot questions (date, time, person, place_name). Query types chosen
before inspecting scores and available in original train. Candidate taxonomy
comes exclusively from original train. Taxonomy names are weak descriptions,
not a claim of independently authored business rules. Heldout candidate definitions
remain visible at inference: this is zero-shot domain definition access, not a
hidden-label recognition task.

Slots use the first annotated occurrence in native token-ID order; retain token
span, original entity list, native intent/action, recording hash and speaker.
Null means absent annotation; this is not an acoustic uncertainty target. The
native labels absent from train are retained and reported, never silently changed.

Report per-domain scenario accuracy, paired both-correct rates, open-slot exact
match and null/present support separately. Reconstruct full native outputs and
use the original SLURP scorer for a full SLU claim; this projection alone cannot
produce official joint SLU-F1. No Score task without native ordinal supervision.

## Construction result

Release `ba31cd72120ab28309fe40089274b6b70e9a9605c2fcd56e5de4a0323805f3f8`.
59,014 real recordings, 413,098 derived decisions:

| Partition | Recordings | Decisions |
| --- | ---: | ---: |
| train | 37,246 | 260,722 |
| valid | 6,516 | 45,612 |
| test | 9,953 | 69,671 |
| heldout-valid | 2,174 | 15,218 |
| heldout-test | 3,125 | 21,875 |

Excluded 13,306 recordings from heldout-domain train utterances, plus 76 train
recordings whose transcript duplicates evaluation. No admitted train/evaluation
normalized transcript overlap or cross-official-split audio-byte overlap.

The native intent names `likeness` (valid) and `locations` (test) are absent from
train; they remain in metadata rather than being relabeled as an existing intent.
Scenario prediction is a coarser task and is not native intent reproduction.

Official speaker overlap is substantial (e.g. 135 speakers between admitted train
and seen-domain valid). This release is not speaker-disjoint. Open slots are
imbalanced; all-null performance and present-only exact match are required controls.
For example train person has only 1,049 present vs 36,197 missing recordings.
No model has been evaluated on these partitions in this release.
