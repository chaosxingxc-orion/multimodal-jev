# FSC-Jev and full native SLURP progress

2026-11-10. No model weight updates. This extends the frozen-baseline stage;
the full academic reproduction, cross-dataset transfer and RL objective remain
incomplete.

## FSC-Jev

Original frozen ASR→text prompt completed 64 TRAIN recording groups × six views
(384 predictions), exit0. These are the native teacher learning groups, so this
probe is not held-out generalization evidence.

| View | Correct / 64 | Paired both-correct with native / 64 |
|---|---:|---:|
| native | 20 | — |
| permutation | 13 | 9 |
| rebind | 22 | 15 |
| missing_candidate | 0 | 0 |
| noul.True | 0 | Noul pair: 0 |
| noul.False | 0 | Noul pair: 0 |

All-six-correct: 0/64. In Noul views the parsed outputs are strings naming the
schema or decision type, not booleans. Missing-candidate outputs choose IDs or
fail JSON instead of returning null. Exact counts are preserved in
frozen-fsc-jev-train64x6-v1-value-types.json. Do not repair them.

The native-selected p1 prompt is now evaluating the same six-view requests with
the original run's pinned ASR evidence on physical175 card7. Its supervisor
will score both completed conditions. Physical4 original probe has exited;
physical5 native full TEST remains running. The previously registered native
p1 TEST supervisor still waits for its exact baseline before using card6.
No devices0–3 or weight-training queues are used.

Six-view scorer tests cover complete groups, request custody, semantic
rebindings, missing candidates, Noul question contrasts, invalid null,
integer-vs-boolean typing and constant answer controls. Twelve new tests pass;
the three existing consumption tests also pass. A gold-output fixture on all
384 actual requests gives 64/64 all-six-correct, proving the fixture conversion,
not model quality. Two supervisor tests cover configuration changes and
single dispatch with paired scoring.

## Native SLURP

Original real metadata has 11514/2033/2974 annotated utterances and
50628/8690/13078 recordings in train/devel/test. Repeated entity types appear
in 380/74/85 utterances, so first-span extraction would lose labels.
Raw intent aliases differ from scenario/action concatenation in 162/30/37
utterances; all evaluation scenario/action pairs occur in TRAIN.

New composite release preserves scenario, action, every type/filler occurrence,
native spans/tokens, raw intent, speakers and recording identity. Runtime
requests exclude reference text and gold. All recordings of a slurp_id share
one source group. Seven parser/export tests pass. Full release export runs
detached and hashes every real recording; it has not completed yet.

Pinned upstream scorer source: pswietojanski/slurp
8eb16545762be97ace75334109d73824217311f1, stored under shared repos. Upstream
metric code remains unmodified. tabulate0.8.7 fails on Python3.11; formatting
dependency upgraded to0.9.0, with the exception recorded. Distance dependencies
jiwer2.0.0/textdistance4.1.5 remain fixed in a separate scorer environment.
Seven official metric fixtures pass; malformed composites are rejected.
All 16521 real metadata rows match upstream native filler conversion exactly.
This is not a full benchmark scorer/inference result.

Next gates: finish six-view p1 probe and native full TEST; finish SLURP native
release audit; implement complete prediction-ID gating and malformed-output
policy around official scoring; run frozen native SLURP TRAIN engineering
probe before locking full TEST execution. Actual six-view teacher learning and
cross-dataset transfer remain separate future experiments.
