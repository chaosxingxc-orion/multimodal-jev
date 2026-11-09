# Fixed FSC schema development evaluation v1

Freeze before evaluating the original or trained policy: one official VALID
recording per class by minimum SHA256(native example_id), all six schema views.
31 source groups, 186 decisions, separate from the 31 training recordings and
official speakers. Selection release:
`87d0e85343d8bfc6fbbc213263fcaed3de2170995480b10d3186ef6ebd879a68`.
No test use. This is development evidence; final claims need full independent test.

Compare original Qwen3-ASR-0.6B-hf and the completed SFT-init adapter (47 updates)
on exactly these same inputs. Greedy decoding, 64 max new tokens, no constrained
JSON decoding or answer repair. Record raw output, token IDs, length limit and
latency. Malformed output is a format failure, never semantic null. Strict JSON
rejects duplicate keys, extra keys and nonfinite values. SFT model gets no access
to the reference transcript/target during generation.

Report format rate, per-variant accuracy, valid schema rate, null precision/recall,
risk/coverage, candidate-permutation and rebind both-correct rates. Derivatives of
one recording are correlated: no claim of 186 independent recording samples.
All-null, constant first-candidate and gold-oracle controls validate the scorer.
Native closed-set acoustic head accuracy is not directly comparable to the
aggregate of these six different task views.

Do not select a checkpoint by observed outputs: evaluate the single completed
SFT-init checkpoint. Follow-up data/hyperparameter changes require a new run ID.
Model export must pass loaded-adapter versus merged/reloaded token log-prob
comparison before use in vLLM. A CPU export pass still does not prove NPU rollout.

## Scorer revision 2 (before completed model evaluations)

Add explicit format-valid/schema-valid rates, malformed versus absent counts,
Noul opposite-question both-correct rate, and per-variant target-present/null
strata. Reparse saved raw text when available; parsed convenience fields cannot
override raw output. Duplicate manifest IDs or source-group variants fail.
Generation, selection and decoding remain as frozen above. Running generation
snapshots retain their original scorer. Re-score both completed prediction files
with this same revision into separate artifacts; preserve their original metrics.

Controls on the frozen 186-decision manifest: oracle 186/186, all-null 31/186,
first-candidate/always-true 34/186, malformed 0/186. Noul both-correct is 31/31
for oracle and 0/31 for each other control. This validates scoring behavior only.
Receipt: `docs/engineering/fsc-schema-controls.json`; full generated control
predictions are outside Git in `experiments/multimodal-jev/fsc-schema-controls-v1`.
