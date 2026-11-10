# Full FSC frozen audio control v1

Registered before launch. Extend the prior pilot to the entire official train
(23,132 recordings) and valid (3,118) splits. No test inference or tuning.
Same model revision, pooling, train-only feature standardization, optimizer,
seeds 17/29/43, 100 full-batch updates, LR .001, weight decay .01 as the pilot.
CPU eager float32, 4 threads, unchanged source audio. This is the closed-label
frozen Qwen ASR control, not reproduction of an external paper's pretrained model.

All native examples included; no per-class balancing. Report natural-frequency
accuracy, per-class recall/macro recall, counts and predictions. Do not compare
full-valid accuracy to pilot-balanced-valid accuracy as a controlled learning curve.
Any follow-up tuning must use valid and will require a new preregistered test run.

Run detached with immutable source snapshot, per-recording feature cache,
identity hashes, progress and exit receipt. Prior pilot cache is not implicitly
trusted or adopted. Expected CPU feature extraction is hours; the live container
and progress, not the presence of a lock file, determine execution status.

## Completed full-valid result

Exact container1c7bcdbca0655a09f280ff2fd3e1497a9b91eb5e1bb07055cd79918253c120ae
exited0 after26,250 feature extractions and three100-step head fits. Model and
manifest hashes, complete selected IDs and all3,118 predictions per seed were
independently verified. See [receipt](../docs/engineering/fsc-frozen-audio-full-v1-result.json).

| Seed | Natural-frequency accuracy | Macro class recall |
| --- | ---: | ---: |
|17|94.3554%|94.4201%|
|29|93.9064%|93.9997%|
|43|94.1950%|94.4179%|

Mean accuracy94.1522%, sample standard deviation0.2275 percentage points across
three seeds; mean macro recall94.2792%. These seed summaries are not confidence
intervals over speakers. Majority-class accuracy is10.1988%. Elapsed runtime
was26,153.94 seconds (about7.27h); this includes shared feature extraction and
the three linear fits. Per-class support/recall and prediction/head hashes are
retained in the receipt. No test inference was performed.

The frozen acoustic representation supports strong closed-label FSC classification
with the full official training split. This control does not consume dynamic
candidate definitions and cannot demonstrate candidate rebinding, unknown
handling, slot extraction or cross-task transfer. The current31-record SFT/RL
initializer uses far fewer recordings and a different output objective; the
accuracy gap does not isolate an RL or Jev architecture effect. It motivates
retaining this acoustic/label control in the final comparison without replacing
the required budget-matched dynamic-schema and external-task experiments.

A separate read-only CPU torch2.10 forward reloaded all26,250 cached finite
1024-dimensional features and all three saved heads. Recomputed training-only
mean/std matched exactly; all3,118 validation predictions and confidences per
seed reproduced exactly (maximum confidence error0). Script and receipt are
retained with the experiment. This validates cached-feature/head persistence,
without recomputing the encoder or fitting another model.
