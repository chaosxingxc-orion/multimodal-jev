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
