# FSC frozen audio development pilot v1

Registered before launch, 2026-11-09. This pilot tests the executable closed-label
control required before dynamic-schema/RL comparisons, not a paper reproduction.

Hypothesis: frozen Qwen3-ASR projector representations contain FSC command
information usable by a learned linear head. No candidate definitions are fed
into this closed-label control; success cannot establish schema understanding.

Selection: original train and valid only, native view only. Sort example IDs by
SHA256 within each of the 31 train labels. Select 4 train and 2 valid per class
(124/62 recordings). Labels and feature normalization derive from train only.
Verify disjoint speakers, source groups and audio hashes. No test inference.

Model: official `Qwen/Qwen3-ASR-0.6B-hf`, revision
`7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`, CPU float32/eager, four threads.
Mean-pool projected audio tokens for one recording at a time. Freeze the full
model. Standardize features using train means/std (floor 1e-6). Linear 1024→31
head; full-batch CE + AdamW, LR .001, weight decay .01, 100 updates per seed
17/29/43. No validation-based hyperparameter or checkpoint selection.

Report train/valid accuracy per seed, first/last loss, counts, runtime, selected
IDs and source/config/weight hashes. Uniform prediction expected accuracy 1/31;
constant-class accuracy 2/62 on this balanced validation subset. Small validation
size limits conclusions. Full official split and multi-task replication remain
required even if this pilot succeeds.

The output records per-clip features for restart, atomic progress, run lock,
head checkpoints, predictions and terminal exit code. Cache identity includes
code/config/manifest/weight hashes. Do not overwrite a completed run.

## Environment-only amendment before any completed head update

v1 extracted all 186 features, then failed in the first AdamW step because
Torch-NPU's automatic foreach capability query initializes a missing NPU.
v2 sets `foreach=False, fused=False`; data, loss, hyperparameters and selection
are unchanged. Feature cache adoption verifies the old/new extractor and config
are identical and the only baseline code change is the optimizer flags. Per-file
hashes and the old failure remain in the v2 cache-adoption receipt.

## Observed result

[Receipt](../../../../docs/engineering/fsc-frozen-audio-pilot-result.json), verified again
against saved prediction IDs and official validation targets:

| Seed | Correct / validation total | Accuracy |
| --- | --- | --- |
| 17 | 26 / 62 | 41.94% |
| 29 | 29 / 62 | 46.77% |
| 43 | 31 / 62 | 50.00% |

Mean 46.24%; all heads reached 100% train accuracy on 124 examples. This gap
reinforces the need for more data and independent evaluation. No claim of
schema understanding, cross-task generalization or improvement due to RL.

v1 spent about 177 seconds extracting the 186 recordings before its optimizer
failure. v2 reused those features and completed head training; its recorded
1.17 seconds excludes model loading, manifest/weight hashing and v1 extraction.
Neither value alone is end-to-end system cost. Docker exit code and run terminal
both equal zero for v2. Checkpoints, features and predictions remain outside Git.
