# Audio Jev SFT initialization development v1

Motivation: pretrained Qwen3-ASR generated transcriptions on all four sampled
Noul responses in the fixed training-only probe. All received -1 format reward,
so those groups yield zero GRPO advantage. Initialize the structured response
behavior with supervised examples before comparing SFT/GRPO/pair-GRPO.

Select one TRAIN native source group per FSC class, minimum SHA256(example_id),
then all six views per group (186 examples, 31 recordings). No validation/test
labels used. Shuffle seed 17. One pass, gradient accumulation 4, final partial
batch normalized by actual size; 47 updates. Each sample's response-token CE is
normalized within that sample, then averaged over accumulation items.

Qwen3-ASR-0.6B-hf fixed revision; LoRA rank 4, alpha 8, q/v projections only in
text decoder; no dropout, AdamW LR 1e-4, weight decay 0, gradient clipping 1.
CPU float32/eager/four threads. This is a development initialization, not final
three-seed training or academic replication. Save adapter and optimizer/RNG resume
state each update. Keep all malformed sampled outputs in later evaluation.

After training, use separate official valid recordings for schema behavior and
compare against the untrained checkpoint. Do not claim generalization from
fitting these 31 recordings. Full-scale matched SFT and RL comparisons remain.

## Completed initialization and export (2026-11-10, Asia/Shanghai)

Training exited zero at 04:48:31 local time. Terminal/result receipts prove 47
updates, 186 decisions and 31 training recordings. Source snapshot hashes match
the recorded identity. See [the verified receipt](../../../../docs/engineering/fsc-sft-init-v1-receipt.json).
Reported training time was 13,934.74 seconds on CPU; this is not NPU training
or a quality measurement.

Export also exited zero. Adapter versus merged/reloaded log-probs on two fixed
TRAIN examples differ by at most 7.33137e-6 (atol 3e-4, rtol 1e-5). Shared model:
`/mnt/cephfs/xingchao/models/multimodal-jev/fsc-sft-init/6a25ad1b89cd956ecc5f1d8234f98987e16f2f14eb1a2a39babe3bd854aaaae8`.
The old exporter snapshot left its directory at 0700 and weights at 0600.
Shared read/traverse permissions were repaired without changing file contents
or receipts. Both merged and adapter weights are readable by the workspace user.

FSC/SLURP development inference started after the training container exited.
Results remain pending; training loss is not evidence of generalization or RL
improvement. The immutable comparison finalizer observes those exact container
IDs and scores only complete, exit-zero prediction sets. Its output root is
`/mnt/cephfs/xingchao/experiments/multimodal-jev/sft-development-comparison-v1/run`.
It produces `fsc/comparison.json` and `slurp/comparison.json` with common
base/candidate metrics and the preregistered speaker-cluster bootstrap.
