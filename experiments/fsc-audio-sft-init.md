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
