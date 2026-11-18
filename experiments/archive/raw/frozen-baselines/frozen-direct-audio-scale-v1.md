# Direct-audio Qwen ASR scale probe

2026-11-10, after explicit user end-to-end correction. No transcript cascade.
Question: can the already downloaded official Qwen3-ASR-1.7B-hf checkpoint
produce task values when directly given audio and the unchanged task definition?
This compares scale, not prompt tuning or a claim of instruction capability.

Use the previously locked first64 FSC TRAIN source groups and original
frozen-decision-prompt-v1.txt. Audio-only input plus schema, HF eager BF16,
greedy n1, seed17, max128, context4096. No text_model, transcription evidence,
adapter or weight update. Model revision bcd2b5b7f32b480ab5790554cfa8347f246a14f3,
verified local asset hashes before/after. Same pinned NGA-aligned NPU image.
Physical175 card4 only, one-hour application deadline, immutable run snapshot.

Report joint accuracy, JSON/schema validity, raw failures and actual cost.
The earlier 0.6B TRAIN64 direct run (0/64) is the reference; do not compare
TRAIN probe metrics as official TEST scores. Do not auto-launch full TEST or
teacher calls. If there is no valid task feedback, record no-signal rather
than inventing group advantages or adding a text decision model.
