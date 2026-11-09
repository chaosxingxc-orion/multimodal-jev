# Pretrained Qwen ASR policy diagnosis

Use the first FSC TRAIN source group, two fixed positive/negative Noul questions,
two samples each, seed 17, temperature 1, top_p 1, no top-k truncation, max 32 new
tokens. Do not select examples by observed success or use TEST. The purpose is
to determine whether pretrained ASR already emits the requested JSON format;
all malformed outputs keep their failure reward. No optimization in this probe.

HF CPU generation is an explicitly labeled diagnostic, not the vLLM RL backend.
The production rollout entrypoint is `sample_audio_policy.py` / `VllmAudioRollout`.
If all rewards are equal, GRPO has zero relative advantage; the next action is
supervised task initialization, not assigning artificial varied rewards.

Observed: 0/4 JSON objects; all four responses remained ASR transcriptions
(`language ... <asr_text>Change language.`). Reward -1 for all. This very small
probe demonstrates the failure on these inputs, not a universal inability to
follow instructions. [Receipt](../docs/engineering/pretrained-policy-probe.json).
The SFT initialization protocol was registered after this development evidence.
