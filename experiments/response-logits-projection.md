# Response-position vocabulary projection probe

The full-batch actor OOM motivates a memory optimization without changing batch
size or the objective. The pinned Qwen3-ASR forward natively accepts
logits_to_keep. Retain response width + 1 trailing positions: the preceding
prompt position predicts the first response token; the final position is unused.
The transformer still sees the entire prompt/audio/response sequence. Only
unneeded vocabulary projection positions are omitted. The default remains off.

Before enabling this in real training, compare full and shortened projection
on the same complete three-response TRAIN fixture, unequal response lengths,
same initial adapter and fresh identical AdamW states. Use the existing strict
parameter/gradient/moment thresholds (atol 2e-6, rtol 2e-4), pre-update log-prob
gate 3e-4 and loss difference <2e-5. Persist every comparison error before
asserting. Also require corrupted saved probabilities to reject the update with
unchanged parameters and zero optimizer states. Do not loosen any threshold.

Use tools/jev/actor_response_logits_smoke.py in the NGA-aligned CPU image, four
cores and 16 GiB. This controlled-fixture test is independent of live v5, whose
immutable source, full projection and 128 GiB resource-only retry remain intact.
Success here permits a separately identified actual-vLLM actor test, not an
efficacy claim. Activation checkpointing and microbatch changes are not included.

The independent probe is running as npu CPU container
2be0767333a2aecd49b67d219e862da741e67d1c37dbaa890b649793aee9e833,
root actor-response-logits-smoke-v1. See
[launch receipt](../docs/engineering/response-logits-v1-launch.json).
No current training configuration enables response_logits_only.
