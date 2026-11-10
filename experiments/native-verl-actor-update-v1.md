# Native actor microbatch autograd comparison

After exact successful native-verl-actor-forward-v2 completion, replay the same
12 saved initializer-policy responses used by vllm-cpu-actor-sft-v5. Start both
implementations from the same immutable SFT adapter and fresh AdamW LR1e-5,
rank4/alpha8, zero weight decay, float32, full projection, clip1. Use all six
TRAIN views, unchanged pair reward weight.2 and standardized GRPO advantages.
These old responses are a numerical replay from their matching initializer,
not fresh samples or continuation data for another policy.

Run the current custom actor once, capture pre-clipping gradients, parameters
and AdamW moments; restore exact initial adapter tensors, then run the native
DataParallelPPOActor microbatch forward/backward with the explicit CPU log-prob
dispatcher. Use the same verl loss, complete batch and saved rollout old
log-probs. Enforce the existing.02 rollout parity gate before each update.
The native update_policy loop is deliberately not exercised: its device and
normalization integration remain separate work.

Fix parameter/gradient/moment atol2e-6 and rtol2e-4 before execution. Require
the reference to update; compare every trainable tensor, both moments and
pre-clipping gradient. Save both tensor snapshots for independent auditing,
including failed comparisons. Do not relax tolerances after observing errors.
One Gloo rank only supports native constructor initialization; no Ray/FSDP
or multi-rank ASR training claim follows. No model export or evaluation uses
this replay adapter.

Use the existing pinned NGA image on four remote CPU threads with256GiB and
no NPU devices. Queue after the exact successful forward-v2 container. Retain
immutable source/config, launch intent, exact container and terminal receipts;
no automatic retry or fallback on failure.
