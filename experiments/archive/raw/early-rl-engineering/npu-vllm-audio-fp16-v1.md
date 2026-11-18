# NPU FP16 actor/rollout diagnostic v1

The BF16 NPU rollout had max absolute token log-probability errors 0.1861983
against HF BF16 and 0.1274023 against HF FP32 on all 12 saved responses. Neither
passes 0.02. A higher precision actor alone therefore did not resolve the gate.
Do not infer a specific kernel bug or claim precision is the sole cause.

Register a separate FP16 condition, using the same initializer/checkpoint,
TRAIN recording/six views, sampling distribution, seed 17, n=2/max32 and
actor_features as npu-vllm-audio-v2. Change only engine dtype to float16, use
allocated physical NPU 6 and the same pinned image/source except the explicit
FP16 audit CLI option. The HF audit uses FP16 on the same device after the
sampler exits; compare all sampled response tokens with threshold 0.02.
No threshold tuning, RL update or evaluation/model selection occurs here.

Explicit shutdown is required; Docker/application success, complete 12-response
artifact and hash validation precede dependent audit. Retain each precision
condition and failure. Success establishes only this small inference/probability
gate, not optimizer stability, distributed actors, full training or generalization.
