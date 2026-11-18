# Ascend vLLM audio sampling and probability gate v1

Use allocated physical device 4 on `npu` (175), exposing only that accelerator
plus the required manager nodes, no privileged mode. Use the existing pinned
NGA-derived runtime image `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`.
No live CPU run or source snapshot is modified.

Sample the same first TRAIN recording and its six registered Jev views as
`vllm-cpu-rollout-sft-v4`, using its exact merged SFT initializer, processor and
sampling settings: n=2, max 32 tokens, temperature=1, top-p=1, top-k=-1, seed 17.
Use actor feature inputs, TP1, eager mode, context 2048, max one sequence,
memory utilization 0.5. Change engine dtype to BF16 for the Ascend probe.
The initializer is only a development model, not an academic baseline.

Persist all 12 responses with sampled token log probabilities and policy identity.
After sampler exit, score every response on the same physical device using the
identical merged HF checkpoint, BF16 and eager attention. Preserve the existing
maximum absolute token log-probability error gate of **0.02**. A precision/backend
failure is recorded at that threshold, not admitted by relaxing it. Original
CPU/FP32 audit behavior remains the default.

Report sampler and audit Docker/application exits separately. This is NPU
inference/probability validation, not a verl optimizer update, distributed model
test or task score. No checkpoint export or model selection occurs here.
