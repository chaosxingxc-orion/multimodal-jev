# NPU acoustic-stage diagnosis v1

The FP16 HF/vLLM response probability gate failed at max absolute log-prob
error 0.0615768433 against the unchanged threshold 0.02. Diagnose the upstream
acoustic stages before attempting an optimizer update. This is not an accuracy
experiment or a replacement admission gate.

Use the existing npu-vllm-audio-fp16-v1 checkpoint, six TRAIN views of the same
recording, seed 17, n=2/max32, FP16, actor_features and pinned actor image.
Physical NPU 6 only; 8 CPU cores and 64 GiB RAM; no network. Save first-call
HF and real vLLM worker conv1/2/3, conv_out, ln_post and projected features.
Hooks return None and do not replace any activation. Compare shapes, finite
values and absolute differences; retain complete tensors and hashes. Hooked
sampling must match all saved FP16 response records before treating hooks as
observational. Confirm exact expanded prompt token IDs against HF on all six
views, including explicit context admission. Preserve failed historical runs.

The HF model is unloaded before starting vLLM on the same device. Explicit
engine shutdown and detached supervision retain both application and Docker
terminal status. A single diagnostic invocation is authorized here; no RL,
test tuning, threshold relaxation or full-training claim. Walltime guard is
20 minutes for this isolated diagnostic container.
