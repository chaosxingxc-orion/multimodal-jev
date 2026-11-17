# Qwen3-Omni probe v2: NPU worker spawn

V1 container484b41074effe937668a69cfb43b8fb63fe66e33b41d923cae09ac3c5544b981
and application exited1 before predictions. Root cause in worker initialization:
`Cannot re-initialize NPU in forked subprocess ... use the 'spawn' start method`.
The pinned vLLM env registry defaults VLLM_WORKER_MULTIPROC_METHOD to fork.
Retain all v1 logs, source, device time and failure. No partial accuracy.

New root frozen-qwen3-omni-direct-train1group-v2 uses exact v1 source4884a0a,
model revision and audit, source group, instruction, BF16/TP2/eager, seed17,
context4096/max128tokens. Only add explicit container environment
VLLM_WORKER_MULTIPROC_METHOD=spawn. Root paths differ for custody; prompt bytes
and dataset contents do not. Same physical4/5, image,3600s cap; inspect exact
old terminal and current allocation before dispatch. One attempt, no auto retry.
This runtime fix does not establish model task capability or HF/vLLM parity.
