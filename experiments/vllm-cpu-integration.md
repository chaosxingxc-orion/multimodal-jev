# Real vLLM CPU rollout integration v1

Registered 2026-11-10 before engine execution. Both Ascend hosts remain occupied
by existing jobs; they are not preempted. Use the official ARM64 CPU image for
vLLM 0.30.0 in an isolated environment, while retaining the NGA-aligned Ascend
runtime for verl actor updates. This can verify real vLLM sampling and a
cross-process update chain but cannot validate NPU kernels or performance.

Official manifest: vllm/vllm-openai-cpu:v0.30.0, ARM64 child digest
sha256:09e656588c2dc572240cb62a22f75681dc640a99c61537bbd55e9eb1dc3d4fbc.
Verify compressed layer SHA256 and uncompressed rootfs diff IDs against the
manifest/config, then verify the loaded image ID matches the config digest.
Record installed torch/Transformers/vLLM versions before execution. The official
CPU torch may differ from Ascend; do not overwrite the Ascend environment or
claim exact dependency parity. Source: the tagged upstream Dockerfile at
https://github.com/vllm-project/vllm/blob/v0.30.0/docker/Dockerfile.cpu.

Use the same completed SFT merged export, TRAIN manifest, first six-view recording,
n=2, temperature=1, top_p=1, top_k=-1, max_tokens=32, seed=17 and FP32 as the
existing SFT smoke schedule. Set generation_config=vllm to prevent checkpoint
generation defaults from silently changing the registered sampling distribution.
No forced output schema or repair. Persist every response token, including EOS,
its vLLM log probability and raw decoded text. Model/data/source mounts read-only.

Bound the CPU engine to four cores, a separate 16 GiB memory limit, 1 GiB KV
cache and one sequence at a time. First inspect actual Qwen3-ASR architecture
support and processor behavior. Failure terminates the run with evidence; any
fix uses a new run/config identity. No HF generation substitution is permitted.

Before a verl optimizer step, require successful real-vLLM receipts, complete
groups and the existing actor log-prob maximum error <=0.02. Do not loosen the
gate to accommodate a backend mismatch. Equal rewards may produce zero updates;
this is an observed outcome, not a reason to fabricate an alternative group.
After a successful update/export, a second fresh vLLM process must use the new
weights. Only then can this CPU chain be called end-to-end verified. Independent
matched SFT/GRPO/pair-GRPO and cross-task evaluation remain required.

## Image and first execution

Image verification completed: 15 compressed layers and their uncompressed diff
IDs match, loaded image ID is
sha256:fdd4a28bb7515717e9ff1988669b4ae7c24588091cf0b73f05abff1f6ff6f87b.
Inventory succeeded on ARM64: Python 3.12.14, vLLM 0.30.0+cpu, torch 2.13.0+cpu,
Transformers 5.17.0, soundfile 0.14.0; Qwen3-ASR architecture is registered.
See the [image receipt](../docs/engineering/vllm-cpu-image-receipt.json).
This differs from the Ascend image's torch 2.10 / Transformers 5.14.1 and needs
the existing actor/rollout numerical gate before any update.

Local Docker container multimodal-jev-vllm-cpu-rollout-sft-v1,
ID 170f55682ff1202e28de69c94a6c8f20c34a866b9e6ec65b0e1442a08c08e148,
has started real engine initialization. Output root:
/mnt/cephfs/xingchao/experiments/multimodal-jev/vllm-cpu-rollout-sft-v1.
Copied source, config/protocol hashes and launch argv are retained there.
Engine initialization is not successful sampling or an optimizer update.

The Docker daemon proxy prevented direct pulls. A verified Registry v2 transfer
using the system CA succeeded; TLS verification stayed enabled. The first
transfer failure and inventory script-name collision are retained in separate
v1 directories. Redundant transfer blobs/tars (8.86 GB) were removed only after
image identity verification; the loaded image and evidence remain available.
