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

## First real sampling completed; processor discrepancy

The waveform run exited 0 and produced all 12 responses for six TRAIN views,
80 response tokens including EOS. Hash and complete-group checks passed; see
[sampling receipt](../docs/engineering/vllm-cpu-rollout-v1-result.json).
Several Choice groups have unequal task rewards, so a nonzero GRPO signal is
possible, but no learning result is established before actor verification.
The original NGA-aligned actor is running on npu under the independent root
vllm-cpu-actor-sft-v1, full batch of 12, four CPU threads, 48 GiB memory limit.
Its detached supervisor checks the exact successful upstream Docker identity
and rollout artifacts before launching; failed parity never triggers a retry.

The complete vLLM processor check failed for waveform input: 186 feature frames
versus HF's 185 on the first recording. Source inspection shows the inherited
Qwen3Omni processor right-pads waveforms to the next hop boundary for caching.
The earlier direct bundled-processor check did not exercise that transformation.
The .02 actor probability threshold is unchanged, and this original trajectory
is retained without altering its input or preprocessing after sampling.

A separately identified actor_features option now passes the actor's exact
valid log-mel frames through vLLM's precomputed feature contract. The initial
feature-path check failed with StopIteration: upstream invokes a dummy audio
placeholder through its waveform processor without a waveform before merging
passthrough features. The narrow vllm_audio_features compatibility helper bypasses
only that call when processor_data is empty and explicit audio features exist;
waveform behavior remains upstream. This is a local compatibility change, not an
unmodified official implementation. vllm-cpu-processor-parity-v3 checks it before
any new feature-mode rollout. Existing schedules default to waveform.

The v3 processor check exited 0: both tested audio lengths have exactly equal
prompt token IDs and zero feature error (185/92 valid frames). See the
[receipt](../docs/engineering/vllm-cpu-feature-parity-v3-receipt.json).
The feature-mode engine run is now separately launched as
multimodal-jev-vllm-cpu-rollout-sft-v2, ID
778ca3dfe9512f8ba311aea9d4f3c50ac23623ea46144ecda88b0754123cd320.
It retains the same weights, TRAIN views, sampling settings and resource bounds;
only the explicitly recorded audio_input_mode and associated compatibility
helper differ. Processor parity is not proof of model log-probability parity.

## Actor gate failure and feature admission correction

The waveform actor v1 has now exited 1: maximum token log-probability error
0.16855794191360474 exceeds the unchanged 0.02 gate. No optimizer step ran.
See [actor failure receipt](../docs/engineering/vllm-waveform-actor-v1-result.json).
A separate read-only HF merged-checkpoint audit scores each saved response to
distinguish preprocessing/backend error from full-batch and adapter effects.
The audit preserves all token errors and completes all responses before reporting
its fixed-threshold verdict; it cannot establish full-batch actor parity.

Feature engine v2 persisted an application terminal failure: the renderer requires
enable_mm_embeds for dictionary audio features. Its engine workers remained live
after the exception; they were explicitly stopped after saving failure evidence.
The failed run and supervisor are retained. No actor v2 was launched.

Feature engine v3 initializes enable_mm_embeds=True only for actor_features and
rejects switching input modes after construction. The checkpoint, six TRAIN views,
sampling distribution, resource limits and 0.02 gate are unchanged. Container
7ca24f88f3916682cbca5e32a1338708670f18b7411f6cecff3f8424f8e2f4f2 is a new run;
its source/config/protocol hashes are retained under vllm-cpu-rollout-sft-v3.
A detached supervisor under vllm-cpu-actor-sft-v3 validates successful upstream
termination and artifacts before launching the actor. Actor phase heartbeats now
identify parity, gradient, backward and optimizer steps and retain per-response
parity errors. This instrumentation does not alter rewards, objectives or gates.
