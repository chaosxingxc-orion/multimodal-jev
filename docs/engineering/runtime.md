# Multimodal Jev runtime release

2026-11-10 update: the user allocated physical NPU 4–7 on host `npu` (175).
The original image passed [real NPU runtime admission](npu-audio-runtime-v1-result.json):
four-device arithmetic and a single-device BF16 Qwen3-ASR audio forward/backward
plus one LoRA update. Native verl training, NPU vLLM inference and distributed
model training remain separate gates. Earlier build-time statements below are
historical observations; see [allocation](npu175-allocation.md) for current scope.

The first runtime is an additive child image of the image used by the NGA Qwen3.5/verl work. It pins the base by digest:

- base tag at inventory time: `nga-verl-qwen35-v030:20261029`
- base digest: `sha256:8190b3ec51929baca2388452039ccf05d681c01b7c91cc55dd1e7ac5570a76b7`
- platform: Linux ARM64, Huawei Ascend
- Python 3.12.13, torch-NPU 2.10.0.post7, Transformers 5.14.1, vLLM 0.30.0+empty, vLLM-Ascend 0.30.0rc2, verl 0.8.0.dev0, Ray 2.48.0
- verl source commit: `e9405d70b77d56a23e72bb2e40a17b906856e913`

The inventory was collected inside the image on `npu`; it is saved in `.local/runtime/nga-base-inventory.json`. The base registers `Qwen3ASRForConditionalGeneration` and contains the Transformers Qwen3-ASR implementation. It does not contain the separate `qwen-asr` package or librosa. Audio preprocessing therefore uses `soundfile`/the official Transformers processor first; adding the Qwen ASR package requires a separate compatibility review.

Build on an NPU host where the base digest is already loaded:

```bash
docker build --pull=false -f docker/multimodal-jev/Dockerfile -t multimodal-jev:verl-v030-20261109 .
```

The build checks every required package version and installs only the local package. It never runs pip dependency resolution. The image does not contain model weights or datasets.

The runtime mounts read-only datasets and models, and a separate writable experiment root. A run must record the image digest, model revision, dataset manifest, config digest, NPU allocation, and exact command. Do not mount a live unrelated worktree as executable code.

The child image has been built on `npu` and passed package/import checks and a real FSC recording on CPU. See `runtime-receipt.json`. NPU forward/backward and verl audio rollout validation remain required before training.

## Compatibility findings

The original `Qwen/Qwen3-ASR-{0.6B,1.7B}` checkpoints have a nested
`thinker_config`; the installed Transformers implementation expects flattened
HF configuration and converted weight names. Use the separately downloaded
**official** converted checkpoints, without changing the original assets:

| Model | Revision |
| --- | --- |
| `Qwen/Qwen3-ASR-0.6B-hf` | `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c` |
| `Qwen/Qwen3-ASR-1.7B-hf` | `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` |

All 18 files (5,664,027,655 bytes) passed source SHA256/git-blob verification.
Shared directories use `models/Qwen__Qwen3-ASR-0.6B-hf` and
`models/Qwen__Qwen3-ASR-1.7B-hf`. Local download receipts are in
`.local/runtime/hf-model-download/`. Models run with local-files-only loading.

Even CPU import checks need the host Ascend **driver libraries** mounted read-only;
Torch/Transformers auto-import torch-npu. This does not require mounting NPU
character devices. The entrypoint sets the toolkit and driver library paths.
Turning off Torch backend auto-loading alone is insufficient: Transformers still
imports torch-npu. No NPU device is allocated for CPU smoke tests.

```bash
docker run --rm --cpus=4 --memory=16g \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  multimodal-jev:verl-v030-20261109 \
  python3 /opt/multimodal-jev/tools/jev/runtime_smoke.py
```

The pinned verl default multimodal dataset / vLLM async server exposes image and
video paths, not an audio training path. Qwen-ASR model registration does **not**
prove audio RL support. Required work: a custom audio dataset, rollout request
adapter, actor/rollout input equivalence tests, log-prob alignment, and a real NPU
forward/backward/rollout smoke before training. Existing NGA/Omni jobs occupy both
NPU hosts; no job was stopped or device reassigned.

## Built release receipt

Built image ID: `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`
(local Docker image ID; not a published registry manifest digest).
[Receipt](runtime-receipt.json) records source file hashes and the CPU probe.
The real FSC clip was transcribed as `Change language.`; decoder features were
`[1, 1024]`, audio features `[24, 1024]`, all finite. This is one smoke input,
not WER, SLU accuracy, a trained head, or evidence of RL improvement.

Final build includes `multimodal_jev.evaluate`, the FSC builder and reward contract.
Runtime `README.md` and config snapshots are identified by their hashes in the receipt;
subsequent documentation changes in Git do not change the built image.
