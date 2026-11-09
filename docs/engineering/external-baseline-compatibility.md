# External baseline compatibility audit

The shared AudioJev checkpoint is available, and the official inference repository
is now pinned at `8eeeab5e77d2c410fd6058e7bdf2b4125f11903e`:
`/mnt/cephfs/xingchao/repos/jev/SihanLv__AudioJev-Inference`.
Source: https://github.com/SihanLv/AudioJev-Inference/tree/8eeeab5e77d2c410fd6058e7bdf2b4125f11903e
Code license: Apache-2.0. Weights retain the Qwen Research License.

| Component | Official AudioJev | Current Jev RL image |
| --- | --- | --- |
| torch | >=2.8,<2.9 | 2.10 + torch-npu 2.10 |
| Transformers | 4.57.6 | 5.14.1 |
| PEFT | >=0.17,<0.18 | inherited pinned runtime |
| Backend | explicit indexed CUDA device; BF16 SDPA | Ascend NPU; CPU checks available |

`audiojev_inference/backend.py:OmniBackend.__init__` explicitly rejects CPU and
NPU devices. Do not claim official inference succeeds merely because model files
are local. Do not install its pinned dependencies over the RL image. A CPU/Ascend
port needs separate provenance and numerical comparison with the official method;
a faithful environment reproduction needs a separate image and suitable device.
The official scorer uses next-token candidate-label logits, not generated JSON.
Its request-scoped audio feature cache is another behavior to preserve and check.

The public API's `_prepare` sorts unique Choice/Noul descriptions by UTF-8 text
and strips whitespace; it also rejects blank descriptions. The model card and
paper use original-order evaluation. MMAR includes four blank-candidate
questions and six references that are not a single option. The dedicated
[MMAR protocol](../../experiments/audiojev-mmar.md) preserves these inputs and
calls the official renderer/backend directly. No model inference has been run
through this entrypoint yet. A canonical service score must not be labeled as
the published original-order result.

SFC data and the unmodified official scorer are usable without this CUDA backend.
See [SFC protocol](../../experiments/sfc-external-evaluation.md) for the input
release, scorer differences, and remaining model reproduction gates.
