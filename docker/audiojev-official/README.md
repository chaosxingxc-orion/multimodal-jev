# Official AudioJev baseline image

Prepared, **not built or runtime-validated**. Current accessible hosts are
ARM64 Ascend. The pinned CUDA12.8 torch2.8.0/torchvision0.23.0 CPython3.12
publisher wheels are x86_64; no matching ARM64 wheels appeared in those indexes
at inspection. This recipe targets Linux x86_64 and requires NVIDIA allocation
for inference. It does not alter the NGA-aligned RL image or port the backend.

The Python base is pinned to the Linux amd64 manifest digest queried from Docker
Hub. Direct versions satisfy the pinned upstream pyproject.toml. OS packages and
transitive Python dependencies are not a complete lock; successful builds retain
pip freeze/inspect, and the image ID must be frozen before experiments. Build
success still requires actual resolution, native imports and pip check.

On the allocated x86_64 Docker host, from an immutable project source snapshot:

```bash
docker build -f docker/audiojev-official/Dockerfile -t multimodal-jev-audiojev:official-v1 .
docker image inspect multimodal-jev-audiojev:official-v1
```

Record the immutable image ID and use it in subsequent commands. The default
command checks imports and CUDA BF16 SDPA only. Expose only the allocated GPU
using Docker's NVIDIA GPU selection; inside the container it is cuda:0. Keep
the clean official repository pinned to commit
8eeeab5e77d2c410fd6058e7bdf2b4125f11903e and mount it read-only at its registered
absolute path, alongside read-only models and datasets and a separate writable
experiment directory. `git` is installed so the runner can verify source state.

Then execute the existing `tools/jev/infer_audiojev_mmar.py` with its committed
config, via a detached supervisor retaining launch intent, exact Docker ID,
terminal state and logs. Full1000-item inference and independent official scoring
are still required by `../../experiments/archive/raw/frozen-baselines/audiojev-mmar.md`. A successful environment
check must not be reported as a reproduced baseline score.
