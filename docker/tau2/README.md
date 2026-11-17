# Pinned τ² core runtime

This image isolates τ² dependencies from the existing verl/vLLM environment.
It provides core environment/orchestrator imports, not voice extras.

Build context contains `Dockerfile`, `upstream/` (MIT source revision
`033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f`, including `src`, `pyproject.toml`,
`README.md`, `LICENSE`) and `wheels/`. The completed context, source/wheel hashes
and resolved package freeze are preserved at:

`/mnt/cephfs/xingchao/experiments/multimodal-jev/tau2-runtime-build-v2`

To reproduce the recorded dependency set, verify `artifact-hashes.json`, then
build the preserved context with `docker build --network=none`. Resolving new
wheels from the declared ranges creates a different environment and needs a new
receipt. This build does not claim upstream `uv.lock` fidelity.

Run the fixed environment check with the image ID from
[`tau2-environment-v2-result.json`](../../docs/engineering/tau2-environment-v2-result.json),
network disabled, the complete project source mounted read-only at `/project`,
shared `datasets/tau2-voice/data` read-only at `/data`, and a new writable evidence
directory at `/evidence`. Set `PYTHONPATH=/project/src`, `TAU2_DATA_DIR=/data`,
`PYTHON_DOTENV_DISABLED=1`; the image entrypoint is Python:

```text
-B /project/tools/jev/tau2_environment_probe.py --output /evidence/run
```

Use no provider credentials for this environment-only check. See the
[protocol](../../experiments/archive/raw/complex-deferred/tau2-environment-bridge.md) for its precise scope.
