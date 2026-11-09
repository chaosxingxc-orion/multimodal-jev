# Two-image CPU iteration scheduler

tools/jev/run_container_audio_rl.py adapts the existing sequential scheduler to
the actually available runtimes: local official ARM vLLM CPU sampling and remote
NGA-aligned verl CPU actor/export. Runtime identities/resources are fixed in
configs/runtime/container-cpu-v1.json. No NPU devices are mounted or discovered.
This is not a distributed verl worker implementation or accelerator validation.

Run only from an immutable source snapshot inside a dedicated shared experiment
root. Each iteration launches a fresh sampler, waits for exact Docker exit0 and
application receipts, then updates the fixed-base actor with its parent adapter
and AdamW state, and verifies a separately published export before continuing.
Reuse the existing schedule/data/lineage/parity gates. All sources and runtime
configuration enter persisted identity; container commands are fixed per stage.

Launch intent is written before submission. A timeout causes observation of
the same deterministic container name; no automatic resubmission. Once observed,
the full container ID and image are fixed. Observation failures retry every30s;
only authoritative terminal state stops the stage. Untracked incomplete output
is rejected. Failed runs/artifacts remain for inspection. A stage cannot be
claimed complete from application output while Docker remains running.

All shared assets are read-only except the experiment root and the current
iteration's export destination; the executable source is separately read-only.
Ascend image entrypoint remains active for CPU driver-library setup. Runtime
version differences remain documented and actor/rollout probability checks stay
mandatory. Memory limits are allocations, not observed peaks or guarantees.

Unit fixtures cover command identity, timeout observation without duplicate
launch, failed states and two-iteration checkpoint/parent handoff. A lightweight
real transport probe checks sequential local and remote image execution/imports,
mount permissions and exit/result receipts. That probe is infrastructure evidence,
not real inference or optimizer continuation. The full scheduler still needs its
own two-iteration actual-engine run after existing numerical gates finish.

The first transport attempt was rejected by the local Docker CLI before
container creation because it does not accept --pull. Inspection confirmed
the deterministic name absent; its observation-only supervisor was stopped and
the failure retained. The revised executor checks the pinned image exists first,
uses --pull=never only on the remote Docker, and treats this explicit CLI parse
error as terminal. A separate v2 probe is running; see
[launch](../docs/engineering/container-transport-v2-launch.json).

The v2 local container exited0, verified CPU torch import and a read-only source
mount. Remote actor container creation then failed with Docker exit125:
no space left on device under /var/lib/docker/overlay2. /var was984GiB full;
inode capacity was available. Two inspections found the exact intended name
absent. The observation-only supervisor was stopped and the failure retained;
existing training containers were not stopped. See
[partial result](../docs/engineering/container-transport-v2-result.json).
No remote probe or full scheduler execution is claimed. This project's container
writable layers are too small to resolve the host-wide shortage; unrelated
projects' containers/images/files were not deleted. Retry requires disk capacity
and a separately named run after inspecting the prior launch state.
