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

## Recovery and instrumented schedule validation

Remote `/var` subsequently recovered to approximately570GiB available without
this project deleting unrelated assets. A separately named v3 transport probe
uses the current executor and additionally imports vLLM locally and verl DataProto
remotely. Its [launch receipt](../docs/engineering/container-transport-v3-launch.json)
preserves the v2 failure. Successful imports remain transport evidence only.
All three exact v3 containers subsequently exited0 with successful application
receipts. Local imports report torch2.13.0+cpu, Transformers5.17.0 and
vLLM0.30.0+cpu; remote imports report torch2.10.0+cpu, Transformers5.14.1 and
verl0.8.0.dev0. Source mounts were read-only in each role. See the
[verified result](../docs/engineering/container-transport-v3-result.json).

The next actual-engine validation is prepared under the shared experiment root
`container-audio-rl-two-step-v1`, with a read-only snapshot of source commit
0e86525 and the pinned two-host runtime. See the
[preparation receipt](../docs/engineering/container-two-step-v1-preparation.json)
for all executable source hashes, config hashes and actual asset preflight.
It has not launched. Require successful v3 transport, completed existing
vllm-cpu-chain-sft-v3 two-update continuation with optimizer step2, and completed
vllm-cpu-mixed-actor-v1 numerical/update checks before launch.

Use the existing six-view FSC TRAIN recording in the registered SFT smoke
schedule for both iterations: seed17 then18, n2, max_tokens32, actor_features,
full projection, float32, standard-normalized pair-GRPO, pair weight.2, LR1e-5,
rank4/alpha8, fresh optimizer on the same SFT initializer. This changes no
algorithm selection and introduces no development-based sample selection.
The second iteration must use the first exported policy and persisted optimizer;
both iterations must complete export verification in fresh processes.

Unlike the legacy two-update supervisor, this run records generation, actor and
export budgets through the current snapshot. After exact six-container exit0
checks and application success, independently verify optimizer states, changed
adapter weights and lineage, then run the trajectory budget audit. Require
complete accounting for both iterations; preserve any no-op as an observed
outcome, not a successful optimizer-update validation. The integration does not
establish algorithm improvement, equal compute, cross-task transfer or a full
distributed verl training worker. The multi-seed comparison remains separate.

The legacy two-update dependency has now completed: exact second-actor Docker
exit0, successful chain/application terminals and matching artifact hashes.
The fresh second rollout used policy119cf54b..., and second-actor probability
error was5.28097e-5. Independent read-only tensor inspection found112 finite
AdamW states at step2 and112 changed adapter tensors relative to step1.
See [chain result](../docs/engineering/vllm-two-step-chain-v3-result.json) and
[saved state audit](../docs/engineering/actor-artifact-step2-v1-result.json).
Both iterations had zero pair bonuses: this demonstrates on-policy continuation,
not an algorithmic advantage. The separate mixed-length actor is still pending;
the prepared instrumented scheduler has not launched.

A detached one-shot launcher now waits for mixed-actor Docker/application
success and a read-only artifact audit, followed by successful SFC CPU and
full-pool sampling/audit completion to release the local CPU allocation. It
checks the prepared source/config hashes before invoking the unchanged scheduler.
After two iterations it verifies all six exact container exits, runs independent
actor step1/step2 audits and requires complete trajectory budget receipts.
Failures stop without resubmission. The initial queue was stopped before launch
when SFC v2 failed; the [replacement queue](../docs/engineering/sfc-v3-dependent-queues.json)
observes SFC v3 and the corresponding new full-pool queue. This is queued work,
not a completed scheduler integration or algorithm comparison.

The mixed-length actor dependency has now passed: exact Docker exit0,24 real
responses/153 response tokens, maximum probability difference0.0001261234,
one optimizer update with gradient norm2.8016663. The separately executed
read-only audit also exited0 and verified112 finite AdamW states at step1 and
112 changed adapter tensors. The legacy snapshot's adapter remains root-owned
0600; hash/tensor validation used the container's read-only mount, without
changing its contents. See [receipt](../docs/engineering/vllm-mixed-actor-v1-result.json).
The instrumented scheduler now waits for the running full-pool diagnosis to
release local CPU resources; its numerical dependencies have completed.
