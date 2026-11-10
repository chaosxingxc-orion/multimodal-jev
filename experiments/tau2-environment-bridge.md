# τ² environment bridge validation

Use upstream MIT-licensed `sierra-research/tau2-bench` commit
`033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f` (version2.0.0), already downloaded
under shared `repos/tau2`. Its current voice-v2 release differs from the original
τ-Voice paper. Shared domain data are under `datasets/tau2-voice/data`.

Before execution, fix this integration check to the **first retail TRAIN task
in upstream split order**, ID `0`. Preserve the upstream task, initial state,
tool schemas, policy and evaluator. Exercise its five reference actions only
as an oracle interface fixture, never as a model prediction or training label
fed through an agent observation. Gold criteria remain evaluator-side.

Run in a separate Python3.12 image, with the source pin copied into the image
and installed with its declared core dependencies. Record image ID, resolved
package versions and source/data hashes. Domain data and project source mounts
are read-only, network is disabled during validation, and no provider credentials
are passed. Do not modify the shared pinned source or existing training images.

Required checks:

- Execute all reference calls through upstream `Environment.get_response`,
  retain full tool messages and state hashes, and require every response to
  report successful execution. The final DB must differ from its initial state.
- Strictly replay the generated trajectory in a fresh environment and require
  identical DB hashes and upstream DB reward1.
- Omit the final exchange write and require unchanged DB and upstream DB
  reward0. Never count the number of syntactically valid calls as task success.
- Corrupt the recorded final write response and require strict replay to reject
  it. Preserve upstream execution/rejection semantics.
- Ensure a fresh environment resets state and agent-facing observations omit
  task evaluation criteria and private user instructions.

The reusable bridge will expose policy/tool schemas, sequential tool execution
and a replayable trajectory. Its score is explicitly the upstream **environment
component**: this task also names NL assertions, and environment scoring alone
does not validate those or conversational policy compliance. No LLM user, ASR,
TTS, speech policy, turn-taking, native verl worker, or model task completion is
validated by oracle replay. A later voice run must use the upstream orchestrator
and configured user/audio providers, preserve train/test splits and release pins,
and evaluate full task completion plus conversation/audio behavior. This fixture
must not be reported as a faithful academic baseline reproduction.

## Completed environment check

The [v2 receipt](../docs/engineering/tau2-environment-v2-result.json) records
successful exact Docker and application exits. All five upstream actions execute;
the final DB differs from its initial state and strict replay reproduces it.
The upstream environment component gives1 to the oracle trajectory and0 when
the final exchange write is omitted. Corrupting the write result is rejected.
Fresh construction resets the DB, duplicate call IDs fail before execution,
and a private-criteria sentinel leaves the agent observation unchanged.

This last check covers the bridge observation contract for this fixture. It is
not a security boundary for arbitrary in-process access to private Python fields
and does not prove every future orchestrator is leakage-free. The fixture has
no initial message history; histories and user-side tools need further coverage.

The standalone image is `multimodal-jev-tau2:033bf082-core-v2`; use its full
image ID in the receipt for execution. Python3.12 runs in an isolated venv; dependency versions
were resolved within upstream declared ranges, **not upstream uv.lock**.
Build evidence preserves every source/wheel hash and `requirements.freeze`.
The Dockerfile installs solely from those downloaded wheels with network disabled.

Failure evidence is retained: the first online build could not connect directly
to PyPI and was explicitly stopped after a separate connectivity check. Host-side
download using the existing proxy succeeded; no proxy credentials were placed
in the image. The first probe in the successful image failed before task execution
because a selective project source snapshot omitted imported modules. V2 uses
a complete package snapshot; no tool behavior, evaluator or threshold was changed.

Next integration gates are actual agent observations and tool responses across
conversation turns, upstream user simulation/audio providers, policy-compliance
scoring, and model-driven voice evaluation. The present bridge only executes
assistant tool calls sequentially and exposes the environment component score.
