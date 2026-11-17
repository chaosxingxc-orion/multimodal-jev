# Response-position vocabulary projection probe

The full-batch actor OOM motivates a memory optimization without changing batch
size or the objective. The pinned Qwen3-ASR forward natively accepts
logits_to_keep. Retain response width + 1 trailing positions: the preceding
prompt position predicts the first response token; the final position is unused.
The transformer still sees the entire prompt/audio/response sequence. Only
unneeded vocabulary projection positions are omitted. The default remains off.

Before enabling this in real training, compare full and shortened projection
on the same complete three-response TRAIN fixture, unequal response lengths,
same initial adapter and fresh identical AdamW states. Use the existing strict
parameter/gradient/moment thresholds (atol 2e-6, rtol 2e-4), pre-update log-prob
gate 3e-4 and loss difference <2e-5. Persist every comparison error before
asserting. Also require corrupted saved probabilities to reject the update with
unchanged parameters and zero optimizer states. Do not loosen any threshold.

Use tools/jev/actor_response_logits_smoke.py in the NGA-aligned CPU image, four
cores and 16 GiB. This controlled-fixture test is independent of live v5, whose
immutable source, full projection and 128 GiB resource-only retry remain intact.
Success here permits a separately identified actual-vLLM actor test, not an
efficacy claim. Activation checkpointing and microbatch changes are not included.

The independent probe is running as npu CPU container
2be0767333a2aecd49b67d219e862da741e67d1c37dbaa890b649793aee9e833,
root actor-response-logits-smoke-v1. See
[launch receipt](../../../../docs/engineering/response-logits-v1-launch.json).
No current training configuration enables response_logits_only.

The controlled probe has now exited 0: full and shortened projection have
exactly equal losses, gradients, AdamW first moments and updated parameters
(maximum error 0 for each recorded tensor class). The corrupt-logprob test
rejected before any optimizer state or parameter changed. See
[result receipt](../../../../docs/engineering/response-logits-v1-result.json).

A separate real-vLLM actor now explicitly enables the option on the successful
v4 rollouts, with the same original base/SFT adapter, full batch12, pair reward,
learning rate and 0.02 gate. Its memory limit is32GiB and container is
5b5c40e5f2b0f5583f944ed4fd1142e3ffbded3b762ea52e0f7cd42bb5064ee2.
See [launch receipt](../../../../docs/engineering/vllm-response-only-actor-v1-launch.json).
The 128GiB full-projection v5 remains unchanged and live. Fixture equivalence
does not prove every real trajectory or memory saving; inspect actual artifacts.

The independent response-only chain supervisor now waits for that exact actor,
then requires a real update, verified export, fresh vLLM process and persisted
optimizer restoration in the second actor. It uses its own experiment/export
roots; the existing full-projection chain remains separate. See
[launch receipt](../../../../docs/engineering/response-only-chain-v1-launch.json).

The real response-only v1 actor exited137 with Docker OOMKilled=true during
gradient forward at its 32GiB limit. Its complete parity check passed at
4.8220157623291016e-5; no saved optimizer or result exists. Both downstream
supervisors stopped on that exact failed container. See
[failure receipt](../../../../docs/engineering/response-only-actor-v1-oom.json).
Response-only vocabulary projection alone therefore does not establish a
32GiB training path. A separately named v2 retry raises only the container
limit to 128GiB, preserving byte-identical source/config and fixed gates;
see [retry launch](../../../../docs/engineering/response-only-actor-v2-launch.json).

The [v2 chain](../../../../docs/engineering/response-only-chain-v2-launch.json) observes
the exact retry container, requires a successful update, and retains the same
export parity and fresh second-engine/optimizer restoration gates. The failed
v1 chain and mixed-response queue are retained without restart.

The128GiB v2 actor exited0 with one actual update, gradient norm3.0577147,
probability error4.82202e-5,112 changed adapter tensors and112 finite AdamW
states at step1. The token ledger records8,050 prompt tokens,80 response tokens
and2,220 valid audio frames. See
[result](../../../../docs/engineering/response-only-actor-v2-result.json).

Independent saved-parameter comparison to the full-projection v5 actor found
maximum error2.1555279090534896e-6 and failed the unchanged atol2e-6/rtol2e-4
gate. This real-batch finding supersedes any inference of general equivalence
from the smaller controlled fixture. No full gradient comparison was retained
for these live runs. Keep full projection for registered formal configs;
do not relax tolerances or label the optimization validated for those studies.

Both v2 downstream supervisors stopped before launching export/mixed actor
because their old snapshots saved root-only adapter weights. Hashes were checked
before/after making weights readable, and saved optimizer contents verified.
The failures are retained; no downstream retry was launched while remote /var
was full. The separate full-projection second actor and mixed actor keep running.
