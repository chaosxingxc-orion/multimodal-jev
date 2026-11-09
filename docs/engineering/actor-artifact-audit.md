# Saved actor artifact validation

`tools/jev/audit_actor_artifacts.py --run ACTOR_RUN --output NEW_REPORT.json`
validates a completed application's adapter and AdamW artifacts without loading
the base model or running a forward/backward pass. Execute with torch and
safetensors available; the NGA-aligned pinned image provides both. Inputs can
remain mounted read-only, and the output path must not already exist.

Checks include saved file hashes, exact parameter name/shape coverage, finite
adapter tensors and first/second moments, nonnegative second moments, persisted
CPU RNG format, optimizer settings and every parameter's actual AdamW step.
The result's update count must agree with those steps. For a parent continuation,
the immediate parent's artifacts and counters are checked, parameter coordinates
must agree, and the update/iteration counters must advance consistently.
No-op iterations must preserve adapter weights and optimizer moments. Input
adapter hashes are compared with the original actor identity.

This does not prove gradient equivalence, successful Docker termination,
rollout/export provenance or model quality. Those remain separate gates. The
parent check covers one immediate link, so a complete trajectory must audit
each actor. A changed optimizer counter alone never establishes learning gains.

The [real validation receipt](actor-artifact-audit-v1-result.json) checks the full
projection v5 and response-only v2 first actors in a new CPU container with
read-only training assets. Exact Docker exit0 and application success were
independently observed. Both contain112 finite AdamW states at step1 and112
changed adapter tensors. The snapshot hash is recorded; two subsequent strict
checks for no-op parameter groups and finite initializer tensors are covered by
unit tests but were not part of that container snapshot. The existing failed
response-only parameter-equivalence gate remains failed.

The lightweight project environment passes151 tests and skips this torch-only
module. A separate installed CPU torch2.10 environment passes all14 tests in
the module, including real serialization of a parent/no-op pair and rejection
of corrupted no-op moments. The tensor fixtures are unit checks, not training.
