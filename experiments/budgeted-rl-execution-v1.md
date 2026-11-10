# Real n4 RL budget-stop gate

Use the first two complete six-view TRAIN groups from the seed17
pair-GRPO-standard comparison configuration. Use the frozen SFT initializer,
original LoRA base, float32 full projection, LR1e-5, rank4/alpha8, raw-policy
n4/max_tokens32 sampling and actor_features. No outcome-based selection.

Pin the successful train-pool-budget-profile-v2 result. Set the effective
training limit to exactly the first group's reservation: four responses per
prompt, maximum 32 response tokens, one gradient and one parity pass. Set the
generated-token limit to exactly the first group's response reservation, 768.
The second group is admitted only if both conservative reservations fit the
actual remaining budget. Expect a stop after one group when the first actor
updates; if zero advantage leaves sufficient budget, retain and report any
second admitted group rather than inventing an update or truncating receipts.
At most the two frozen groups are attempted. Every admitted iteration must
finish rollout, parity-gated actor and fresh export before the next admission.

Queue behind the successful exact-container checks, actor-state audit and
budget audit of container-two-step-gated-launch-v2, and the successful real
budgeted SFT gate. These dependencies serialize CPU vLLM allocation and avoid
changing their immutable snapshots. Validate the prepared source, config and
runtime hashes before a one-shot launch. Observation timeouts do not restart
containers or supervisors; failed dependencies retain terminal evidence.

After the schedule completes, inspect each exact child Docker ID and its
application exit. Audit every saved actor with the read-only actor auditor,
then independently reconstruct generation/training costs and the cap ledger
from stage receipts. Require actual n4 group sizes, profile-aligned prompt and
audio counts, finite saved optimizer states, correct parent/export lineage,
and both caps respected. If the first group updates, require one completed
iteration and budget_exhausted before the second group. No-op outcomes are
retained explicitly and cannot prove an optimizer update.

This is a real integration gate, not one of the formal 45 efficacy trajectories.
It does not establish distributed verl workers, NPU execution, normalization
superiority or cross-task gains. The primary comparison remains unlaunched
until resource allocation and all required execution gates are verified.
