"""EDIT THIS FILE to experiment with Stage2 rewards and optimization.

The default reproduces the existing dual-credit objective. Change reward
values in compute_rewards(), and/or replace/add differentiable terms in
compute_loss(). No changes to the training loop are needed. Publish a new
file hash/config for each experiment; never edit a running package's copy.
"""
API_VERSION = 'jev-stage2-objective-v1'


def compute_rewards(ctx):
    """Return one {ranking, existence} detached M x G tensor map per query.

    ctx.records: authenticated behavior records, actions, labels, old logprobs.
    ctx.identities: query/input/policy/update identities.
    ctx.options: arbitrary JSON parameters from objective_plugin.options.
    Continuous rewards are allowed. For a whole-set reward use its matching
    joint-action loss in compute_loss(), rather than broadcasting to nodes.
    """
    return [{name: branch['rewards'].clone() for name, branch in record['branches'].items()}
            for record in ctx.records]


def compute_loss(ctx, rewards):
    """Return (differentiable scalar loss, metrics dict).

    ctx.logits: current differentiable B x C logits.
    ctx.reference_probability: fixed Stage1 B x C probabilities.
    ctx.real_requests: Boolean padding mask; padded rows MUST have zero credit.
    ctx.global_real_requests / ctx.world_size: original DP normalization.
    ctx.baseline_loss(rewards): dual GRPO/RLOO + existing auxiliary/KL terms,
      with the supplied continuous rewards and authenticated sampled actions.

    To add a term, compute it here and return base + coefficient * term.
    Normalize per-query terms by world_size/global_real_requests, using the
    real_requests mask. Leave optimizer/backward/LoRA export to the trainer.
    """
    return ctx.baseline_loss(rewards)
