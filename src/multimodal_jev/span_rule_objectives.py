"""Matched supervised and sampled objectives for reviewed Boolean rule tasks.

Final-only refers to the objective, independently of the input question arm.
Auxiliary scores measure reviewed fact correctness, not textual reasoning.
"""
import math
import torch

from .span_rewards import decision_rewards


def validate_rule_targets(targets, supervised, width, auxiliary_weight):
    if (targets.ndim != 2 or targets.shape[1] != width or width not in (1, 3)
            or targets.dtype != torch.long or supervised.dtype != torch.bool
            or supervised.shape != targets.shape or targets.shape[0] == 0
            or type(auxiliary_weight) not in (int, float)
            or not math.isfinite(auxiliary_weight) or not 0 <= auxiliary_weight < 1):
        raise ValueError('aligned Boolean rule targets and bounded auxiliary weight required')
    if not supervised[:, -1].all() or ((targets[supervised] < 0) | (targets[supervised] > 1)).any():
        raise ValueError('reviewed Boolean Final required; missing facts must be masked')
    if width == 1 and auxiliary_weight != 0:
        raise ValueError('one-span input has no auxiliary facts')


def rule_sft_loss(candidate_logprobs, targets, supervised, auxiliary_weight=0.0):
    """Mean Final CE plus weighted per-request mean of reviewed fact CEs.

    Use candidate-normalized FP32 probabilities, matching the finite RL action
    space. Missing facts never become false, and question count adds no weight.
    """
    validate_rule_targets(targets, supervised, len(candidate_logprobs), auxiliary_weight)
    for logp in candidate_logprobs:
        if (logp.shape != (len(targets), 2) or not torch.isfinite(logp).all()
                or not torch.allclose(logp.logsumexp(-1), logp.new_zeros(len(targets)), atol=1e-6, rtol=0)):
            raise ValueError('normalized Boolean candidate probabilities required')
    safe = targets.masked_fill(~supervised, 0)
    nll = torch.stack([-p.gather(1, safe[:, i:i+1]).squeeze(1)
                       for i, p in enumerate(candidate_logprobs)], -1)
    facts = supervised.clone()
    facts[:, -1] = False
    auxiliary = (nll * facts).sum(-1) / facts.sum(-1).clamp_min(1)
    return (nll[:, -1] + auxiliary_weight * auxiliary).mean()


def rule_sample_rewards(actions, targets, supervised, auxiliary_weight=0.0):
    """Same Final/auxiliary weighting as SFT; returns complete-action rewards."""
    validate_rule_targets(targets, supervised, actions.shape[-1], auxiliary_weight)
    if ((actions < 0) | (actions > 1)).any():
        raise ValueError('Boolean action outside support')
    return decision_rewards(actions, targets, supervised, actions.shape[-1]-1, auxiliary_weight)


def expected_rule_reward(candidate_logprobs, targets, supervised, auxiliary_weight=0.0):
    """Exact finite-policy expectation for diagnostics, never a sampled reward.

    Gold is used only by this scorer. No fact prediction is fed into Final.
    Comparing this value before/after updates avoids sampling noise obscuring
    a local reward change; it is still an in-sample development measurement.
    """
    # Validate support/normalization through the shared supervised scorer.
    rule_sft_loss(candidate_logprobs, targets, supervised, auxiliary_weight)
    safe = targets.masked_fill(~supervised, 0)
    correct = torch.stack([p.exp().gather(1, safe[:, i:i+1]).squeeze(1)
                           for i, p in enumerate(candidate_logprobs)], -1)
    facts = supervised.clone()
    facts[:, -1] = False
    return correct[:, -1] + auxiliary_weight * (correct * facts).sum(-1) / facts.sum(-1).clamp_min(1)
