"""Joint Bernoulli RL plus Final-only auxiliary for native attribute arms.

Independent candidate only: no actor registration or runtime mutation. Uses
the admitted probability arithmetic and pinned verl estimators/surrogate.
The caller owns input/policy/catalog identities and global DP admission.
"""
import torch
from multimodal_jev.span_finite_credit import validate_policy
from multimodal_jev.span_group_advantage import group_advantages
from multimodal_jev.span_policy import joint_logprobs
from multimodal_jev.span_sampled_hybrid import loss as sampled_loss
from tools.reports.context_probability_policy_v2 import binary_logprobs, PROFILE
from tools.reports.context_attribute_reward_candidate import (
    rewards, validate_truth, final_auxiliary)


def prepare_groups(behavior, actions, labels, observed, menu, *,
                   arm, request_ids, algorithm):
    """Build all G16 rewards and advantages before DP/microbatch partition."""
    utility, families = rewards(actions, labels, observed, menu, arm=arm)
    if actions.shape[1] != 16:
        raise ValueError('Frozen G16 action groups required')
    validate_policy(behavior, labels.shape[0], detached=True)
    if (len(behavior) != labels.shape[1]
            or any(p.shape != (labels.shape[0], 2) or p.dtype != torch.float32
                   or p.device != labels.device for p in behavior)):
        raise ValueError('Actual FP32 binary behavior on complete arm support required')
    advantage = group_advantages(utility, request_ids, algorithm=algorithm)
    return dict(rewards=utility, family_rewards=families, advantages=advantage,
                old_joint=joint_logprobs(behavior, actions),
                arithmetic=PROFILE, arm=arm, algorithm=algorithm)


def loss(logits, labels, observed, request_weights, real_requests, menu, *,
         arm, global_real_requests, world_size=1,
         ranking_coefficient, probability_coefficient,
         actions, old_joint, advantages, reference_policy):
    """One global query denominator; joint policy/KL use all active nodes."""
    validate_truth(labels, observed, menu, arm)
    if (actions.ndim != 3 or actions.shape != (labels.shape[0], 16, labels.shape[1])
            or reference_policy is None):
        raise ValueError('Frozen G16 and fixed Stage1 reference required')
    auxiliary, aux_metrics = final_auxiliary(logits, labels, observed,
        request_weights, real_requests, menu, arm=arm,
        global_real_requests=global_real_requests, world_size=world_size,
        ranking_coefficient=ranking_coefficient, probability_coefficient=probability_coefficient)
    policy = binary_logprobs(logits)
    # The legacy categorical supervision is explicitly disabled. Our sole
    # supervision is Final ranking/NLL; process nodes affect RL and KL only.
    rl_kl, sampled_metrics = sampled_loss(policy, actions, old_joint, advantages,
        labels.long(), torch.zeros_like(observed), torch.zeros_like(logits),
        request_weights, real_requests, global_real_requests=global_real_requests,
        world_size=world_size, coefficient=0., reference_policy=reference_policy,
        kl_coefficient=.01, clip_ratio=.2)
    return rl_kl + auxiliary, dict(arm=arm, arithmetic=PROFILE,
        action_nodes=labels.shape[1], auxiliary=aux_metrics, sampled=sampled_metrics,
        scope='Objective only; caller authenticates rollout, catalog, reference and DP population')
