"""Pinned verl outcome estimators for complete finite-action request groups."""
import inspect
import numpy as np
import torch
from .baseline import file_hash

CORE_SHA256 = 'fadfafa83d94ba40d7ac2abeedb6e99ea0716b42f8744c421ecbb371db8b4d24'


def group_advantages(rewards, request_ids, *, algorithm='grpo_v1'):
    """Compute before DP/microbatch splitting; zero-signal groups are retained.

    Rewards are B x G, with equal G >= 2. No per-class weight, padding,
    process normalization or extra whitening enters the estimator.
    """
    if algorithm not in ('grpo_v1', 'rloo_v1'):
        raise ValueError('registered GRPO or RLOO estimator required')
    if (rewards.ndim != 2 or rewards.shape[0] < 1 or rewards.shape[1] < 2
            or not rewards.is_floating_point() or rewards.requires_grad
            or not torch.isfinite(rewards).all() or len(request_ids) != rewards.shape[0]
            or any(type(uid) is not str or not uid for uid in request_ids)
            or len(set(request_ids)) != len(request_ids)):
        raise ValueError('complete distinct detached finite reward groups required')
    from verl.trainer.ppo import core_algos
    if file_hash(inspect.getfile(core_algos)) != CORE_SHA256:
        raise ValueError('pinned verl advantage implementation changed')
    size = rewards.shape[1]
    flat = rewards.reshape(-1, 1).clone()
    mask = torch.ones_like(flat)
    index = np.array([uid for uid in request_ids for _ in range(size)], dtype=object)
    if algorithm == 'grpo_v1':
        advantages, _ = core_algos.compute_grpo_outcome_advantage(
            flat, mask, index, norm_adv_by_std_in_grpo=True)
    else:
        advantages, _ = core_algos.compute_rloo_outcome_advantage(flat, mask, index)
    if not torch.isfinite(advantages).all() or advantages.requires_grad:
        raise ValueError('invalid detached group advantages')
    return advantages.reshape_as(rewards)
