"""Offline task-level reward contract for complete sampled decision sets.

Not wired into the frozen dual-credit trainer. Caller authenticates TRAIN
labels, candidate order, behavior policy and sampled actions before use.
"""
import math

import torch

from .context_set_reward import set_f1


def validate(config):
    required = {'schema', 'semantics', 'exact_weight', 'overlap_weight', 'beta'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('explicit complete task reward configuration required')
    if (config['schema'] != 'jev-task-set-reward-v1'
            or config['semantics'] != 'exclusive_event'):
        raise ValueError('simple exclusive event reward contract required')
    for key in ('exact_weight', 'overlap_weight', 'beta'):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('finite numeric reward setting required: '+key)
    exact, overlap = config['exact_weight'], config['overlap_weight']
    if (not 0 < exact <= 1 or not 0 <= overlap < 1
            or not math.isclose(exact+overlap, 1., rel_tol=0., abs_tol=1e-12)
            or not .1 <= config['beta'] <= 10):
        raise ValueError('positive exact bonus, normalized weights and bounded beta required')
    return dict(config)


def score(actions, labels, observed, config):
    """Return detached B x G rewards and diagnostics, no model forward.

    actions are authentic full-set Bernoulli samples, not thresholded marginal
    probabilities repeated G times. Complete singleton benchmark truth only.
    """
    c = validate(config)
    f1 = set_f1(actions, labels, observed)  # original shape/device/truth checks
    if not (labels.sum(-1) == 1).all():
        raise ValueError('exactly one annotated event per simple task query required')
    selected = actions.bool()
    truth = labels[:, None, :]
    tp = (selected & truth).sum(-1)
    fp = (selected & ~truth).sum(-1)
    fn = (~selected & truth).sum(-1)
    exact = ((fp == 0) & (fn == 0)).float()
    b2 = c['beta']**2
    overlap = (1+b2)*tp.float()/((1+b2)*tp+b2*fn+fp).clamp_min(1e-12)
    reward = c['exact_weight']*exact + c['overlap_weight']*overlap
    return {key:value.detach() for key,value in dict(
        reward=reward, exact=exact, f1=f1, fbeta=overlap,
        true_positive=tp, false_positive=fp, false_negative=fn,
        selected_count=selected.sum(-1), partial=(tp > 0) & (exact == 0),
        empty=selected.sum(-1) == 0).items()}


def advantages(rewards, *, algorithm, std_floor):
    """Continuous task-group advantages, separate from binary node advantages.

    Does not implement policy ratios or a training loss. The integration must
    use the likelihood of these exact joint actions and bind old policy IDs.
    """
    if (algorithm not in ('task_grpo_floor_v1', 'task_rloo_v1')
            or type(std_floor) not in (int, float) or not math.isfinite(std_floor)
            or not 0 < std_floor <= 1):
        raise ValueError('explicit supported task algorithm and positive std floor required')
    if (not isinstance(rewards, torch.Tensor) or rewards.ndim != 2
            or rewards.shape[0] < 1 or rewards.shape[1] < 2
            or rewards.dtype != torch.float32 or rewards.requires_grad
            or not torch.isfinite(rewards).all()
            or not ((rewards >= 0) & (rewards <= 1)).all()):
        raise ValueError('detached bounded FP32 B x G task rewards required')
    if algorithm == 'task_rloo_v1':
        return (rewards - (rewards.sum(-1, keepdim=True)-rewards)/(rewards.shape[1]-1)).detach()
    centered = rewards-rewards.mean(-1, keepdim=True)
    return (centered/rewards.std(-1, unbiased=False, keepdim=True).clamp_min(std_floor)).detach()
