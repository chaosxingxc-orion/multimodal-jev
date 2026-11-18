"""Candidate-normalized CE plus native verl finite-support PPO.

Both losses use a weighted request mean, not a mean over padded candidates.
The same global denominator must be provided for every microbatch/rank. Under
DDP/FSDP mean gradient reduction multiply each local numerator by world_size.
Caller owns collective denominator authentication and lifecycle integration.
"""
import math
import torch

from .span_finite_credit import validate_policy, verl_loss


def sft_coefficient(update, total_updates, start=1., end=.1, decay_fraction=.8):
    if (type(update) is not int or type(total_updates) is not int or total_updates < 1
            or not 0 <= update < total_updates
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end, decay_fraction))
            or not 0 <= end <= start or not 0 < decay_fraction <= 1):
        raise ValueError('fixed bounded stage schedule required')
    end_index = max(1, math.ceil((total_updates-1)*decay_fraction))
    fraction = min(update/end_index, 1.)
    return end + (start-end)*.5*(1+math.cos(math.pi*fraction))


def supervised_numerator(policy, targets, supervised, span_weights, request_weights):
    if (targets.ndim != 2 or targets.dtype != torch.long or not len(targets)
            or targets.shape[1] != len(policy) or supervised.shape != targets.shape
            or supervised.dtype != torch.bool or span_weights.shape != targets.shape
            or request_weights.shape != (len(targets),)):
        raise ValueError('aligned labels, masks and request weights required')
    validate_policy(policy, len(targets))
    device = policy[0].device
    if any(x.device != device for x in (targets, supervised, span_weights, request_weights)):
        raise ValueError('supervision and policy devices differ')
    for w in (span_weights, request_weights):
        if w.requires_grad or not w.is_floating_point() or not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError('detached finite nonnegative weights required')
    weights = span_weights*supervised
    active_weight = request_weights*(weights.sum(-1) > 0)
    per_request = policy[0].sum(-1)*0
    for j, p in enumerate(policy):
        known = supervised[:, j]
        y = targets[:, j]
        if ((y[known] < 0) | (y[known] >= p.shape[1])).any():
            raise ValueError('supervised class outside candidate support')
        safe = y.masked_fill(~known, 0)
        per_request = per_request - p.gather(1, safe[:, None])[:, 0]*weights[:, j]
    return (per_request*active_weight).sum(), active_weight.sum()


def loss(policy, targets, supervised, span_weights, request_weights, *, stage,
         global_effective_weight, world_size=1, behavior=None, coefficient=0., clip_ratio=.2,
         real_request_mask=None, global_real_requests=None):
    """Return local loss ready for backward and detached local contribution metrics.

    gradient accumulation must SUM these global-normalized microbatch losses;
    an additional division by accumulation steps would shrink the gradient.
    No old probabilities are needed for instruction CE-only stage.
    """
    if (stage not in ('instruction', 'task') or type(world_size) is not int or world_size < 1
            or type(global_effective_weight) not in (int, float)
            or not math.isfinite(global_effective_weight) or global_effective_weight <= 0
            or type(coefficient) not in (int, float) or not math.isfinite(coefficient) or coefficient < 0):
        raise ValueError('explicit stage and global effective request normalization required')
    if (real_request_mask is None) != (global_real_requests is None):
        raise ValueError('real request mask and global count must be supplied together')
    if real_request_mask is not None:
        if (stage != 'instruction' or type(global_real_requests) is not int or global_real_requests < 1
                or real_request_mask.dtype != torch.bool or real_request_mask.shape != request_weights.shape
                or real_request_mask.device != request_weights.device
                or int(real_request_mask.sum()) > global_real_requests
                or (request_weights[~real_request_mask] != 0).any()):
            raise ValueError('explicit instruction real-request population and zero-weight padding required')
    ce_numerator, local_weight = supervised_numerator(policy, targets, supervised, span_weights, request_weights)
    if local_weight.item() > global_effective_weight+1e-6:
        raise ValueError('local effective weight exceeds global denominator')
    scale = world_size/(global_real_requests if global_real_requests is not None else global_effective_weight)
    ce = ce_numerator*scale
    rl = ce*0
    if stage == 'instruction':
        if behavior is not None or coefficient != 0:
            raise ValueError('instruction stage uses CE only, no dummy behavior or decay coefficient')
        total = ce
    else:
        if behavior is None:
            raise ValueError('task stage requires authenticated detached behavior probabilities')
        validate_policy(behavior, len(targets), detached=True)
        if len(behavior) != len(policy) or any(a.shape != b.shape or a.device != b.device for a, b in zip(policy, behavior, strict=True)):
            raise ValueError('behavior support differs')
        if local_weight.item() > 0:
            # Native verl returns its policy loss and diagnostic dictionary.
            raw, _ = verl_loss(behavior, policy, targets, supervised, span_weights,
                              request_weights, clip_ratio)
            rl = raw*local_weight*scale
        total = rl + coefficient*ce
    return total, dict(ce=float(ce.detach()), rl=float(rl.detach()), total=float(total.detach()),
                      coefficient=coefficient if stage == 'task' else 1.,
                      local_effective_weight=float(local_weight), global_effective_weight=global_effective_weight,
                      world_size=world_size, global_real_requests=global_real_requests)
