"""Offline one-step conditional-credit surrogate; not wired into an actor.

Reward pairs must score both values of each bit with all other sampled bits
fixed. The caller owns truth coverage and sample provenance. This module does
not grant runtime admission or validate reward semantics.
"""
import torch


def conditional_credit_loss(probabilities, behavior, reward_pairs):
    """Mean over B/G, SUM over nodes; pairs have shape [B,G,N,2].

At theta=behavior, minus this loss gradient is an unbiased estimate of the
raw expected reward gradient if other bits were sampled independently from
behavior. No standardization, importance ratio, clip, or multi-epoch reuse.
The scalar loss is a coordinate surrogate, not negative expected reward.
"""
    if (probabilities.ndim != 2 or min(probabilities.shape) < 1
            or not probabilities.is_floating_point()
            or not torch.isfinite(probabilities).all()
            or ((probabilities <= 0) | (probabilities >= 1)).any()):
        raise ValueError('finite interior B x N probabilities required')
    if (behavior.shape != probabilities.shape or behavior.requires_grad
            or behavior.dtype != probabilities.dtype or behavior.device != probabilities.device
            or not torch.equal(behavior, probabilities.detach())):
        raise ValueError('one update at exactly the detached behavior probabilities required')
    if (reward_pairs.ndim != 4 or reward_pairs.shape[0] != probabilities.shape[0]
            or reward_pairs.shape[1] < 1
            or reward_pairs.shape[2:] != (probabilities.shape[1], 2)
            or reward_pairs.requires_grad or not reward_pairs.is_floating_point()
            or reward_pairs.device != probabilities.device
            or reward_pairs.dtype != probabilities.dtype
            or not torch.isfinite(reward_pairs).all()):
        raise ValueError('detached finite B x G x N x 2 reward pairs required')
    delta = (reward_pairs[..., 1] - reward_pairs[..., 0]).mean(dim=1)
    return -(probabilities * delta).sum(dim=-1).mean()
