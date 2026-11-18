"""Explicit binary-policy parity before a context actor optimizer update.

Bounds belong to a validated runtime configuration, never inherited from the
old categorical Span protocol. This check uses the existing training forward.
"""
import math

import torch


def validate_bounds(bounds):
    keys={'probability_max_abs','binary_logprob_max_abs'}
    if (set(bounds)!=keys or any(type(bounds[k]) not in (int,float) or
            not math.isfinite(bounds[k]) or bounds[k]<0 for k in keys)):
        raise ValueError('explicit finite nonnegative context parity bounds required')


def compare(current, behavior, bounds):
    validate_bounds(bounds)
    if (current.ndim!=2 or current.shape!=behavior.shape or current.shape[0]<1 or
            current.shape[1]<2 or current.shape[1]%2 or
            not current.is_floating_point() or not behavior.is_floating_point() or
            not torch.isfinite(current).all() or not torch.isfinite(behavior).all()):
        raise ValueError('complete finite paired binary log-probabilities required')
    # FP64 comparison is independent of BF16 execution/FP32 loss rounding.
    # Ascend does not provide general FP64 arithmetic. Only this small detached
    # decision vector is compared on CPU; forward/backward stay on accelerator.
    a=current.detach().to(device='cpu',dtype=torch.float64).reshape(current.shape[0],-1,2)
    b=behavior.detach().to(device='cpu',dtype=torch.float64).reshape_as(a)
    for value in (a,b):
        if (value>0).any() or not torch.allclose(value.exp().sum(-1),
                torch.ones_like(value[...,0]),atol=2e-6,rtol=0):
            raise ValueError('normalized binary distributions required')
    delta=(a-b).abs()
    probability_error=float((a.exp()-b.exp()).abs().max())
    logprob_error=float(delta.max())
    # Bounds the log ratio for any complete sampled decision tuple.
    joint_bound=float(delta.max(-1).values.sum(-1).max())
    flips=int(((a[...,1].exp()>=.5)!=(b[...,1].exp()>=.5)).sum())
    return dict(passed=probability_error<=bounds['probability_max_abs'] and
        logprob_error<=bounds['binary_logprob_max_abs'] and flips==0,
        probability_max_abs=probability_error,binary_logprob_max_abs=logprob_error,
        joint_logratio_abs_bound=joint_bound,threshold_flips=flips)
