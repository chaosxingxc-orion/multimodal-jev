"""Differentiable exact singleton set-F1 candidate; not connected to any actor.

Consumes authoritative probabilities, preserving the caller's sigmoid path.
No sampled ratio, group normalization, clipping or calibration claim.
"""
import torch
from torch.nn import functional as F


def expected_reward(probabilities, labels, observed):
    if (probabilities.ndim != 2 or min(probabilities.shape)<1
            or probabilities.dtype not in (torch.float32,torch.float64)
            or not torch.isfinite(probabilities).all()
            or ((probabilities<0)|(probabilities>1)).any()):
        raise ValueError('Finite FP32/FP64 B x C probabilities required')
    if (labels.shape!=probabilities.shape or labels.dtype!=torch.bool
            or observed.shape!=labels.shape or observed.dtype!=torch.bool
            or labels.device!=probabilities.device or observed.device!=probabilities.device
            or not observed.all() or not (labels.sum(-1)==1).all()):
        raise ValueError('Complete observed singleton truth required; missing labels cannot be negatives')
    batch,candidates=probabilities.shape
    gold=probabilities[labels].reshape(batch)
    negative=probabilities[~labels].reshape(batch,candidates-1)
    counts=probabilities.new_ones((batch,1))
    for column in range(candidates-1):
        p=negative[:,column:column+1]
        counts=F.pad(counts*(1-p),(0,1))+F.pad(counts*p,(1,0))
    coefficients=2/(torch.arange(candidates,device=probabilities.device,dtype=probabilities.dtype)+2)
    return gold*(counts*coefficients).sum(-1)


def loss(probabilities,labels,observed,request_weights,real_requests,*,global_real_requests,world_size=1):
    rewards=expected_reward(probabilities,labels,observed)
    batch=probabilities.shape[0]
    if (request_weights.shape!=(batch,) or not request_weights.is_floating_point()
            or request_weights.requires_grad or not torch.isfinite(request_weights).all()
            or (request_weights<0).any() or real_requests.shape!=(batch,)
            or real_requests.dtype!=torch.bool
            or request_weights.device!=probabilities.device or real_requests.device!=probabilities.device
            or (request_weights[~real_requests]!=0).any()):
        raise ValueError('Detached nonnegative weights and zero-weight padding required')
    if (type(global_real_requests) is not int or global_real_requests<1
            or int(real_requests.sum())>global_real_requests
            or type(world_size) is not int or world_size<1):
        raise ValueError('Global real-query denominator and integer world size required')
    return -(rewards*request_weights).sum()*world_size/global_real_requests
