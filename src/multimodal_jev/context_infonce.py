"""Supervised candidate InfoNCE for the simple exclusive-event task.

The model still emits independent sigmoid probabilities. The softmax here is
only the contrastive training objective; it is not the inference readout.
"""
import math

import torch


def loss(logits, labels, observed, request_weights, real_requests, *,
         global_real_requests, world_size=1, temperature=1.):
    if (logits.ndim != 2 or min(logits.shape) < 1 or not logits.is_floating_point()
            or not torch.isfinite(logits).all()
            or labels.dtype != torch.bool or observed.dtype != torch.bool
            or labels.shape != logits.shape or observed.shape != logits.shape
            or labels.device != logits.device or observed.device != logits.device):
        raise ValueError('finite candidate scores and matching Boolean supervision required')
    b = logits.shape[0]
    if (request_weights.shape != (b,) or not request_weights.is_floating_point()
            or request_weights.requires_grad or not torch.isfinite(request_weights).all()
            or real_requests.shape != (b,) or real_requests.dtype != torch.bool
            or request_weights.device != logits.device or real_requests.device != logits.device
            or not torch.equal(request_weights, real_requests.to(request_weights.dtype))):
        raise ValueError('uniform real queries and zero-weight padding required')
    if (type(global_real_requests) is not int or global_real_requests < 1
            or int(real_requests.sum()) > global_real_requests
            or type(world_size) is not int or world_size < 1
            or type(temperature) not in (float, int)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('global real denominator, DP width and positive temperature required')
    # Only real rows require supervision. Padding has no effect even on a rank
    # whose entire tail consists of padding; unknown labels never become false.
    active = real_requests
    if ((~observed[active]).any() or logits.shape[1] < 2
            or (labels[active].sum(-1) != 1).any()):
        raise ValueError('complete exclusive-event panel with one positive and negatives required')
    z = logits.float() if logits.dtype in (torch.bfloat16, torch.float16) else logits
    if active.any():
        scores = z[active] / temperature
        positive = scores[labels[active]]
        numerator = (torch.logsumexp(scores, dim=-1) - positive).sum()
    else:
        numerator = z.sum() * 0
    value = numerator * world_size / global_real_requests
    return value, dict(infonce_numerator=float(numerator.detach()),
        local_real_queries=int(active.sum()), global_real_queries=global_real_requests,
        temperature=temperature, calibration_claim=False)
