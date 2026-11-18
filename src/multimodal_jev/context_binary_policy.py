"""Adapter from context-head logits to the existing finite-action RL interface.

Applicability marginals remain sigmoid probabilities. For exploration ONLY we
explicitly choose a factorized Bernoulli action sampler over decision bits.
This sampling policy is not a claim that the underlying semantic events are
independent, and its joint mass must never be reported as calibrated set truth.
"""
import torch
from torch.nn import functional as F


def binary_logprobs(logits):
    """Return one [B, 2] normalized distribution per span, ordered false/true.

No padded or absent spans: group requests of equal actual decision count before
calling the existing finite-span actor. Label missingness affects reward/loss,
not whether an existing decision is part of the sampled action.
"""
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or
            min(logits.shape) < 1 or not logits.is_floating_point() or
            not torch.isfinite(logits).all()):
        raise ValueError('finite nonempty B x decisions logits required')
    values = logits.float() if logits.dtype in (torch.bfloat16, torch.float16) else logits
    logps = torch.stack((F.logsigmoid(-values), F.logsigmoid(values)), -1)
    return list(logps.unbind(1))
