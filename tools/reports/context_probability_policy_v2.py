"""Versioned Bernoulli arithmetic matching the actual FP32 rollout sampler.

Only the small decision tensor crosses CPU. Autograd retains both transfers;
no detached correction, clipping, straight-through gradient or normalization.
The original actor module and all frozen packages remain unchanged.
"""
import torch

PROFILE = 'context-bernoulli-probability-cpu-log-v2'


def binary_logprobs(logits):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or
            min(logits.shape) < 1 or not logits.is_floating_point() or
            not torch.isfinite(logits).all()):
        raise ValueError('finite nonempty B x decisions logits required')
    values = logits.float() if logits.dtype in (torch.bfloat16, torch.float16) else logits
    if values.dtype != torch.float32:
        raise ValueError('actual FP32 sampler arithmetic required')
    probability = values.sigmoid().to('cpu')
    if not torch.isfinite(probability).all() or ((probability <= 0) | (probability >= 1)).any():
        raise ValueError('finite interior actual probabilities required')
    logs = torch.stack((torch.log1p(-probability), torch.log(probability)), -1)
    return list(logs.to(logits.device).unbind(1))
