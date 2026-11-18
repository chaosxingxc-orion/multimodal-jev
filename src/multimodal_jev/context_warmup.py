"""Cursor-derived LR warmup with exact checkpoint/restart semantics."""
import math


def learning_rate(spec, update, total_updates):
    """Return LR for zero-based optimizer update, including after a restart."""
    if (set(spec) != {'kind', 'peak_lr', 'warmup_updates'}
            or spec['kind'] != 'linear_warmup_constant'
            or type(total_updates) is not int or total_updates < 1
            or type(update) is not int or not 0 <= update < total_updates
            or type(spec['warmup_updates']) is not int
            or not 1 <= spec['warmup_updates'] <= total_updates
            or type(spec['peak_lr']) not in (float, int)
            or not math.isfinite(spec['peak_lr']) or spec['peak_lr'] <= 0):
        raise ValueError('bounded linear warmup and exact update cursor required')
    return spec['peak_lr'] * min((update + 1) / spec['warmup_updates'], 1.)


def set_learning_rate(optimizer, spec, update, total_updates):
    value = learning_rate(spec, update, total_updates)
    for group in optimizer.param_groups:
        group['lr'] = value
    return value
