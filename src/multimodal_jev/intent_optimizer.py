"""Explicit AdamW contract for separately registered intent RL controls."""
import math


def optimizer_kwargs(config):
    lr = config['lr']; betas = config.get('adam_betas', [0.9, 0.999])
    if (type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0
            or not isinstance(betas, (list, tuple)) or len(betas) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1 for v in betas)):
        raise ValueError('positive learning rate and two finite Adam decay factors required')
    return dict(lr=lr, betas=tuple(betas), eps=1e-8, weight_decay=0., foreach=False, fused=False)


def validate_saved_optimizer(config, state):
    groups = state['param_groups']
    if len(groups) != 1: raise ValueError('one Adam parameter group required')
    group = groups[0]
    for key, value in optimizer_kwargs(config).items():
        actual = group.get(key)
        if key == 'betas': actual = tuple(actual) if isinstance(actual, (list, tuple)) else actual
        if actual != value: raise ValueError('saved optimizer differs from configuration: '+key)
    if group.get('amsgrad') or group.get('maximize'):
        raise ValueError('ordinary AdamW descent required')
