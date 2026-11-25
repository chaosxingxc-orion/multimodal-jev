"""Resolve card operation settings on the packaging host, before freezing a run."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "configs/experiments/card-execution-v1.json"


def merge(base, override):
    if not isinstance(override, dict) or override.keys() - base.keys():
        raise ValueError("unknown card execution setting")
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(base[key], value) if isinstance(base[key], dict) else deepcopy(value)
    return result


def settings(section, operation):
    with DEFAULT_PROFILE.open() as stream:
        return merge(json.load(stream)[section], operation.get(section, {}))


def positive_int(x):
    return type(x) is int and x > 0


def finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def reference_settings(operation):
    runtime = settings('reference_inference', operation)
    if not positive_int(runtime['torch_threads']) or type(runtime['seed']) is not int or not 0 <= runtime['seed'] < 2**64:
        raise ValueError('invalid inference seed or thread count')
    return runtime


def validate_stage1(s):
    opt = s['optimizer']
    if (type(s['actor_seed']) is not int or not 0 <= s['actor_seed'] < 2**64
            or not all(positive_int(s[k]) for k in ('actor_threads', 'collective_timeout_seconds'))
            or not finite(s['grad_clip']) or s['grad_clip'] <= 0
            or not isinstance(opt['betas'], list) or len(opt['betas']) != 2 or not all(finite(v) and 0 <= v < 1 for v in opt['betas'])
            or not finite(opt['eps']) or opt['eps'] <= 0
            or not finite(opt['weight_decay']) or opt['weight_decay'] < 0
            or not all(finite(v) and v > 0 for v in s['actor_clip'].values())):
        raise ValueError("invalid card Stage1 execution settings")
    return s


def bind_stage1(parent, operation):
    overrides = operation.get('stage1', {})
    if 'stage1_runtime' in parent:
        runtime = merge(parent['stage1_runtime'], overrides)
    else:
        runtime = settings('stage1', operation)
        if 'seed' in operation and 'actor_seed' not in overrides:
            runtime['actor_seed'] = operation['seed']
    validate_stage1(runtime)
    if 'stage1_runtime' in parent and parent['stage1_runtime'] != runtime:
        raise ValueError("Stage1 runtime differs from the bound parent")
    return dict(parent, stage1_runtime=runtime)


def prior_runtime(original, current, origin):
    """Only a registered digest may supply settings absent from a legacy cursor."""
    if 'stage1_runtime' in original:
        return original
    raw = json.dumps(current, sort_keys=True, allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest() != origin.get('legacy_runtime_sha256'):
        raise ValueError('legacy cursor lacks a matching frozen Stage1 runtime')
    return dict(original, stage1_runtime=current)


def adamw(params, runtime, lr):
    from torch.optim import AdamW
    opt = runtime['optimizer']
    return AdamW(params, lr=lr, **dict(opt, betas=tuple(opt['betas'])), foreach=False, fused=False)
