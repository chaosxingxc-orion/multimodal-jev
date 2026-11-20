"""Per-objective numerical settings; scoped so concurrent actors/replays cannot leak settings."""
from contextlib import contextmanager
from contextvars import ContextVar
import math

_active = ContextVar('jev_grpo_numerics', default=None)


def validate(settings):
    def finite(x):
        return type(x) in (int, float) and math.isfinite(x)
    if not isinstance(settings, dict) or set(settings) != {'advantage_std_floor', 'policy_clip'}:
        raise ValueError('explicit GRPO floor and native verl clip settings required')
    floor, clip = settings['advantage_std_floor'], settings['policy_clip']
    if (not finite(floor) or floor <= 0 or not isinstance(clip, dict)
            or set(clip) != {'clip_ratio', 'clip_ratio_low', 'clip_ratio_high', 'clip_ratio_c'}
            or not finite(clip['clip_ratio']) or clip['clip_ratio'] < 0
            or not finite(clip['clip_ratio_c']) or clip['clip_ratio_c'] <= 1
            or any(v is not None and (not finite(v) or v < 0)
                   for v in (clip['clip_ratio_low'], clip['clip_ratio_high']))):
        raise ValueError('invalid GRPO normalization or clipping settings')
    return settings


@contextmanager
def use(settings):
    token = _active.set(validate(settings) if settings is not None else None)
    try:
        yield
    finally:
        _active.reset(token)


def current():
    return _active.get()
