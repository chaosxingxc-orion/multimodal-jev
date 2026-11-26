"""One immutable, shared logit calibration for a frozen Stage1 policy."""
import math
import torch


def validate(spec):
    if set(spec) != {'a', 'b', 'rank_weight', 'probability_weight'} or not all(
            type(spec[k]) in (float, int) and math.isfinite(spec[k]) for k in spec):
        raise ValueError('invalid frozen probability calibration')
    if min(spec['a'], spec['rank_weight'], spec['probability_weight']) <= 0:
        raise ValueError('positive slope and objective weights required')


def transform(logits, spec):
    return logits * spec['a'] + spec['b']


def parity(report, logits, reference_logits, spec, bound):
    actual = transform(logits.float(), spec).sigmoid()
    reference = transform(reference_logits.float(), spec).sigmoid()
    error = float((actual-reference).abs().max())
    flips = int(((actual >= .5) != (reference >= .5)).sum())
    report.update(calibrated_max_abs=error, calibrated_threshold_flips=flips,
                  passed=report['passed'] and error <= bound and flips == 0)
