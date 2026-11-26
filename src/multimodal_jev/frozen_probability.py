"""One immutable, shared logit calibration for a frozen Stage1 policy."""
import math
import torch


def validate(spec):
    fields = {'a', 'b', 'rank_weight', 'probability_weight'}
    if 'reference_logit_dtype' in spec:
        if spec['reference_logit_dtype'] != 'bfloat16':
            raise ValueError('unsupported recorded logit precision')
        fields |= {'reference_logit_dtype'}
    if set(spec) != fields or not all(
            type(spec[k]) in (float, int) and math.isfinite(spec[k]) for k in fields - {'reference_logit_dtype'}):
        raise ValueError('invalid frozen probability calibration')
    if min(spec['a'], spec['rank_weight'], spec['probability_weight']) <= 0:
        raise ValueError('positive slope and objective weights required')


def transform(logits, spec):
    return logits * spec['a'] + spec['b']


def reference_logits(probabilities, spec):
    # The audited BF16 head widens its emitted logits to FP32 before sigmoid.
    # Recover only the reference; never quantize the live side of the parity check.
    p = probabilities.double()
    if 'reference_logit_dtype' not in spec:
        return p.logit()
    if not ((p > 0) & (p < 1)).all():
        raise ValueError('cannot invert a saturated reference')
    grid = p.logit().to(torch.bfloat16)
    center = grid.double()
    neighbors = [torch.nextafter(grid, torch.full_like(grid, v)).double()
                 for v in (-math.inf, math.inf)]
    error = (center.sigmoid() - p).abs()
    # Require a unique recoverable source logit inside the FP32 sigmoid envelope.
    atol = torch.finfo(torch.float32).eps
    if not ((error <= atol) & ((neighbors[0].sigmoid()-p).abs() > atol)
            & ((neighbors[1].sigmoid()-p).abs() > atol)).all():
        raise ValueError('cached probability does not identify a unique original BF16 logit')
    return center


def parity(report, logits, reference_logits, spec, bound):
    actual = transform(logits.float(), spec).sigmoid().cpu()
    reference = transform(reference_logits.float(), spec).sigmoid().cpu()
    error = float((actual-reference).abs().max())
    flips = int(((actual >= .5) != (reference >= .5)).sum())
    report.update(calibrated_max_abs=error, calibrated_threshold_flips=flips,
                  passed=report['passed'] and error <= bound and flips == 0)
