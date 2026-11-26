"""Bidirectional fixed reference utilities in independent Bernoulli score space."""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ContrastSpec:
    temperature: float
    probability_epsilon: float
    max_displacement: float
    audio_to_label: float
    label_to_audio: float
    brier: float

    def __post_init__(self):
        if (not all(math.isfinite(v) and v > 0 for v in vars(self).values())
                or not self.probability_epsilon < .5):
            raise ValueError("invalid contrast reward settings")


@torch.no_grad()
def contrast_utilities(p, gold, temperature):
    """Derivatives w.r.t. p of the sum of positive-record log shares.

    The sum gives per-record units (the direction mean times its positive
    denominator). Each column's other positives are absent from each contest.
    The caller supplies only real, fully observed, exclusive-event records.
    """
    if (p.ndim != 2 or min(p.shape) == 0 or p.shape[1] < 2
            or not ((p > 0) & (p < 1)).all() or gold.shape != (len(p),)
            or not ((gold >= 0) & (gold < p.shape[1])).all()):
        raise ValueError("complete observed native candidate matrix required")
    mask = F.one_hot(gold, p.shape[1]).bool()
    score = p.log() / temperature
    row = (mask.to(p.dtype) - score.softmax(dim=1)) / (temperature * p)
    # One denominator per positive: same-label negatives plus this positive.
    # Compute column contest probabilities in log space.
    negative_logsum = score.masked_fill(mask, -torch.inf).logsumexp(dim=0)
    log_denom = torch.logaddexp(score, negative_logsum[None, :])
    positive = (negative_logsum[None, :] - log_denom).exp()
    # For a negative member, sum the derivative from each positive contest
    # of its label. Reduction is over positives only; a class can be absent.
    log_inverse_sum = (-log_denom).masked_fill(~mask, -torch.inf).logsumexp(dim=0)
    negative = -(score + log_inverse_sum[None, :]).exp()
    column = torch.where(mask, positive, negative) / (temperature * p)
    return {
        "audio_to_label": row,
        "label_to_audio": column,
        "brier": 2 * (mask.to(p.dtype) - p),
    }, int((mask & ~torch.isfinite(negative_logsum)[None, :]).sum())


@torch.no_grad()
def project(reference, gold, spec, calibration=None):
    if not ((reference >= 0) & (reference <= 1)).all():
        raise ValueError("reference must contain probabilities")
    raw = reference.double()
    p = raw.clamp(spec.probability_epsilon, 1 - spec.probability_epsilon)
    utilities, skipped = contrast_utilities(p, gold, spec.temperature)
    if not all(torch.isfinite(v).all() for v in utilities.values()):
        raise ValueError("nonfinite contrast utility")
    cap = spec.max_displacement
    if calibration is not None:
        from .frozen_probability import transform, reference_logits
        calibrated = transform(reference_logits(raw, calibration), calibration)
        utilities['brier'] = 2 * (F.one_hot(gold, p.shape[1]).to(p.dtype) - calibrated.sigmoid())
        probability_target = calibrated + spec.brier * utilities['brier'].clamp(-cap, cap)
    logits = p.logit() + sum(getattr(spec, key) * v.clamp(-cap, cap)
                             for key, v in utilities.items() if calibration is None or key != "brier")
    audit = {
        "members": p.numel(),
        "boundary_clipped": int((p != raw).sum()),
        "column_positive_skipped": skipped,
        "column_positive_total": len(p),
        "clipped": {name: int((v.abs() > cap).sum()) for name, v in utilities.items()},
    }
    if calibration is not None:
        return (logits.float(), probability_target.float()), audit
    return logits.float(), audit
