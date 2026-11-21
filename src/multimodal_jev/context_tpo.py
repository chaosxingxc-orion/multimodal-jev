"""Full-slate, fixed-reference target policy projection for single-label Jev."""
from dataclasses import dataclass, fields
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Objective:
    temperature: float
    projection: float
    existence: float
    hard_negative: float
    margin: float
    probability_gap: float
    consistency: float
    contrast: float = 0.

    def __post_init__(self):
        if any(not math.isfinite(getattr(self, f.name)) or getattr(self, f.name) < 0
               for f in fields(self)):
            raise ValueError("finite nonnegative TPO parameters required")
        if self.temperature <= 0 or self.probability_gap >= 1:
            raise ValueError("invalid temperature or probability gap")


@torch.no_grad()
def target(reference_probability, gold, utility, temperature):
    """Input is the interior FP32 sigmoid cache, not an original-logit cache."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive temperature required")
    ref = reference_probability.detach().float()
    if ref.ndim != 2 or ref.size(1) < 2 or not ((ref > 0) & (ref < 1)).all():
        raise ValueError("complete interior reference slate required")
    if gold.shape != utility.shape or gold.shape != (len(ref),):
        raise ValueError("one gold and utility per audio required")
    if not torch.isfinite(utility).all() or (utility <= 0).any():
        raise ValueError("positive finite recall utility required")
    scores = torch.logit(ref)
    scores.scatter_add_(1, gold[:, None], utility.float()[:, None] / temperature)
    return scores.log_softmax(-1)


def objectives(logits, log_target, gold, partner, paired, cfg, confusion=None):
    """Per-record loss. Partner indexes are local to this same forward."""
    z = logits.float()
    mask = F.one_hot(gold, z.size(1)).bool()
    positive = z.gather(1, gold[:, None]).squeeze(1)
    hardest = z.masked_fill(mask, -torch.inf).amax(-1)
    logq = log_target.detach()
    projection = (logq.exp() * (logq - z.log_softmax(-1))).sum(-1)
    existence = F.binary_cross_entropy_with_logits(z, mask.float(), reduction='none').mean(-1)
    hard_negative = (F.softplus(-positive) + F.softplus(hardest)) / 2
    gap = positive.sigmoid() - hardest.sigmoid()
    # Stable Bernoulli JS in log space, with gradients through both recordings.
    logp = torch.stack((F.logsigmoid(z), F.logsigmoid(-z)), -1)
    logr = logp[partner]
    logm = torch.logaddexp(logp, logr) - math.log(2)
    js = ((logp.exp() * (logp - logm) + logr.exp() * (logr - logm))
          .sum(-1).mean(-1) * .5) * paired
    # For an (A, B) pair compare the SAME candidate across two recordings.
    # This is separate from the strongest-negative gap within one recording.
    cross_gap = positive.sigmoid() - z[partner].gather(1,gold[:,None]).squeeze(1).sigmoid()
    contrast = (torch.zeros_like(positive) if confusion is None else
                F.relu(cfg.probability_gap - cross_gap) * confusion)
    terms = dict(contrast=contrast, projection=projection, existence=existence, hard_negative=hard_negative,
                 margin=F.relu(cfg.probability_gap - gap), consistency=js)
    total = sum(getattr(cfg, name) * value for name, value in terms.items())
    return total, dict(terms, probability_gap=gap)
