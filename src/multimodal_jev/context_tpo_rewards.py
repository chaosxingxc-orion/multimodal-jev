"""Offline action utilities for a factorized Bernoulli Jev policy."""
from dataclasses import dataclass, fields
import math

import torch
from torch.nn import functional as F

from .context_tpo import Objective, objectives


@dataclass(frozen=True)
class RewardSpec:
    temperature: float
    class_utility: float
    rank: float
    log_rank_margin: float
    per_term_cap: float
    existence: float
    hard_negative: float
    margin: float
    probability_gap: float
    contrast: float
    consistency: float

    def __post_init__(self):
        if any(not math.isfinite(getattr(self, f.name)) or getattr(self, f.name) < 0 for f in fields(self)):
            raise ValueError('invalid reward settings')
        if self.temperature <= 0 or self.per_term_cap <= 0 or not 0 < self.probability_gap < 1:
            raise ValueError('invalid reward temperature, cap or margin')


def reward_components(reference, gold, utility, partners, paired, confusion, spec):
    """R(yes)-R(no); derivatives of joint paired costs include both directions."""
    p = reference.detach().double()
    if p.ndim != 2 or not ((p > 0) & (p < 1)).all():
        raise ValueError('interior reference required')
    z = p.logit()
    mask = F.one_hot(gold, p.shape[1]).bool()
    recall = mask * utility[:, None]
    prior = z + spec.class_utility * recall / spec.temperature
    peak = prior.masked_fill(mask, -torch.inf).amax(-1, keepdim=True)
    deficit = (spec.log_rank_margin - (prior.gather(1, gold[:, None]) - peak)).clamp_min(0)
    shift = (deficit * spec.temperature / 2).clamp_max(spec.per_term_cap)
    ranking = torch.where(mask, shift, -(shift + spec.temperature * (prior - peak)).clamp_min(0))
    signals = dict(class_utility=recall, rank=ranking)
    # A local first-order reward for a Bernoulli decision is -d cost / d p.
    # Evaluate at the fixed reference, and clip before combining reward terms.
    with torch.enable_grad():
        logits = z.float().detach().requires_grad_()
        opts = Objective(1,0,0,0,0,spec.probability_gap,0)
        _, costs = objectives(logits, logits.detach().log_softmax(-1), gold, partners, paired, opts, confusion)
        for name in ('existence','hard_negative','margin','contrast','consistency'):
            derivative, = torch.autograd.grad(costs[name].sum(), logits, retain_graph=True)
            marginal = -derivative.double() / (p * (1-p))
            signals[name] = marginal.clamp(-spec.per_term_cap,spec.per_term_cap)
    return signals


def compile_targets(reference, rows, gold, utility, batches, spec, *, reward_shift=None):
    output = torch.empty((max(b['epoch'] for b in batches)+1, *reference.shape))
    seen = torch.zeros(output.shape[:2],dtype=torch.bool)
    activity = {}
    changed = 0
    for batch in batches:
        slots = batch['slots']
        indexes = torch.tensor([rows[s['id']]['index'] for s in slots])
        paired = torch.tensor([s['paired'] for s in slots])
        confusion = torch.tensor([s['confusion'] for s in slots])
        real = torch.tensor([not s['padding'] for s in slots])
        positions = torch.arange(len(slots))
        partners = torch.where(paired | confusion, positions ^ 1, positions)
        p = reference[indexes]
        signals = reward_components(p, gold[indexes], utility[indexes], partners, paired, confusion, spec)
        advantage = sum(getattr(spec,name)*signal for name,signal in signals.items())
        for name, signal in signals.items():
            activity[name] = activity.get(name,0) + int((signal[real].abs().amax(-1)>0).sum())
        q_logits = (p.double().logit() + advantage/spec.temperature).float()
        if not torch.isfinite(q_logits).all():
            raise ValueError('nonfinite reward target')
        epoch = batch['epoch']
        keys = indexes[real]
        if seen[epoch,keys].any() or len(keys.unique())!=len(keys):
            raise ValueError('duplicate target in one epoch')
        seen[epoch,keys] = True
        if reward_shift is not None:
            reward_shift[epoch,keys] = (advantage/spec.temperature)[real]
        output[epoch,keys] = q_logits[real]
        changed += int((q_logits[real].argmax(-1)!=p[real].argmax(-1)).sum())
    if not seen.all():
        raise ValueError('incomplete target population')
    return output,dict(real_exposures=int(seen.sum()),active_by_signal=activity,changed_top1=changed)


def projection(logits, target_logits):
    """Per-record KL(q||actor), averaged over independent yes/no decisions."""
    z = logits.float()
    t = target_logits.detach().float()
    q = t.sigmoid()
    return (q*(F.logsigmoid(t)-F.logsigmoid(z)) +
            (-t).sigmoid()*(F.logsigmoid(-t)-F.logsigmoid(-z))).mean(-1)
