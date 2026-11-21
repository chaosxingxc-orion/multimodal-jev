"""Paired Noul / open-Choice TPO on matched native intent definitions."""
import math

import torch
import torch.nn.functional as F

from .context_choice import distribution


@torch.no_grad()
def targets(noul_recent, choice_recent, noul_anchor, choice_anchor, indices, gold, spec):
    """Return detached Bernoulli target log odds and categorical target log masses.

    Inputs in a call have the same slate width. Logits include the final coverage
    marker on Choice; gold and indices both refer to native ontology positions.
    """
    if noul_recent.ndim != 2 or indices.ndim != 2:
        raise ValueError('expected batched native panel and slate')
    batch, k = noul_recent.shape
    l = indices.size(1)
    if (indices.size(0) != batch or gold.shape != (batch,) or k < 2 or not 0 < l <= k
            or choice_recent.shape != (batch, l + 1) or noul_anchor.shape != noul_recent.shape
            or choice_anchor.shape != choice_recent.shape or indices.dtype != torch.long
            or gold.dtype != torch.long):
        raise ValueError('invalid paired geometry or index dtype')
    # IDs/ranges/uniqueness are also validated at data admission.
    if ((indices < 0) | (indices >= k)).any() or ((gold < 0) | (gold >= k)).any() or (
            indices.sort(dim=1).values.diff(dim=1) == 0).any():
        raise ValueError('invalid or duplicate native intent')
    rho, tau, temperature, margin = (spec[key] for key in ('rho', 'tau', 'temperature', 'margin'))
    weights = [spec[key] for key in ('classification_weight', 'ranking_weight',
                                     'probability_weight', 'consistency_weight')]
    if (not 0 <= rho <= 1 or not 0 <= margin <= 1 or not math.isfinite(tau) or tau <= 0
            or not all(math.isfinite(v) and v >= 0 for v in weights)):
        raise ValueError('invalid TPO coefficients')
    n = noul_recent.float()
    p = n.sigmoid()
    logc = distribution(choice_recent, 'open', temperature)
    c = logc.exp()
    label = F.one_hot(gold, k).bool()
    hits = indices == gold[:, None]
    selected = torch.cat((hits, ~hits.any(-1, keepdim=True)), dim=-1)
    # Binary utilities below are yes-minus-no. The true action for a Choice
    # miss is NONE; a competing semantic action can only receive a penalty.
    def margin_signal(prob, truth):
        wrong = prob.masked_fill(truth, -torch.inf).argmax(-1)
        rival = F.one_hot(wrong, prob.size(1)).bool()
        active = (prob * truth).sum(-1) - prob.gather(1, wrong[:, None]).squeeze(1) < margin
        return (truth.float() - rival.float()) * active[:, None]

    # Coverage is the mass on the slate. For misses, all candidates have the
    # same derivative, so they receive no category-specific supervision here.
    cov_error = hits.any(-1).float() - (1 - c[:, -1])
    cov_signal = torch.zeros_like(c)
    cov_signal[:, -1] = -2 * cov_error
    diff = p.gather(1, indices) - c[:, :-1]
    n_cons = torch.zeros_like(p).scatter(1, indices, -2 * diff / l)
    c_cons = F.pad(2 * diff / l, (0, 1))
    n_terms = (2 * label.float() - 1, margin_signal(p, label),
               2 * (label.float() - p), n_cons)
    c_terms = (selected.float(), margin_signal(c, selected), cov_signal, c_cons)
    # Bound each centered action utility. Binary centering gives ±gap/2.
    n_shift = sum(w * (term / 2).clamp(-1, 1) * 2 for w, term in zip(weights, n_terms))
    c_shift = sum(w * (term - term.mean(-1, keepdim=True)).clamp(-1, 1)
                  for w, term in zip(weights, c_terms))
    qn = (1 - rho) * n + rho * noul_anchor.float() + n_shift / tau
    refc = distribution(choice_anchor, 'open', temperature)
    qc = ((1 - rho) * logc + rho * refc + c_shift / tau).log_softmax(-1)
    return qn, qc


def losses(noul_actor, choice_actor, targets, spec):
    """KL(target || actor) per recording, before global mask/DP averaging."""
    return (view_loss(noul_actor, targets[0], 'noul', spec),
            view_loss(choice_actor, targets[1], 'open', spec))


def view_loss(actor, target, mode, spec):
    """The two KL terms can backpropagate in separate microbatches."""
    q = target.detach().float()
    z = actor.float()
    if z.shape != q.shape:
        raise ValueError('actor/target geometry differs')
    if mode == 'open':
        return (q.exp() * (q - distribution(z, 'open', spec['temperature']))).sum(-1)
    if mode != 'noul':
        raise ValueError('unknown joint TPO view')
    yes = q.sigmoid() * (F.logsigmoid(q) - F.logsigmoid(z))
    no = (-q).sigmoid() * (F.logsigmoid(-q) - F.logsigmoid(-z))
    return (yes + no).mean(-1)
