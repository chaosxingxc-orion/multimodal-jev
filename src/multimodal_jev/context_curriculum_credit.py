"""Annealed supervised-to-RL curriculum contracts; no model, engine or dispatch.

Stage weights are a convex schedule over a declared update total: the supervised
template/representation terms start dominant and anneal to exactly zero, while
the sampled-reward mixture moves towards listwise ranking. All three rewards are
functions of one shared Plackett-Luce trajectory drawn from the current policy,
so they share one exact FP64 log probability and one sampled action; nothing
here is differentiated. Only the supervised terms stay differentiable in the
trainer. The candidate builds the clipped surrogate from the returned log_prob.
"""
import hashlib
import math

import torch


SCHEDULES = ('annealed_convex_linear_v1',)
ALGORITHMS = ('curriculum_grpo_floor_clipped_v1',)


def schedule(update, total, *, name='annealed_convex_linear_v1',
             start=(0.5, 0.3, 0.2), end=(0.0, 0.4, 0.6)):
    """Declared (supervised, contrast, ranking) weights at one optimizer update.

    Both endpoints are exact; interior values interpolate linearly and the three
    weights sum to one at every update. Values are experiment constants, not
    tunables: callers must pass the pre-registered endpoints unchanged.
    """
    if (name not in SCHEDULES or type(update) is not int or type(total) is not int
            or total < 1 or not 0 <= update <= total):
        raise ValueError('registered schedule and declared update inside the run required')
    for group in (start, end):
        if (not isinstance(group, tuple) or len(group) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in group)
                or abs(sum(group) - 1.0) > 1e-12):
            raise ValueError('nonnegative convex start/end weights required')
    if update == 0:
        weights = tuple(float(v) for v in start)
    elif update == total:
        weights = tuple(float(v) for v in end)
    else:
        weights = tuple(a + (b - a) * update / total for a, b in zip(start, end, strict=True))
    return dict(supervised=weights[0], contrast=weights[1], ranking=weights[2])


def _probability(weights):
    if (not isinstance(weights, torch.Tensor) or weights.ndim != 1
            or weights.dtype != torch.float32
            or not torch.isfinite(weights).all() or ((weights <= 0) | (weights >= 1)).any()):
        raise ValueError('finite interior FP32 candidate probabilities required; no clipping')
    # Rewards are scalar contracts, never differentiable losses.
    return weights.double().detach()


def _labels(labels, weights):
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.bool
            or labels.shape != weights.shape):
        raise ValueError('complete Boolean candidate labels required')


def _group_size(group_size):
    if type(group_size) is not int or group_size < 1:
        raise ValueError('positive integer group size required')
    return group_size


def _generator(seed, update, query_id, channel):
    if (type(seed) is not int or seed < 0 or type(update) is not int or update < 0
            or any(type(x) is not str or not x for x in (query_id, channel))):
        raise ValueError('explicit seed, update, query identity and reward channel required')
    # Unambiguous tuple identity avoids the ordinal-sum collisions of the old
    # key (e.g. 'ab' vs 'ba') so distinct queries get distinct draw streams.
    key = repr((seed, update, query_id, channel))
    digest = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63 - 1)
    return torch.Generator(device='cpu').manual_seed(digest)


def _draw_without_replacement(weights, labels, *, group_size, k, generator):
    """Plackett-Luce draws of k distinct candidates per group, gold-independent."""
    if type(group_size) is not int or group_size < 2 or type(k) is not int or not 1 <= k <= len(weights):
        raise ValueError('complete group and draw size inside the menu required')
    uniform = torch.rand(group_size, len(weights), generator=generator, dtype=torch.float64)
    # Gumbel-top-k is an exact without-replacement categorical sampler.
    keys = torch.log(weights)[None, :] - torch.log(-torch.log(uniform))
    order = torch.argsort(keys, dim=1, descending=True)[:, :k]
    return order, labels[order]


def plackett_luce_log_prob(weights, draws):
    """Exact FP64 log probability of ordered draws without replacement.

    For an ordered sequence c_1..c_k the probability is
    prod_j w_{c_j} / sum_{i not already drawn} w_i. draws may have any shape
    ending in k; the result has draws.shape[:-1]. No gradient is retained.
    """
    p = _probability(weights).detach()
    if (not isinstance(draws, torch.Tensor) or draws.dtype != torch.long
            or draws.ndim < 1 or draws.shape[-1] < 1):
        raise ValueError('long ordered candidate indices required')
    if draws.numel() == 0:
        return torch.empty(draws.shape[:-1], dtype=torch.float64)
    if int(draws.min()) < 0 or int(draws.max()) >= len(p):
        raise ValueError('draw indices inside the menu required')
    rows = draws.reshape(-1, draws.shape[-1])
    if any(len(torch.unique(row)) != row.numel() for row in rows):
        raise ValueError('distinct candidates inside each ordered draw required')
    terms = []
    for row in rows:
        remaining = p.clone()
        total = remaining.sum()
        accum = torch.zeros((), dtype=torch.float64)
        for index in row.tolist():
            accum = accum + torch.log(remaining[index] / total)
            total = total - remaining[index]
            remaining[index] = 0
        terms.append(accum)
    return torch.stack(terms).reshape(draws.shape[:-1])


def trajectory(weights, labels, *, k, group_size=16, margin=0.2, seed=317, update=0, query_id='q'):
    """One shared Plackett-Luce trajectory and all three rewards on it.

    The k ordered candidates are drawn once per group from the current policy.
    rank uses the first three positions, brier averages squared error over the
    same k candidates, and margin pairs drawn gold with drawn non-gold inside the
    same trajectory. When the trajectory has no gold or no non-gold, margin is
    zero and margin_reason names the empty branch; nothing is silently dropped.
    """
    _labels(labels, weights)
    p = _probability(weights)
    _group_size(group_size)
    if type(k) is not int or not 1 <= k <= len(p):
        raise ValueError('trajectory length inside the menu required')
    if (type(margin) not in (int, float) or not math.isfinite(margin)
            or not 0 <= margin < 1):
        raise ValueError('declared margin in [0,1) required')
    draws, hits = _draw_without_replacement(p, labels, group_size=group_size, k=k,
                                            generator=_generator(seed, update, query_id, 'trajectory'))
    log_prob = plackett_luce_log_prob(weights, draws)
    top1 = hits[:, :1].any(dim=1)
    top3 = hits[:, :min(3, k)].any(dim=1)
    rank = torch.where(top1, torch.ones_like(top3, dtype=torch.float64),
                       torch.where(top3, torch.full_like(top3, 0.5, dtype=torch.float64),
                                   torch.zeros_like(top3, dtype=torch.float64)))
    brier = 1.0 - ((p[draws] - labels[draws].double()) ** 2).mean(dim=1)
    margins = []
    reasons = []
    for row_draws, row_hits in zip(draws, hits):
        golds = p[row_draws[row_hits]]
        negatives = p[row_draws[~row_hits]]
        if golds.numel() == 0:
            margins.append(0.0)
            reasons.append('no_gold_in_trajectory')
        elif negatives.numel() == 0:
            margins.append(0.0)
            reasons.append('no_non_gold_in_trajectory')
        else:
            gap = golds[:, None] - negatives[None, :]
            margins.append(float((gap >= margin).double().mean()))
            reasons.append('ok')
    return dict(draws=draws, log_prob=log_prob.detach(), rank=rank.float(),
                top1=int(top1.sum()), top3=int(top3.sum()),
                brier=brier.clamp(0.0, 1.0).float(),
                margin=torch.tensor(margins, dtype=torch.float32),
                margin_reason=tuple(reasons))


def rank_tiers(weights, labels, *, group_size=16, seed=317, update=0, query_id='q', k=None):
    """Three-tier listwise reward from the first three positions of the trajectory.

    top-1 hit 1.0, top-3 hit 0.5, otherwise 0. k defaults to the trajectory
    length min(8, menu size); pass k explicitly to force the shared length.
    """
    out = trajectory(weights, labels, k=min(8, len(weights)) if k is None else k,
                     group_size=group_size, seed=seed, update=update, query_id=query_id)
    return dict(reward=out['rank'], top1=out['top1'], top3=out['top3'],
                draws=out['draws'], log_prob=out['log_prob'])


def brier_reward(weights, labels, *, group_size=16, subset=8, seed=317, update=0,
                 query_id='q', k=None):
    """Calibration reward 1 - mean squared error on the shared trajectory.

    subset is kept as the trajectory length for backward compatibility; pass k
    to use the same explicit length as trajectory(). Labels never enter the draw.
    """
    out = trajectory(weights, labels, k=subset if k is None else k,
                     group_size=group_size, seed=seed, update=update, query_id=query_id)
    return dict(reward=out['brier'], picks=out['draws'], draws=out['draws'],
                log_prob=out['log_prob'])


def margin_reward(weights, labels, *, group_size=16, negatives=8, margin=0.2,
                  seed=317, update=0, query_id='q', k=None):
    """Contrast reward from drawn gold x drawn non-gold inside one trajectory.

    negatives is kept as the trajectory length for backward compatibility; pass
    k to use the same explicit length as trajectory(). The reward is the fraction
    of pairs with p_gold - p_negative >= margin. Empty gold or non-gold branches
    return zero plus margin_reason, never an exception or a silent drop.
    """
    length = negatives if k is None else k
    out = trajectory(weights, labels, k=length, group_size=group_size, margin=margin,
                     seed=seed, update=update, query_id=query_id)
    return dict(reward=out['margin'], margin=float(margin), negatives=length,
                picks=out['draws'], draws=out['draws'], margin_reason=out['margin_reason'],
                log_prob=out['log_prob'])


def composite(rewards, *, contrast_fraction=0.5):
    """Weighted reward mixture from the declared schedule weights.

    Weights arrive from schedule() and must stay a convex combination; the
    contrast family splits its share evenly between the margin and calibration
    rewards unless the experiment declares a different pre-registered fraction.
    """
    if (type(contrast_fraction) not in (int, float) or not math.isfinite(contrast_fraction)
            or not 0 < contrast_fraction < 1):
        raise ValueError('declared contrast/margin split inside (0,1) required')
    if not isinstance(rewards, dict):
        raise ValueError('named reward mapping required')
    try:
        weights = rewards['weights']
        terms = tuple(rewards[name] for name in ('margin', 'brier', 'rank'))
    except KeyError as exc:
        raise ValueError('schedule weights and all three named rewards required') from exc
    if (not isinstance(weights, dict) or set(weights) != {'supervised', 'contrast', 'ranking'}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                   for v in weights.values())
            or abs(sum(weights.values()) - 1.0) > 1e-12):
        raise ValueError('nonnegative convex schedule weights required')
    for term in terms:
        if (not isinstance(term, torch.Tensor) or term.ndim != 1 or term.dtype != torch.float32
                or term.requires_grad or not torch.isfinite(term).all()
                or ((term < 0) | (term > 1)).any() or term.shape != terms[0].shape):
            raise ValueError('detached bounded FP32 group reward vectors required')
    return (weights['contrast'] * (contrast_fraction * rewards['margin']
                                   + (1 - contrast_fraction) * rewards['brier'])
            + weights['ranking'] * rewards['rank'])


def advantages(rewards, algorithm='curriculum_grpo_floor_clipped_v1', *, floor=0.25):
    """GRPO-style component centering for continuous bounded group rewards.

    The existing dual-credit contract only admits {0,1} rewards; this curriculum
    admits any value in [0,1], so the validator is relaxed and the same
    population-std floor is kept.
    """
    if (not isinstance(rewards, torch.Tensor) or algorithm not in ALGORITHMS
            or rewards.ndim != 2 or rewards.shape[1] < 2
            or rewards.requires_grad or rewards.dtype != torch.float32
            or not torch.isfinite(rewards).all()
            or not ((rewards >= 0) & (rewards <= 1)).all()):
        raise ValueError('detached bounded M x G rewards and registered algorithm required')
    centered = rewards - rewards.mean(-1, keepdim=True)
    return centered / rewards.std(-1, unbiased=False, keepdim=True).clamp_min(floor)


def surrogate_terms(log_prob, scores, *, algorithm='curriculum_grpo_floor_clipped_v1', floor=0.25):
    """Detached GRPO terms for one shared trajectory.

    scores are the composite rewards; advantages use the existing group centering
    and population-std floor. The candidate forms the probability ratio from
    log_prob outside this module. No autograd is introduced here; the returned
    tensors are detached and shape-matched.
    """
    if (not isinstance(log_prob, torch.Tensor) or not isinstance(scores, torch.Tensor)
            or log_prob.shape != scores.shape or log_prob.ndim not in (1, 2)
            or log_prob.dtype not in (torch.float32, torch.float64)
            or log_prob.requires_grad or not torch.isfinite(log_prob).all()
            or bool((log_prob > 0).any())):
        raise ValueError('detached finite nonpositive log-probabilities matching composite scores required')
    advantages_1d = scores.ndim == 1
    values = scores.unsqueeze(0) if advantages_1d else scores
    centered = advantages(values, algorithm, floor=floor)
    if advantages_1d:
        centered = centered[0]
    return dict(log_prob=log_prob.detach(), scores=scores.detach(),
                advantages=centered.detach())
