"""Select a fixed TRAIN view using a shared, frozen probability scale."""
from collections import Counter, defaultdict, deque
import hashlib
import math

import numpy as np


def mine(clean, probabilities, labels, policy):
    a, b = policy['a'], policy['b']
    threshold, margin = policy['confidence'], policy['boundary_gap']
    budget, replay_budget = policy['budget'], policy['replay_recordings']
    if (not math.isfinite(a) or a <= 0 or not math.isfinite(b)
            or not 0 < threshold < 1 or not 0 < margin < 1
            or not 0 < replay_budget < budget <= len(clean)):
        raise ValueError('invalid calibrated mining policy')
    ids = sorted(clean)
    raw = np.asarray([probabilities[uid] for uid in ids], dtype=np.float64)
    if raw.shape != (len(ids), len(labels)) or not np.all((raw > 0) & (raw < 1)):
        raise ValueError('calibrated mining requires complete interior scores')
    label_index = {label: i for i, label in enumerate(labels)}
    gold = np.array([label_index[clean[uid]['intent']] for uid in ids])
    ix = np.arange(len(ids))
    neg = raw.copy()
    neg[ix, gold] = -1
    rival = neg.argmax(1)
    pos, neg = raw[ix, gold], raw[ix, rival]

    def transform(p):
        return np.exp(-np.logaddexp(0, -(a * (np.log(p) - np.log1p(-p)) + b)))

    q_pos, q_neg = transform(pos), transform(neg)
    gap = q_pos - q_neg
    strata = np.select(
        [pos < neg, pos == neg, q_neg > threshold, gap < margin, q_pos <= threshold],
        ['error', 'tie', 'high_negative', 'boundary', 'low_positive'], default='replay',
    )
    tiers = defaultdict(list)
    for i, name in enumerate(strata.tolist()):
        tiers[name].append(i)

    # Prefer source coverage within each tier, while cycling intent, rival and
    # recording type. Use raw ranks to preserve ties even under saturation.
    groups = set()

    def choose(indices, count):
        if count == 0:
            return []
        if len(indices) < count:
            raise ValueError('insufficient recordings in requested mining tier')
        buckets = defaultdict(deque)

        def key(i):
            return hashlib.sha256(f"{policy['seed']}:{ids[i]}".encode()).digest()

        for i in sorted(indices, key=key):
            buckets[(int(gold[i]), int(rival[i]), clean[ids[i]]['synthetic'])].append(i)
        active = deque(buckets[k] for k in sorted(buckets, key=lambda k: key(buckets[k][0])))
        selected, deferred = [], []
        while active and len(selected) < count:
            bucket = active.popleft()
            i = bucket.popleft()
            group = clean[ids[i]]['group']
            if group in groups:
                deferred.append(i)
            else:
                selected.append(i)
                groups.add(group)
            if bucket:
                active.append(bucket)
        selected.extend(deferred[:count-len(selected)])
        return selected

    hard = [i for k in ('error', 'tie', 'high_negative', 'boundary') for i in tiers[k]]
    remaining = budget - replay_budget - len(hard)
    if remaining < 0:
        raise ValueError('mandatory calibrated hard tiers exceed the budget')
    groups.update(clean[ids[i]]['group'] for i in hard)
    picked = sorted(hard + choose(tiers['low_positive'], remaining) + choose(tiers['replay'], replay_budget))
    if len(set(picked)) != budget:
        raise ValueError('calibrated view does not cover the exact requested budget')
    rows = {}
    for n, i in enumerate(picked):
        uid = ids[i]
        row = clean[uid].copy()
        del row['text']
        rows[uid] = dict(row, index=n, gold=int(gold[i]), competitor=int(rival[i]),
                         ref_gap=float(pos[i]-neg[i]), stratum=str(strata[i]),
                         calibrated_gold=float(q_pos[i]), calibrated_rival=float(q_neg[i]),
                         calibrated_gap=float(gap[i]))
    evidence = dict(eligible_strata=dict(Counter(strata.tolist())), hard=budget-replay_budget,
                    replay_requested=replay_budget, replay_actual=replay_budget, replay_shortfall=0,
                    overlapping=dict(high_negative=int((q_neg > threshold).sum()),
                                     low_positive=int((q_pos <= threshold).sum()),
                                     boundary=int(((pos > neg) & (gap < margin)).sum())))
    return rows, evidence
