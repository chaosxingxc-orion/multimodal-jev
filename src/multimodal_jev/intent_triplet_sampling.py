"""Choose three independent TRAIN records by intent; card rendering is downstream."""
from collections import defaultdict
import random
import math
from itertools import accumulate


def sample_epoch(records, *, seed, epoch, distribution=None, negative_cards_per_anchor=None):
    by_label = defaultdict(list)
    by_id = {}
    for row in records:
        if row['role'] != 'learning' or row['official_split'] != 'train':
            raise ValueError('sampling requires learning recordings from official TRAIN')
        uid, label = row['id'], row['intent']
        if not label or uid in by_id:
            raise ValueError('each recording needs one intent and a unique ID')
        by_id[uid] = row
        by_label[label].append(uid)

    labels = sorted(by_label)
    if len(labels) < 2:
        raise ValueError('an independently labeled negative is required')
    k = negative_cards_per_anchor
    if k is not None and (type(k) is not int or not 1 <= k < len(labels)):
        raise ValueError('negative_cards_per_anchor must fit the non-gold vocabulary')
    # Separate RNG keeps the historical positive and first negative draws intact.
    extra_rng = random.Random(f'{seed}|additional-cards|{epoch}')
    draws = {}
    if distribution is not None:
        if set(distribution) != set(labels):
            raise ValueError('incomplete anchor classes')
        for label in labels:
            others = [key for key in labels if key != label]
            weights = distribution[label]
            if (set(weights) != set(others) or any(not math.isfinite(x) or x <= 0 for x in weights.values())
                    or not math.isclose(sum(weights.values()),1.,abs_tol=1e-9)):
                raise ValueError('normalized full-support distribution over other classes required')
            draws[label] = others, list(accumulate(weights[key] for key in others))
    label_positions = {label: index for index, label in enumerate(labels)}
    for ids in by_label.values():
        ids.sort()
        if len(ids) < 2:
            raise ValueError('no other recording for intent '+by_id[ids[0]]['intent'])
    positions = {uid: i for ids in by_label.values() for i, uid in enumerate(ids)}
    order = sorted(by_id)
    rng = random.Random(f'{seed}|triplets|{epoch}')
    rng.shuffle(order)
    for anchor in order:
        label = by_id[anchor]['intent']
        members = by_label[label]
        pos = rng.randrange(len(members)-1)
        positive = members[pos + (pos >= positions[anchor])]
        if draws:
            classes, cumulative = draws[label]
            neg_label = rng.choices(classes,cum_weights=cumulative,k=1)[0]
        else:
            other = rng.randrange(len(labels)-1)
            neg_label = labels[other + (other >= label_positions[label])]
        negative = rng.choice(by_label[neg_label])
        if k is None:
            yield anchor, positive, negative
            continue
        available = [key for key in labels if key not in (label, neg_label)]
        if distribution is None:
            extra = extra_rng.sample(available, k - 1)
        else:
            # Independent exponential clocks give weighted draws without replacement.
            extra = (sorted(available, key=lambda key: extra_rng.expovariate(distribution[label][key]))
                     [:k - 1] if k > 1 else [])
        yield anchor, positive, (negative, (neg_label, *extra))
