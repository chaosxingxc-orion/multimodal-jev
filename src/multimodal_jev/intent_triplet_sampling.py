"""Choose three independent TRAIN records by intent; card rendering is downstream."""
from collections import defaultdict
import random
import math
from itertools import accumulate


class AnchorSampler:
    """Index the admitted TRAIN roster once. Draws only visit candidate classes."""
    def __init__(self, population, labels, k, seed, *, contrast=None):
        self.population = population
        self.contrast = contrast
        self.by_label = defaultdict(list)
        for uid, row in population.items():
            if row['role'] != 'learning' or row['official_split'] != 'train' or row['id'] != uid:
                raise ValueError('sampling requires indexed official TRAIN records')
            self.by_label[row['intent']].append(uid)
        if (set(self.by_label) != set(labels) or len(set(labels)) != len(labels)
                or len(labels) < 2 or type(k) is not int or k < 1
                or (contrast is None and k >= len(labels))):
            raise ValueError('invalid TRAIN vocabulary or negative card count')
        if contrast and not contrast.replacement:
            if ((contrast.audio_weight and contrast.cards >= len(labels))
                    or (contrast.card_weight and (contrast.recordings >= len(labels)
                        or min(map(len, self.by_label.values())) <= contrast.positives))):
                raise ValueError('not enough TRAIN candidates for sampling without replacement')
        self.positions = {}
        for members in self.by_label.values():
            if len(members) < 2:
                raise ValueError('each class needs an independent positive recording')
            members.sort()
            self.positions.update((uid, i) for i, uid in enumerate(members))
        self.k, self.seed = k, seed

    def draw(self, anchor, epoch, update, distribution):
        gold = self.population[anchor]['intent']
        members = self.by_label[gold]
        rng = random.Random(f'{self.seed}|online-triplet|{epoch}|{update}|{anchor}')
        pos = rng.randrange(len(members) - 1)
        positive = members[pos + (pos >= self.positions[anchor])]
        weights = distribution[gold]
        wrong = tuple(sorted(weights, key=lambda label: rng.expovariate(weights[label]))[:self.k])
        return positive, (rng.choice(self.by_label[wrong[0]]), wrong)

    def draw_contrast(self, anchor, epoch, update, distribution):
        d = self.contrast
        gold = self.population[anchor]['intent']
        members = self.by_label[gold]
        rng = random.Random(f'{self.seed}|card-contrast-v1|{epoch}|{update}|{anchor}')
        weights = distribution[gold]  # full-support EMA, excludes gold
        labels = tuple(weights)

        def classes(count):
            if d.replacement:
                return rng.choices(labels, weights=weights.values(), k=count)
            return sorted(labels, key=lambda lab: rng.expovariate(weights[lab]))[:count]

        wrong = classes(d.cards) if d.audio_weight else ()
        positives, negatives = (), ()
        if d.card_weight:
            # Skip the anchor in O(1) without copying a potentially large class roster.
            pool = range(len(members)-1)
            offsets = rng.choices(pool, k=d.positives) if d.replacement else rng.sample(pool, d.positives)
            index = self.positions[anchor]
            positives = tuple(members[pos + (pos >= index)] for pos in offsets)
            negatives = tuple((lab, rng.choice(self.by_label[lab])) for lab in classes(d.recordings))
        return positives, wrong, negatives


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
