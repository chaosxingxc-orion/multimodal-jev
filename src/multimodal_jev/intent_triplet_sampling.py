"""Choose three independent TRAIN records by intent; card rendering is downstream."""
from collections import defaultdict
import random


def sample_epoch(records, *, seed, epoch):
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
        other = rng.randrange(len(labels)-1)
        neg_label = labels[other + (other >= label_positions[label])]
        negative = rng.choice(by_label[neg_label])
        yield anchor, positive, negative
