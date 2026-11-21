"""Construct disjoint same-utterance pairs inside a static optimizer microbatch."""
from collections import defaultdict
import random


def schedule(rows, *, seed, epochs, global_batch, microbatch, confusion_fraction=0.):
    if (not rows or global_batch < 2 or microbatch < 2 or microbatch % 2 or
            global_batch % microbatch or epochs < 1 or not 0 <= confusion_fraction <= 1):
        raise ValueError("aligned even microbatch and positive epoch count required")
    batches = []
    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        groups = defaultdict(list)
        units, singles = [], []
        used = set()
        if confusion_fraction:
            # An anchor must actually misrank or tie its strongest wrong intent.
            # Prefer a reciprocal error; then use another recording of that intent.
            order = sorted(rows)
            rng.shuffle(order)
            pools = defaultdict(list)
            for uid in order:
                pools[rows[uid]['gold']].append(uid)
            for a in order:
                if len(used)+2 > len(rows)*confusion_fraction:
                    break
                ra = rows[a]
                if a in used or ra['ref_gap'] > 0:
                    continue
                partners = (b for b in pools[ra['competitor']] if b not in used and
                            rows[b]['audio_sha256'] != ra['audio_sha256'] and
                            (ra['group'] is None or rows[b]['group'] != ra['group']))
                b = min(partners, key=lambda b: not (rows[b]['ref_gap'] <= 0 and
                        rows[b]['competitor'] == ra['gold']), default=None)
                if b is not None:
                    used.update((a,b))
                    units.append(((a,b), 'confusion'))
        for uid, row in sorted(rows.items()):
            if uid in used:
                continue
            groups[row['group'], row['gold']].append(uid)
        for (group, _), ids in groups.items():
            rng.shuffle(ids)
            while group is not None and len(ids) > 1:
                a = ids.pop()
                j = next((i for i, b in enumerate(ids) if rows[a]['audio_sha256']
                          != rows[b]['audio_sha256']), None)
                if j is None:
                    singles.append(a)
                else:
                    units.append(((a, ids.pop(j)), "invariant"))
            singles.extend(ids)
        rng.shuffle(singles)
        # Place the odd single in its own unit; padding has zero loss.
        units.extend((singles[i:i+2], False) for i in range(0, len(singles), 2))
        rng.shuffle(units)
        slots = []
        for ids, paired in units:
            for uid in ids:
                slots.append(dict(id=uid, paired=paired=="invariant", confusion=paired=="confusion", padding=False, weight=1.))
            if len(ids) == 1:
                slots.append(dict(id=ids[0], paired=False, confusion=False, padding=True, weight=0.))
        while len(slots) % global_batch:
            slots.append(dict(id=slots[0]['id'], paired=False, confusion=False, padding=True, weight=0.))
        for start in range(0, len(slots), global_batch):
            batches.append(dict(epoch=epoch, index=len(batches), slots=slots[start:start+global_batch]))
    return batches
