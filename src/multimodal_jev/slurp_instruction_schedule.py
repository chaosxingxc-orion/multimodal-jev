"""Complete class-interleaved epochs, label-independent instruction selection.

Every recording appears once with weight one. Only the last DP batch pads with
zero-weight copies; padding is excluded from both loss and exposure statistics.
"""
from collections import defaultdict
import hashlib
import math

from .slurp_training import content_hash
from .slurp_official_training import VIEWS
from .span_candidate_intervention import VARIANTS


def _key(seed, *values):
    return hashlib.sha256(':'.join(map(str, (seed, *values))).encode()).hexdigest()


def epoch_schedule(rows, *, seed, epoch, global_batch=128, world_size=4):
    if (type(seed) is not int or type(epoch) is not int or epoch < 0
            or type(global_batch) is not int or type(world_size) is not int
            or world_size < 1 or global_batch < world_size or global_batch % world_size):
        raise ValueError('fixed seed, nonnegative epoch and equal DP rank layout required')
    if (not rows or len({r['id'] for r in rows}) != len(rows)
            or any(r['role'] != 'learning' or r['known_pending_source'] for r in rows)):
        raise ValueError('distinct admitted learning rows required')
    by_class = defaultdict(list)
    for row in rows:
        by_class[row['intent']].append(row)
    labels = sorted(by_class, key=lambda c: _key(seed, 'class', epoch, c))
    for label in labels:
        by_class[label].sort(key=lambda r: _key(seed, 'source', epoch, r['id']))
    ordered = []
    for offset in range(max(map(len, by_class.values()))):
        # Interleave available classes without oversampling tails or dropping heads.
        for label in labels:
            if offset < len(by_class[label]):
                ordered.append(by_class[label][offset])
    slots = []
    for row in ordered:
        view_index = (int(_key(seed, 'view', row['id']), 16) + epoch) % len(VIEWS)
        variant_index = (int(_key(seed, 'binding', epoch, row['id']), 16)) % len(VARIANTS)
        slots.append(dict(id=row['id'], source_group=row['source_group'], view=VIEWS[view_index],
                          variant=VARIANTS[variant_index], weight=1., padding=False))
    real = len(slots)
    while len(slots) % global_batch:
        slots.append(dict(slots[(len(slots)-real) % real], weight=0., padding=True))
    batches = [slots[i:i+global_batch] for i in range(0, len(slots), global_batch)]
    identity = dict(schema='slurp-full-instruction-epoch-v1', seed=seed, epoch=epoch,
                    global_batch=global_batch, world_size=world_size,
                    population_sha256=content_hash(sorted(rows, key=lambda r: r['id'])),
                    batches_sha256=content_hash(batches))
    return batches, dict(**identity, schedule_sha256=content_hash(identity), recordings=real,
                          source_groups=len({r['source_group'] for r in rows}),
                          updates=len(batches), padding_slots=len(slots)-real,
                          class_interleaved=True, strictly_balanced_batches=False,
                          oversampling=False, outer_inverse_class_weight=False)


def resume_cursor(schedule, *, next_batch, actual_identity):
    if (actual_identity != schedule['schedule_sha256'] or type(next_batch) is not int
            or not 0 <= next_batch <= schedule['updates']):
        raise ValueError('exact epoch schedule and bounded next-batch cursor required')
    return dict(schedule_sha256=actual_identity, next_batch=next_batch,
                epoch=schedule['epoch'], complete=next_batch == schedule['updates'])
