"""Freeze optimizer batches independently of the batches used to project TPO targets."""
from bisect import bisect_left
from collections import Counter
from math import gcd

from .card_token_buckets import BucketCursor, validate


def geometry(config):
    rows = config['buckets']
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) or set(row) !=
                   {'max_tokens', 'batch_size_per_rank', 'microbatch_per_rank'} for row in rows)
            or rows[-1]['max_tokens'] is not None):
        raise ValueError('each Stage2 bucket needs a token bound, batch and microbatch width')
    spec = dict(max_tokens=[r['max_tokens'] for r in rows[:-1]],
                max_anchors=[r['batch_size_per_rank'] for r in rows],
                lookahead_batches=config['lookahead_batches'])
    validate(spec)
    micros = [r['microbatch_per_rank'] for r in rows]
    if any(type(m) is not int or m < 1 or cap % m for cap, m in zip(spec['max_anchors'], micros, strict=True)):
        raise ValueError('Stage2 microbatch must divide its bucket width')
    return spec, micros


def make_batches(projected, lengths, config, world, epochs):
    """Preserve each (epoch, input) coordinate, independent of the frozen reward group."""
    spec, micros = geometry(config)
    orders = [[] for _ in range(epochs)]
    for i, batch in enumerate(projected):
        epoch = batch['epoch']
        if batch['index'] != i or not 0 <= epoch < epochs:
            raise ValueError('invalid projected target schedule')
        for slot in batch['slots']:
            if slot['paired'] or slot['confusion']:
                raise ValueError('cannot regroup a batch-coupled actor objective')
            if not slot['padding']:
                orders[epoch].append(slot['id'])
    if any(set(order) != lengths.keys() for order in orders):
        raise ValueError('optimizer population differs from projected targets')
    cursor = BucketCursor(orders, spec, world)
    batches = []
    while cursor.epoch < epochs:
        ids, epoch, bucket, end, maximum = cursor.take(lengths.__getitem__)
        slots = [dict(id=uid, weight=1, padding=False, paired=False, confusion=False) for uid in ids]
        slots.extend(dict(slots[0], weight=0, padding=True) for _ in range((-len(ids)) % world))
        local = len(slots) // world
        batches.append(dict(index=len(batches), epoch=epoch, epoch_end=end, bucket=bucket,
                            max_tokens=maximum, local_batch=local, microbatch=gcd(local, micros[bucket]),
                            real=len(ids), slots=slots))
    return batches


def check_batches(batches, lengths, config, world, epochs):
    spec, micros = geometry(config)
    seen = [set() for _ in range(epochs)]
    totals = Counter()
    epoch = 0
    for i, b in enumerate(batches):
        slots = b['slots']
        uids = [s['id'] for s in slots]
        real = [s['id'] for s in slots if not s['padding']]
        widths = [lengths[uid] for uid in uids]
        maximum = max(widths)
        bucket = bisect_left(spec['max_tokens'], maximum)
        local = -(-len(real) // world)
        if (epoch >= epochs or b['epoch'] != epoch or b['index'] != i or not real
                or len(set(real)) != len(real) or seen[epoch].intersection(real)
                or any(type(s['padding']) is not bool or s['weight'] != int(not s['padding'])
                       or s['paired'] or s['confusion'] for s in slots)
                or uids[:len(real)] != real or any(uid != real[0] for uid in uids[len(real):])
                or len(uids) != local * world or not 0 < local <= spec['max_anchors'][bucket]
                or bisect_left(spec['max_tokens'], min(widths)) != bucket
                or (b['real'], b['bucket'], b['local_batch'], b['microbatch'], b['max_tokens'])
                   != (len(real), bucket, local, gcd(local, micros[bucket]), maximum)):
            raise ValueError('invalid Stage2 bucket update geometry or coverage')
        seen[epoch].update(real)
        if b['epoch_end'] is not (len(seen[epoch]) == len(lengths)):
            raise ValueError('incomplete Stage2 bucket epoch')
        epoch += int(b['epoch_end'])
        totals[bucket] += len(real)
    if epoch != epochs or any(ids != lengths.keys() for ids in seen):
        raise ValueError('incomplete Stage2 optimizer exposures')
    return dict(totals)
