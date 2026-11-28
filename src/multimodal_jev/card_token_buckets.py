"""Online token-length batches. The cursor stores only not-yet-consumed anchor IDs."""

import torch
from torch.utils.data import BatchSampler


def bind_buckets(config):
    public = config['token_bucketing']
    window = config['window']
    if ('token_buckets' in window or not isinstance(public, dict)
            or set(public) != {'buckets', 'lookahead_batches'}):
        raise ValueError('declare token bucketing once at experiment level')
    rows = public['buckets']
    if (not isinstance(rows, list) or not rows or any(not isinstance(r, dict)
            or set(r) != {'max_tokens', 'batch_size_per_rank'} for r in rows)
            or rows[-1]['max_tokens'] is not None):
        raise ValueError('each bucket must pair a token limit with a per-rank width; final limit must be null')
    spec = dict(max_tokens=[r['max_tokens'] for r in rows[:-1]],
                max_anchors=[r['batch_size_per_rank'] for r in rows],
                lookahead_batches=public['lookahead_batches'])
    validate(spec)
    window['token_buckets'] = spec


def validate(spec):
    if not isinstance(spec, dict) or set(spec) != {'max_tokens', 'max_anchors', 'lookahead_batches'}:
        raise ValueError('expected token boundaries, per-rank widths and lookahead')
    bounds, caps = spec['max_tokens'], spec['max_anchors']
    if (not isinstance(bounds, list) or not isinstance(caps, list)
            or len(caps) != len(bounds)+1 or any(type(x) is not int or x <= 0 for x in [*bounds,*caps])
            or sorted(set(bounds)) != bounds or sorted(caps, reverse=True) != caps
            or type(spec['lookahead_batches']) is not int or spec['lookahead_batches'] < 1):
        raise ValueError('invalid token bucket configuration')


class BucketCursor:
    """All DP ranks deterministically select the same global batch before local rendering."""
    def __init__(self, epochs, spec, world, state=None):
        validate(spec)
        if not epochs or any(not ids or len(ids) != len(set(ids)) for ids in epochs) or world < 1:
            raise ValueError('complete nonempty epochs and positive DP width required')
        self.epochs, self.spec, self.world = epochs, spec, world
        self.bounds = torch.tensor(spec['max_tokens'], dtype=torch.long)
        self.limit = max(spec['max_anchors'])*world*spec['lookahead_batches']
        self.epoch, self.offset, self.pending = state if state is not None else (0, 0, [])
        self.pending = list(self.pending)
        if not (0 <= self.epoch <= len(epochs) and len(set(self.pending)) == len(self.pending)
                and len(self.pending) <= self.limit and
                (self.epoch == len(epochs) and self.offset == 0 and not self.pending or
                 self.epoch < len(epochs) and 0 <= self.offset <= len(epochs[self.epoch]) and
                 set(self.pending).issubset(epochs[self.epoch][:self.offset]))):
            raise ValueError('invalid bucket recovery cursor')

    def state(self):
        return self.epoch, self.offset, self.pending.copy()

    def take(self, measure):
        ids = self.epochs[self.epoch]
        stop = min(len(ids), self.offset+self.limit-len(self.pending))
        self.pending.extend(ids[self.offset:stop])
        self.offset = stop
        # Measure again after each new sampling snapshot. Never persist a length keyed by ID.
        lengths = [measure(uid) for uid in self.pending]
        bins = torch.bucketize(torch.tensor(lengths), self.bounds).tolist()
        bucket = bins[0]
        candidates = (i for i, value in enumerate(bins) if value == bucket)
        slots = next(iter(BatchSampler(candidates, self.spec['max_anchors'][bucket]*self.world, False)))
        batch = [self.pending[i] for i in slots]
        used = set(slots)
        self.pending = [uid for i,uid in enumerate(self.pending) if i not in used]
        epoch = self.epoch
        end = self.offset == len(ids) and not self.pending
        if end:
            self.epoch, self.offset = self.epoch+1, 0
        return batch, epoch, bucket, end, max(lengths[i] for i in slots)


class OnlineBuckets:
    """One lookahead thread plans with a bounded-lag EMA snapshot; only local audio is read."""
    def __init__(self, stream, ema, first, stop, pending=None, state=None):
        from .ordered_prefetch import OrderedPrefetch
        epochs = [[] for _ in range(stream.config['batch_config']['epochs'])]
        for b in stream.batches:
            epochs[b['epoch']].extend(s['id'] for s in b['slots'] if s['weight'])
        pressure = stream.config.get('finite_pressure')
        self.fixed = ema is None
        if self.fixed:
            epoch = stream.config['instance_epoch']
            epochs = [epochs[epoch]]
        if pressure:
            # First engineering update must actually exercise the registered long anchor.
            epochs[0].remove(pressure['anchor'])
            epochs[0].insert(0, pressure['anchor'])
        self.stream = stream
        self.cursor = BucketCursor(epochs, stream.config['token_buckets'], stream.world, state)
        self.lengths = [len(p)+len(s) for p,s in stream.templates]
        self.source = (None, None) if self.fixed else (ema.updates, ema.distribution())
        def items():
            for step in range(first, stop):
                if self.cursor.epoch == len(epochs):
                    return
                yield step, *(pending if step == first and pending is not None else self.source)
        if stream.config['triplet_prefetch']['ahead'] != 1:
            raise ValueError('bucket scheduling requires one pending update')
        self.queue = OrderedPrefetch(self.prepare, items(), **stream.config['triplet_prefetch'])

    def prepare(self, item):
        step, version, distribution = item
        stream = self.stream
        if not self.fixed and not 0 <= step-version <= stream.max_lag:
            raise ValueError('bucket draw exceeds EMA staleness limit')
        epoch_offset = stream.config['instance_epoch'] if self.fixed else 0
        epoch = self.cursor.epoch + epoch_offset
        plans, lengths = {}, {}
        def measure(uid):
            pairs, coords = stream.group_plan(uid, epoch, step, distribution)
            plans[uid] = pairs, coords
            lengths[uid] = max(stream.packed.entries[audio][4] + self.lengths[
                stream.spec['arms'][stream.card_arm(pairs,card,epoch)][stream.label_index[label]]]
                for audio,label,card in pairs)
            return lengths[uid]
        ids, local_epoch, bucket, end, maximum = self.cursor.take(measure)
        epoch = local_epoch + epoch_offset
        count = len(ids)
        # Only pad to a multiple of DP. Padding never enters the EMA or loss denominator.
        slots = [dict(id=uid, weight=1) for uid in ids]
        slots.extend(dict(id=ids[0], weight=0) for _ in range((-count) % stream.world))
        groups = stream.render_slots(step, epoch, slots[stream.rank::stream.world], distribution, plans)
        actual = [max(row['inputs']['input_ids'].shape[-1] for row in rows) for rows,*_ in groups]
        if actual != [lengths[s['id']] for s in slots[stream.rank::stream.world]]:
            raise ValueError('rendered and planned group lengths differ')
        local_max = max(actual)
        return step, groups, count, version, dict(epoch=epoch, end=end, bucket=bucket, max_tokens=maximum,
            cursor=self.cursor.state(), local_max=local_max)
