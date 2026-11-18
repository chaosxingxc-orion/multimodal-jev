"""One CPU batch ahead; bind current policy and supervision on consumption.

The caller authenticates the population and full schedule once. Worker closures
never see the model, optimizer or policy. Only direct/native Stage1 is supported;
this is not an RL trajectory cache. The old trainer remains unchanged.
"""
import time

from .cached_span_preparation import bind_instruction_slot
from .ordered_prefetch import OrderedPrefetch
from .slurp_intent_structure import compile_intent
from .slurp_official_training import example
from .slurp_training import content_hash


class CachedInstructionBatches:
    def __init__(self, batches, population, task, preparer, *, start_update, stop_update,
                 context_limit, global_batch=128):
        if (type(start_update) is not int or type(stop_update) is not int or
                not 0 <= start_update < stop_update <= len(batches)):
            raise ValueError('bounded exact schedule cursor required')
        if (type(context_limit) is not int or context_limit < 1 or
                type(global_batch) is not int or global_batch < 1):
            raise ValueError('positive context and batch limits required')
        self.batches, self.population, self.task = batches, population, task
        self.preparer = preparer
        self.context_limit, self.global_batch = context_limit, global_batch
        self.cursor, self.stop = start_update, stop_update
        self.last_timing = None
        self.queue = OrderedPrefetch(self._prepare, range(start_update, stop_update), workers=1, ahead=1)

    def _prepare(self, step):
        started = time.monotonic()
        epoch, batch_index, slots = self.batches[step]
        if len(slots) != self.global_batch:
            raise ValueError('complete scheduled batch including tail padding required')
        prepared = []
        for slot in slots:
            if (slot['view'], slot['variant']) != ('direct', 'native'):
                raise ValueError('only direct/native input cache admitted')
            row = self.population[slot['id']]
            # Input compilation only: do not access semantic targets in worker.
            request = compile_intent(row['audio'], self.task, 'D')
            item = self.preparer((request, row['audio_sha256']))
            if item.inputs['input_ids'].shape[-1] > self.context_limit:
                raise ValueError('untruncated input exceeds context limit')
            prepared.append(item)
        return step, epoch, batch_index, prepared, dict(
            worker_seconds=time.monotonic()-started,
            cache_hits=sum(item.cache_hit for item in prepared), slots=len(prepared))

    def consume(self, step, policy_id):
        if type(step) is not int or step != self.cursor or step >= self.stop:
            raise ValueError('consume exact next update once')
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError('current policy identity required')
        wait_started = time.monotonic()
        actual, epoch, batch_index, prepared, timing = next(self.queue)
        wait_seconds = time.monotonic()-wait_started
        bind_started = time.monotonic()
        if actual != step:
            raise ValueError('prefetch changed schedule order')
        slots = self.batches[step][2]
        items, metadata = [], []
        for index, (slot, encoded) in enumerate(zip(slots, prepared, strict=True)):
            row = self.population[slot['id']]
            request, target, _ = example(row, self.task, slot['view'], slot['variant'])
            item = bind_instruction_slot(encoded, request, target, slot, epoch=epoch,
                                         batch_index=batch_index, index=index, policy_id=policy_id)
            items.append(item)
            metadata.append(dict(id=row['id'], request_id=item['request_id'], view=slot['view'],
                variant=slot['variant'], weight=slot['weight'],
                input_tokens=encoded.inputs['input_ids'].shape[-1],
                input_sha256=content_hash(request.model_input()), audio_receipt=encoded.audio_receipt,
                targets=target['targets'], span_weights=target['span_weights']))
        self.cursor += 1
        self.last_timing = dict(timing,update_index=step,wait_seconds=wait_seconds,
            bind_seconds=time.monotonic()-bind_started,
            scope='Worker service may overlap prior actor update; do not add it to foreground wait')
        return items, metadata

    def close(self):
        self.queue.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
