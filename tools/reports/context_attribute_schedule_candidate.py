"""Adapt the frozen prefetch and resident session to native attribute tasks.

No trainer, checkpoint or engine implementation is copied. Caller authenticates
the full input/target manifests and schedule before constructing these adapters.
"""
import copy
from pathlib import Path
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.cached_context_batches import CachedContextBatches
from multimodal_jev.ordered_prefetch import OrderedPrefetch
from context_rollout_session import ContextRolloutSession
from context_reference_cache import CachedReferenceEngine
from tools.reports.context_attribute_input_candidate import AttributeOnlineBatches
from tools.reports.context_attribute_reward_candidate import validate_truth
from tools.reports.build_context_attribute_candidate import request, digest

BATCH_SHA = 'cfbae6a97482310f8bded2b927d8daef386d01b9829f3860e31db125a4a347db'
SESSION_SHA = 'b8fd074165ef12fe8b2fd9dce5c29f8ec417e29399077985516643f7e8e307c8'


def authenticate(source_root):
    for name, expected in [('src/multimodal_jev/cached_context_batches.py', BATCH_SHA),
                           ('tools/jev/context_rollout_session.py', SESSION_SHA)]:
        if file_hash(Path(source_root)/name) != expected:
            raise ValueError('Frozen scheduling/session dependency changed: '+name)


class AttributeScheduledBatches(CachedContextBatches):
    def __init__(self, batches, inputs, targets, preparer, *, source_root,
                 start_update, stop_update, global_batch=128):
        authenticate(source_root)
        if (type(start_update) is not int or type(stop_update) is not int
                or not 0 <= start_update < stop_update <= len(batches) or global_batch != 128
                or len(inputs) != len(targets) or not inputs
                or len({r['id'] for r in inputs}) != len(inputs)
                or len({r['id'] for r in targets}) != len(targets)):
            raise ValueError('Exact population, bounded cursor and global128 required')
        self.assets = {r['id']:copy.deepcopy(r) for r in inputs}
        self.targets = {r['id']:copy.deepcopy(r) for r in targets}
        if self.assets.keys() != self.targets.keys():raise ValueError('Target population differs')
        for uid, asset in self.assets.items():
            if (set(asset) != {'id','audio','audio_sha256','catalog_sha256'}
                    or asset['catalog_sha256'] != preparer.contract['catalog_sha256']
                    or self.targets[uid]['input_sha256'] != digest(asset)):
                raise ValueError('Input/target/catalog identity differs')
        self.batches = copy.deepcopy(batches)
        self.preparer = preparer; self.global_batch = global_batch
        self.cursor = start_update; self.stop = stop_update; self.last_timing = None
        for batch in self.batches[start_update:stop_update]:
            if len(batch['slots']) != global_batch:raise ValueError('Complete global128 schedule required')
            for slot in batch['slots']:
                if (slot['id'] not in self.assets or type(slot['padding']) is not bool
                        or type(slot['weight']) not in (int,float)
                        or slot['weight'] != (0. if slot['padding'] else 1.)):
                    raise ValueError('Known uniform-real/zero-padding slot required')
        # Inherited _prepare sees only input assets; target binding happens below.
        self.queue = OrderedPrefetch(self._prepare, range(start_update,stop_update), workers=1,ahead=1)

    def _request(self, audio):
        return request(audio,self.preparer.menu,arm=self.preparer.arm)

    def consume(self, step, policy_id):
        if type(step) is not int or step != self.cursor or step >= self.stop:
            raise ValueError('Consume exact next update once')
        if not isinstance(policy_id,str) or not policy_id:raise ValueError('Explicit current policy required')
        started=time.monotonic(); actual,values,timing=next(self.queue);wait=time.monotonic()-started
        if actual != step:raise ValueError('Prefetch order differs')
        started=time.monotonic();batch=self.batches[step];items=[];metadata=[]
        for i,(slot,encoded) in enumerate(zip(batch['slots'],values,strict=True)):
            asset=self.assets[slot['id']];target=self.targets[slot['id']]
            uid=f'{asset["id"]}::epoch{batch["epoch"]}-batch{batch["index"]}-slot{i}'
            item=self.preparer.actor_item(asset,target,encoded,policy_id=policy_id,slot_id=uid,
                                          padding=slot['padding'])
            items.append(item)
            metadata.append(dict(id=asset['id'],request_id=uid,target=copy.deepcopy(target),
                weight=slot['weight'],input_tokens=encoded.inputs['input_ids'].shape[-1],
                audio_receipt=encoded.audio_receipt,attribute_contract_id=self.preparer.contract['contract_id']))
        self.cursor+=1
        self.last_timing=dict(timing,wait_seconds=wait,bind_seconds=time.monotonic()-started,
            scope='Background work overlaps consumption; only foreground wait adds to step time')
        return items,metadata


class AttributeRolloutSession(ContextRolloutSession):
    def __init__(self, engine, reference_cache, *, input_adapter, source_root, **kwargs):
        authenticate(source_root)
        super().__init__(engine,reference_cache,**kwargs)
        self.adapter=input_adapter
        self.groups=AttributeOnlineBatches(input_adapter,
            reference_engine=CachedReferenceEngine(reference_cache),
            reference_policy_id=self.reference['policy_id'],audio_token_id=self.audio_token_id,
            algorithm=kwargs.get('algorithm','grpo_v1'),start_update=kwargs.get('start_update',0),
            total_updates=kwargs.get('total_updates',634))

    def prepare(self, items, *, update):
        # Reject wrong-arm or unscorable truth before even computing reference.
        try:
            c=self.adapter.contract
            for row in items:
                if (row.get('attribute_contract_id') != c['contract_id']
                        or row.get('decision_ids') != c['decision_ids']):
                    raise ValueError('Attribute contract or decision order differs')
                validate_truth(row['labels'],row['observed'],self.adapter.menu,self.adapter.arm)
        except BaseException:
            self.fail_actor_update()
            raise
        return super().prepare(items,update=update)
