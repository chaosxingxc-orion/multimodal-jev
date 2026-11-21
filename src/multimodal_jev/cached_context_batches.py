"""Ordered whole-batch context preparation, with supervision bound on consume.

Caller validates full learning population and schedule once. Worker sees only
input asset metadata, never event labels; no policy or sampled action is cached.
"""
import time
import torch

from .context_candidate_task import compile_candidates,bind_candidate_targets
from .encoded_audio_cache import canonical
from .ordered_prefetch import OrderedPrefetch


class CachedContextBatches:
    def __init__(self,batches,population,task,preparer,*,start_update,stop_update,global_batch=128,
                 slot_request=None):
        if (type(start_update) is not int or type(stop_update) is not int or
                not 0<=start_update<stop_update<=len(batches) or
                type(global_batch) is not int or global_batch<1):
            raise ValueError('exact bounded schedule cursor and batch size required')
        self.batches=batches;self.population=population;self.preparer=preparer
        self.assets={k:{n:r[n] for n in ('audio','audio_sha256')} for k,r in population.items()}
        self.candidates=[dict(id=x['id'],definition=x['description']) for x in task['schema']['candidates']]
        self.requirements=task['thinking']['requirements'];self.global_batch=global_batch
        self.slot_request=slot_request
        self.cursor=start_update;self.stop=stop_update;self.last_timing=None
        self.queue=OrderedPrefetch(self._prepare,range(start_update,stop_update),workers=1,ahead=1)

    def _request(self,audio):
        return compile_candidates(audio,self.candidates,self.requirements,semantics='exclusive_event')

    def _slot_request(self,batch,slot):
        uid=slot['id'];audio=self.assets[uid]['audio']
        factory=getattr(self,'slot_request',None)
        return self._request(audio) if factory is None else factory(audio,uid,batch['epoch'])

    def _prepare(self,step):
        begin=time.monotonic();batch=self.batches[step]
        if len(batch['slots'])!=self.global_batch:raise ValueError('complete optimizer batch required')
        values=[]
        for slot in batch['slots']:
            asset=self.assets[slot['id']]
            values.append(self.preparer((self._slot_request(batch,slot),asset['audio_sha256'])))
        return step,values,dict(worker_seconds=time.monotonic()-begin,cache_hits=sum(x.cache_hit for x in values))

    def consume(self,step,policy_id):
        if type(step) is not int or step!=self.cursor or step>=self.stop:
            raise ValueError('consume exact next update once')
        if type(policy_id) is not str or not policy_id:raise ValueError('current policy identity required')
        start=time.monotonic();actual,values,timing=next(self.queue);wait=time.monotonic()-start
        if actual!=step:raise ValueError('prefetch order differs')
        begin=time.monotonic();batch=self.batches[step];items=[];metadata=[]
        for i,(slot,encoded) in enumerate(zip(batch['slots'],values,strict=True)):
            row=self.population[slot['id']];request=self._slot_request(batch,slot)
            if encoded.request!=canonical(request.model_input()):raise ValueError('prepared request differs')
            if (type(slot['padding']) is not bool or type(slot['weight']) not in (int,float) or
                    slot['weight']!=(0. if slot['padding'] else 1.)):
                raise ValueError('uniform real query and zero padding weights required')
            target=bind_candidate_targets(request,semantics='exclusive_event',exclusive_event=row['intent'])
            if not all(target['supervised']):raise ValueError('complete event labels required')
            uid=f'{row["id"]}::epoch{batch["epoch"]}-batch{batch["index"]}-slot{i}'
            items.append(dict(request_id=uid,policy_id=policy_id,inputs=encoded.inputs,positions=encoded.positions,
                labels=torch.tensor([target['labels']],dtype=torch.bool),observed=torch.tensor([target['supervised']],dtype=torch.bool),
                request_weight=slot['weight'],real_request=not slot['padding']))
            metadata.append(dict(id=row['id'],request_id=uid,target=target,weight=slot['weight'],
                input_tokens=encoded.inputs['input_ids'].shape[-1],audio_receipt=encoded.audio_receipt))
        self.cursor+=1
        self.last_timing=dict(timing,wait_seconds=wait,bind_seconds=time.monotonic()-begin,
            scope='Background service overlaps consumer; add foreground wait only to step time')
        return items,metadata

    def close(self):self.queue.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
