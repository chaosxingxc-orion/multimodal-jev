"""Bulk dual-credit assembly on the existing session/cache lifecycle.

Caller authenticates actual engine/runtime and task candidate ordering. No
trainer, optimizer, daemon or dispatch here. Shared fixed reference entries
are isolated by execution/processor/source identity; behavior is always fresh.
"""
import copy
import time

import torch

from context_online_batches import OnlineContextBatches
from context_reference_cache import ReferenceCache, CachedReferenceEngine
from context_rollout_session import ContextRolloutSession
from multimodal_jev.context_dual_credit import ALGORITHMS, ARMS, prepare as prepare_credit
from multimodal_jev.context_logits_wire import WIRE, validate_logits_record
from multimodal_jev.context_rollout_batch import bind_rollout


def bind_candidates(items,metadata,candidate_ids):
    """Bind the existing loader's independently compiled target order.

    The loader checks its cached encoded request against model_input() before
    building this metadata. This adapter never chooses questions from labels.
    """
    if len(items)!=len(metadata):raise ValueError('complete input metadata required')
    result=[]
    for item,meta in zip(items,metadata,strict=True):
        target=meta['target']
        if (meta['request_id']!=item['request_id'] or meta['weight']!=item['request_weight']
                or target['semantics']!='exclusive_event'
                or target['candidate_ids']!=list(candidate_ids)
                or target['labels']!=item['labels'][0].tolist()
                or target['supervised']!=item['observed'][0].tolist()):
            raise ValueError('compiled candidate order or target binding differs')
        result.append(dict(item,candidate_ids=list(candidate_ids)))
    return result


class RuntimeReferenceCache(ReferenceCache):
    """Reuse atomic fixed-reference storage with explicit execution isolation."""
    def __init__(self,root,*,policy_id,base_id,contract,runtime_identity):
        keys={'image_sha256','source_manifest_sha256','processor_sha256','profile','output_protocol'}
        if (set(runtime_identity)!=keys or runtime_identity['output_protocol']!=WIRE
                or not isinstance(runtime_identity['profile'],str) or not runtime_identity['profile']):
            raise ValueError('complete actual runtime/processor identity required')
        for key in ('image_sha256','source_manifest_sha256','processor_sha256'):
            value=runtime_identity[key]
            if type(value)!=str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('runtime SHA256 required')
        from pathlib import Path
        self.runtime_identity=copy.deepcopy(runtime_identity)
        super().__init__(Path(root)/self.digest(runtime_identity),policy_id=policy_id,
                         base_id=base_id,contract=copy.deepcopy(contract))
        # Include scope in each payload as well as the enclosing namespace.
        self.identity.update(runtime_identity=self.runtime_identity)
        self.namespace=self.root/self.digest(self.identity)
        self.namespace.mkdir(exist_ok=True)


def probability_record(record):
    validate_logits_record(record)
    return {k:copy.deepcopy(record[k]) for k in ('request_id','request_identity','prob','policy_id')}


class DualContextGroups(OnlineContextBatches):
    """Reuse cursor/acknowledgement; replace joint-action assembly with dual groups."""
    def __init__(self,*,candidate_ids,arm='joint',algorithm=ALGORITHMS[0],**kwargs):
        if (algorithm not in ALGORITHMS or arm not in ARMS or not candidate_ids
                or len(set(candidate_ids))!=len(candidate_ids)
                or any(type(v)!=str or not v for v in candidate_ids)):
            raise ValueError('explicit dual algorithm/arm and complete ordered candidates required')
        super().__init__(algorithm='rloo_v1',**kwargs)
        self.algorithm=algorithm;self.arm=arm;self.candidate_ids=tuple(candidate_ids)

    def validate_items(self,items,policy_id,update):
        if self.phase!='ready' or update!=self.cursor or not 0<=update<self.total_updates:
            raise ValueError('exact next ready update required')
        if not items or self._pending_policy not in (None,policy_id):
            raise ValueError('nonempty batch under acknowledged policy required')
        ids=[x['request_id'] for x in items]
        if any(type(x)!=str or not x for x in ids) or len(set(ids))!=len(ids):
            raise ValueError('distinct scheduled query slots required')
        for item in items:
            n=len(self.candidate_ids)
            if (item['policy_id']!=policy_id or tuple(item['candidate_ids'])!=self.candidate_ids
                    or item['labels'].dtype!=torch.bool or item['observed'].dtype!=torch.bool
                    or item['labels'].device.type!='cpu' or item['observed'].device.type!='cpu'
                    or item['labels'].shape!=(1,n) or item['observed'].shape!=(1,n)
                    or not item['observed'].all() or item['positions'].shape!=(1,n)
                    or item['positions'].dtype!=torch.long
                    or type(item['real_request'])!=bool
                    or item['request_weight']!=(1. if item['real_request'] else 0.)):
                raise ValueError('current policy, candidate order, complete truth and zero padding required')
        if not any(i['real_request'] for i in items):raise ValueError('nonempty real population required')

    def prepare(self,items,current_engine,*,policy_id,update):
        try:
            self.validate_items(items,policy_id,update);self.phase='preparing'
            if (current_engine.bundle['policy_id']!=policy_id
                    or self.reference_engine.bundle['policy_id']!=self.reference_policy_id):
                raise ValueError('actual engine policy differs')
            # Exactly one bulk transport call; service controls actual model
            # batch capacity, length grouping and duplicate-input restoration.
            rows=[{k:i[k] for k in ('inputs','positions')} for i in items]
            records=current_engine.score_many(rows,audio_token_id=self.audio_token_id)
            if len(records)!=len(items) or current_engine.bundle['policy_id']!=policy_id:
                raise ValueError('incomplete or switched behavior batch')
            prepared=[];receipts=[]
            for item,record in zip(items,records,strict=True):
                validate_logits_record(record)
                bind_rollout(record,item['inputs'],item['positions'],policy_id=policy_id,group_size=2,seed=0)
                reference,hit=self._reference_record(item)
                identity=dict(policy_id=policy_id,input_id=record['request_id'],query_id=item['request_id'],update=update)
                credit=prepare_credit(torch.tensor(record['logits'],dtype=torch.float32),
                    torch.tensor(record['prob'],dtype=torch.float32),self.candidate_ids,item['labels'][0],
                    **identity,algorithm=self.algorithm,group_size=self.group_size,seed=self.base_seed)
                prepared.append(dict(item,dual_credit=credit,dual_identity=identity,
                    reference_probability=torch.tensor(reference['prob'],dtype=torch.float32),
                    reference_policy_id=self.reference_policy_id))
                receipts.append(dict(request_id=item['request_id'],behavior_record=copy.deepcopy(record),
                    reference_record=copy.deepcopy(reference),credit_binding_sha256=credit['binding_sha256'],
                    reference_memory_hit=hit))
            self.phase='awaiting_update'
            return prepared,dict(algorithm=self.algorithm,arm=self.arm,ranking_coefficient=.1,
                probability_coefficient=.3,kl_coefficient=.01,reference_policy_id=self.reference_policy_id),receipts
        except BaseException:
            self.phase='failed';raise


class DualContextRolloutSession(ContextRolloutSession):
    """Existing activation/ack/failure state machine plus batched reference fill."""
    def __init__(self,engine,reference_cache,*,runtime_identity,candidate_ids,arm='joint',
                 algorithm=ALGORITHMS[0],**kwargs):
        if (not isinstance(reference_cache,RuntimeReferenceCache)
                or reference_cache.runtime_identity!=runtime_identity
                or getattr(engine,'runtime_identity',None)!=runtime_identity):
            raise ValueError('authenticated matching engine/cache runtime required')
        super().__init__(engine,reference_cache,algorithm='rloo_v1',**kwargs)
        self.runtime_identity=copy.deepcopy(runtime_identity)
        self.groups=DualContextGroups(reference_engine=CachedReferenceEngine(reference_cache),
            reference_policy_id=self.reference['policy_id'],audio_token_id=self.audio_token_id,
            candidate_ids=candidate_ids,arm=arm,algorithm=algorithm,
            start_update=kwargs.get('start_update',0),total_updates=kwargs.get('total_updates',634))

    def prepare(self,items,*,update):
        with self._lock:
            if self.phase!='ready':raise ValueError('ready rollout session required')
            self.phase='preparing';started=time.monotonic()
            try:
                if self.engine.runtime_identity!=self.runtime_identity:
                    raise ValueError('engine runtime changed')
                self.groups.validate_items(items,self.current['policy_id'],update)
                missing=[];seen=set()
                for item in items:
                    key=self.cache.digest(self.cache.binding(item['inputs'],item['positions']))
                    if key not in seen and self.cache.read(item['inputs'],item['positions']) is None:
                        missing.append(item)
                    seen.add(key)
                switches=[];reference_started=time.monotonic()
                if missing:
                    receipt=self._activate(self.reference)
                    if receipt is not None:switches.append(receipt)
                    rows=[{k:i[k] for k in ('inputs','positions')} for i in missing]
                    records=self.engine.score_many(rows,audio_token_id=self.audio_token_id)
                    if len(records)!=len(missing) or self.engine.bundle['policy_id']!=self.reference['policy_id']:
                        raise ValueError('incomplete or switched fixed reference batch')
                    for item,record in zip(missing,records,strict=True):
                        self.cache.write(probability_record(record),item['inputs'],item['positions'])
                reference_seconds=time.monotonic()-reference_started
                receipt=self._activate(self.current)
                if receipt is not None:switches.append(receipt)
                behavior_started=time.monotonic()
                prepared,options,receipts=self.groups.prepare(items,self.engine,
                    policy_id=self.current['policy_id'],update=update)
                behavior_seconds=time.monotonic()-behavior_started
                self.phase='awaiting_actor'
                return prepared,options,dict(update=update,behavior_policy_id=self.current['policy_id'],
                    fixed_reference_policy_id=self.reference['policy_id'],runtime_identity=self.runtime_identity,
                    reference_misses=len(missing),reference_seconds=reference_seconds,
                    behavior_and_credit_seconds=behavior_seconds,total_seconds=time.monotonic()-started,
                    bulk_reference_calls=int(bool(missing)),bulk_behavior_calls=1,switches=switches,queries=receipts)
            except BaseException:
                self.phase='failed';self.groups.fail_update();raise
