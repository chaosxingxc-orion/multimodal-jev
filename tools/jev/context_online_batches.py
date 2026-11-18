"""Online Stage2 batch assembly, independent of the frozen Stage1 implementation.

Engine owners authenticate joint-policy assets before calling this helper.
A fresh current-policy forward is required for every consumed query batch.
Only fixed-reference distributions may be reused across updates; no labels or
sampled actions are sent to either engine. This is not a continuous RL trainer.
"""
import copy
import hashlib
import math
import json

import torch

from multimodal_jev.context_rollout_batch import bind_rollout
from multimodal_jev.span_request_identity import tensor_identity


def auxiliary_coefficient(update, total_updates=634):
    """Zero-based update: .3 -> .1 over the first 80% of the schedule."""
    if (type(update) is not int or type(total_updates) is not int or
            total_updates<2 or not 0<=update<total_updates):
        raise ValueError('bounded zero-based stage update required')
    end=max(1,math.ceil(.8*total_updates)-1)
    return .1+.1*(1+math.cos(math.pi*min(update/end,1.)))


def _seed(update, request_id, base_seed):
    data=repr((base_seed,update,request_id)).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8],'big') % (2**63)


class OnlineContextBatches:
    """Fail-closed assembly and explicit acknowledgement of each actor update.

    Caller must prove export/activation correctness separately. Acknowledging
    an update never promotes a policy or verifies its weights. If assembly or
    actor execution fails, create a new session from its committed checkpoint;
    do not retry a partly consumed input stream in this object.
    """
    def __init__(self, *, reference_engine, reference_policy_id, audio_token_id,
                 algorithm='grpo_v1', start_update=0, total_updates=634,
                 group_size=16, base_seed=317):
        auxiliary_coefficient(start_update,total_updates)
        if (algorithm not in ('grpo_v1','rloo_v1') or type(group_size) is not int or group_size<2 or
                type(base_seed) is not int or base_seed<0 or type(audio_token_id) is not int or
                audio_token_id<0 or not isinstance(reference_policy_id,str) or not reference_policy_id):
            raise ValueError('explicit algorithm, reference and sampling configuration required')
        if reference_engine.bundle['policy_id']!=reference_policy_id:
            raise ValueError('fixed reference engine policy differs')
        self.reference_engine=reference_engine
        self.reference_policy_id=reference_policy_id
        self.audio_token_id=audio_token_id
        self.algorithm=algorithm
        self.cursor=start_update
        self.total_updates=total_updates
        self.group_size=group_size
        self.base_seed=base_seed
        self.phase='ready'
        self._reference={}
        self._pending_policy=None

    def _reference_record(self, item):
        # Bind to the entire real input representation; identical waveform with
        # different candidates/questions must not hit the same reference entry.
        identity=dict(policy_id=self.reference_policy_id,positions=item['positions'][0].tolist(),
                      inputs={k:tensor_identity(v) for k,v in sorted(item['inputs'].items())})
        key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if key in self._reference:return copy.deepcopy(self._reference[key]),True
        record=self.reference_engine.score(item['inputs'],item['positions'],audio_token_id=self.audio_token_id)
        bind_rollout(record,item['inputs'],item['positions'],policy_id=self.reference_policy_id,
                     group_size=2,seed=0)
        self._reference[key]=copy.deepcopy(record)
        return record,False

    def prepare(self, items, current_engine, *, policy_id, update):
        if self.phase!='ready' or update!=self.cursor or not 0<=update<self.total_updates:
            raise ValueError('exact next update in ready session required')
        self.phase='preparing'
        try:
            if self.reference_engine.bundle['policy_id']!=self.reference_policy_id:
                raise ValueError('reference engine changed during task stage')
            if not items or current_engine.bundle['policy_id']!=policy_id:
                raise ValueError('nonempty batch and current engine policy required')
            ids=[x['request_id'] for x in items]
            if len(set(ids))!=len(ids):raise ValueError('distinct scheduled query slots required')
            if self._pending_policy is not None and self._pending_policy!=policy_id:
                raise ValueError('engine activation differs from acknowledged update')
            prepared=[];receipts=[]
            for item in items:
                if item['policy_id']!=policy_id:raise ValueError('input bound to stale behavior policy')
                if (type(item['real_request']) is not bool or item['request_weight']!=(1. if item['real_request'] else 0.) or
                        item['labels'].dtype!=torch.bool or item['observed'].dtype!=torch.bool or
                        item['labels'].shape!=item['positions'].shape or item['observed'].shape!=item['labels'].shape or
                        not item['observed'].all()):
                    raise ValueError('complete binary labels, uniform real queries and zero padding required')
                reference,hit=self._reference_record(item)
                # A current distribution is never looked up in a cross-update
                # cache. At task step0 the caller may share one fixed engine;
                # correctness does not depend on this optional identity.
                record=(reference if current_engine is self.reference_engine and not hit else
                        current_engine.score(item['inputs'],item['positions'],audio_token_id=self.audio_token_id))
                bound=bind_rollout(record,item['inputs'],item['positions'],policy_id=policy_id,
                    group_size=self.group_size,seed=_seed(update,item['request_id'],self.base_seed))
                ref=bind_rollout(reference,item['inputs'],item['positions'],policy_id=self.reference_policy_id,
                                 group_size=2,seed=0)
                value=dict(item,behavior=bound['behavior'],actions=bound['actions'],
                    reference_policy=ref['behavior'],reference_policy_id=self.reference_policy_id)
                prepared.append(value)
                receipts.append(dict(request_id=item['request_id'],update=update,rollout=bound['receipt'],
                    behavior_record=copy.deepcopy(record),reference_record=copy.deepcopy(reference),
                    reference_policy_id=self.reference_policy_id,reference_request_id=reference['request_id'],
                    reference_cache_hit=hit))
            if not any(x['real_request'] for x in prepared):raise ValueError('nonempty real population required')
            coefficient=auxiliary_coefficient(update,self.total_updates)
            options=dict(stage='task',ranking_coefficient=coefficient,probability_coefficient=coefficient,
                         algorithm=self.algorithm,kl_coefficient=.01,reference_policy_id=self.reference_policy_id)
            self.phase='awaiting_update'
            return prepared,options,receipts
        except BaseException:
            self.phase='failed'
            raise

    def acknowledge_update(self, *, update, next_policy_id):
        if (self.phase!='awaiting_update' or update!=self.cursor or
                not isinstance(next_policy_id,str) or not next_policy_id):
            raise ValueError('one completed actor update and explicit exported next policy required')
        # A zero-gradient update can legitimately keep the same policy ID.
        self._pending_policy=next_policy_id
        self.cursor+=1
        self.phase='complete' if self.cursor==self.total_updates else 'ready'

    def fail_update(self):
        self.phase='failed'
