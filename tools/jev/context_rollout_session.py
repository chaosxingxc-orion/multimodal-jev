"""One resident engine serves fresh behavior and immutable reference per batch.

Reference misses are filled under the fixed policy, then the same engine is
switched back to the current actor policy. The actor and optimizer stay outside
this class. This is a rollout session, not a training controller or admission.
"""
import copy
import threading
import time

from context_online_batches import OnlineContextBatches
from context_reference_cache import CachedReferenceEngine


class ContextRolloutSession:
    def __init__(self,engine,reference_cache,*,reference,initial_policy,audio_token_id,
                 algorithm='grpo_v1',start_update=0,total_updates=634):
        if (reference_cache.policy_id!=reference['policy_id'] or
                reference_cache.identity['contract']!=engine.bundle['contract'] or
                engine.bundle['policy_id']!=initial_policy['policy_id']):
            raise ValueError('authenticated initial engine and fixed reference contract required')
        for spec in (reference,initial_policy):self._validate_spec(spec)
        self.engine=engine;self.cache=reference_cache
        self.reference=copy.deepcopy(reference);self.current=copy.deepcopy(initial_policy)
        self.audio_token_id=audio_token_id
        self._lock=threading.Lock();self.phase='ready'
        self.groups=OnlineContextBatches(reference_engine=CachedReferenceEngine(reference_cache),
            reference_policy_id=reference['policy_id'],audio_token_id=audio_token_id,
            algorithm=algorithm,start_update=start_update,total_updates=total_updates)

    @staticmethod
    def _validate_spec(spec):
        if (set(spec)!={'snapshot','policy_id','adapter'} or
                any(not isinstance(v,str) or not v for v in spec.values())):
            raise ValueError('complete policy snapshot/id/native adapter paths required')

    def _activate(self,spec):
        if self.engine.bundle['policy_id']==spec['policy_id']:return None
        return self.engine.switch(**spec)

    def prepare(self,items,*,update):
        with self._lock:
            if self.phase!='ready':raise ValueError('ready rollout session required')
            self.phase='preparing';started=time.monotonic()
            try:
                if update!=self.groups.cursor:raise ValueError('exact next schedule update required')
                if not items or any(x['policy_id']!=self.current['policy_id'] for x in items):
                    raise ValueError('all queries must bind current actor policy')
                # Determine all misses before any model switch. Do not switch
                # back and forth once per query; long batches cost two switches.
                missing=[];seen=set()
                for item in items:
                    identity=self.cache.binding(item['inputs'],item['positions']);key=self.cache.digest(identity)
                    if key not in seen and self.cache.read(item['inputs'],item['positions']) is None:
                        missing.append(item)
                    seen.add(key)
                switches=[];reference_started=time.monotonic()
                if missing:
                    receipt=self._activate(self.reference)
                    if receipt is not None:switches.append(receipt)
                    for item in missing:
                        record=self.engine.score(item['inputs'],item['positions'],audio_token_id=self.audio_token_id)
                        self.cache.write(record,item['inputs'],item['positions'])
                reference_seconds=time.monotonic()-reference_started
                receipt=self._activate(self.current)
                if receipt is not None:switches.append(receipt)
                prepared,options,receipts=self.groups.prepare(items,self.engine,
                    policy_id=self.current['policy_id'],update=update)
                self.phase='awaiting_actor'
                return prepared,options,dict(update=update,behavior_policy_id=self.current['policy_id'],
                    fixed_reference_policy_id=self.reference['policy_id'],reference_misses=len(missing),
                    reference_seconds=reference_seconds,total_seconds=time.monotonic()-started,
                    switches=switches,queries=receipts)
            except BaseException:
                self.phase='failed';self.groups.fail_update();raise

    def accept_actor_update(self,*,update,next_policy):
        """Caller commits/authenticates actor FP32/Adam state before invoking.

        Real engine.switch validates policy/adapter assets; this method cannot
        prove the exported weights equal the live actor's parameters.
        """
        with self._lock:
            if self.phase!='awaiting_actor':raise ValueError('one pending actor update required')
            self.phase='activating'
            try:
                self._validate_spec(next_policy)
                if update!=self.groups.cursor:raise ValueError('actor update cursor differs')
                receipt=self._activate(next_policy)
                if self.engine.bundle['policy_id']!=next_policy['policy_id']:
                    raise ValueError('engine did not activate exported actor policy')
                self.groups.acknowledge_update(update=update,next_policy_id=next_policy['policy_id'])
                self.current=copy.deepcopy(next_policy)
                self.phase='complete' if self.groups.phase=='complete' else 'ready'
                return receipt
            except BaseException:
                self.phase='failed';self.groups.fail_update();raise

    def fail_actor_update(self):
        with self._lock:self.phase='failed';self.groups.fail_update()
