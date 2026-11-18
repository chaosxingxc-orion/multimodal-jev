"""Independent bulk transport session; frozen online sampling remains unchanged."""
import time
import torch
from tools.jev.context_rollout_session import ContextRolloutSession
from tools.reports.context_bulk_rpc_candidate import validate_rows
from multimodal_jev.context_rollout_batch import bind_rollout


class OrderedRecords:
    """One-batch consumable records, never a cross-update behavior cache."""
    def __init__(self,records,policy_id):
        self.records=records;self.bundle={'policy_id':policy_id};self.cursor=0

    def score(self,inputs,positions,*,audio_token_id):
        if self.cursor>=len(self.records):raise ValueError('Batch records exhausted')
        result=self.records[self.cursor]
        bind_rollout(result,inputs,positions,policy_id=self.bundle['policy_id'],group_size=2,seed=0)
        self.cursor+=1
        return result


def validate_items(items,policy_id):
    if not items or len(items)>128 or len({x['request_id'] for x in items})!=len(items):
        raise ValueError('Distinct bounded scheduled slots required')
    for item in items:
        if (item['policy_id']!=policy_id or type(item['real_request']) is not bool
                or item['request_weight']!=(1. if item['real_request'] else 0.)
                or item['labels'].dtype!=torch.bool or item['observed'].dtype!=torch.bool
                or item['labels'].shape!=item['positions'].shape
                or item['observed'].shape!=item['labels'].shape or not item['observed'].all()):
            raise ValueError('Current policy, complete binary truth, real weights and zero padding required')
    if not any(x['real_request'] for x in items):raise ValueError('At least one real query required')
    return validate_rows([dict(inputs=x['inputs'],positions=x['positions']) for x in items])


class BulkContextRolloutSession(ContextRolloutSession):
    def prepare(self,items,*,update):
        with self._lock:
            if self.phase!='ready':raise ValueError('Ready bulk session required')
            self.phase='preparing';started=time.monotonic()
            try:
                if update!=self.groups.cursor:raise ValueError('Exact next schedule update required')
                rows=validate_items(items,self.current['policy_id'])
                missing=[];seen=set()
                for row in rows:
                    key=self.cache.digest(self.cache.binding(**row))
                    if key not in seen and self.cache.read(**row) is None:missing.append(row)
                    seen.add(key)
                switches=[];ref_started=time.monotonic()
                if missing:
                    receipt=self._activate(self.reference)
                    if receipt is not None:switches.append(receipt)
                    records=self.engine.score_many(missing,audio_token_id=self.audio_token_id)
                    if len(records)!=len(missing):raise ValueError('Incomplete reference batch')
                    # Validate the entire result before publishing any cache entries.
                    for row,record in zip(missing,records):
                        bind_rollout(record,**row,policy_id=self.reference['policy_id'],group_size=2,seed=0)
                    for row,record in zip(missing,records):self.cache.write(record,**row)
                reference_seconds=time.monotonic()-ref_started
                receipt=self._activate(self.current)
                if receipt is not None:switches.append(receipt)
                records=self.engine.score_many(rows,audio_token_id=self.audio_token_id)
                if len(records)!=len(rows):raise ValueError('Incomplete behavior batch')
                ordered=OrderedRecords(records,self.current['policy_id'])
                prepared,options,receipts=self.groups.prepare(items,ordered,policy_id=self.current['policy_id'],update=update)
                if ordered.cursor!=len(rows):raise ValueError('Unconsumed behavior records')
                self.phase='awaiting_actor'
                return prepared,options,dict(update=update,behavior_policy_id=self.current['policy_id'],
                    fixed_reference_policy_id=self.reference['policy_id'],reference_misses=len(missing),
                    reference_seconds=reference_seconds,total_seconds=time.monotonic()-started,
                    switches=switches,queries=receipts,transport='bulk_rpc_sequential_forward_candidate_v1',
                    score_transactions=1+bool(missing))
            except BaseException:
                self.phase='failed';self.groups.fail_update();raise
