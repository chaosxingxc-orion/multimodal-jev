import copy

import pytest
import torch

from tools.reports.context_dual_session_candidate import RuntimeReferenceCache,DualContextRolloutSession
from multimodal_jev.context_logits_wire import WIRE
from multimodal_jev.context_dual_credit import verify
from test_context_rollout_session import REF,P0,P1,BASE,CONTRACT,spec,item
import context_reference_cache

RUNTIME=dict(image_sha256='1'*64,source_manifest_sha256='2'*64,processor_sha256='3'*64,
             profile='dual-batch4',output_protocol=WIRE)


def query(pid,i=0):return dict(item(pid,i),candidate_ids=['b','a'])


class Engine:
    def __init__(self,cache,policy=P0):
        self.cache=cache;self.bundle=dict(policy_id=policy,contract=CONTRACT)
        self.runtime_identity=copy.deepcopy(RUNTIME);self.calls=[];self.switches=[];self.fault=None
    def switch(self,**kw):
        self.bundle['policy_id']=kw['policy_id'];self.switches.append(kw['policy_id'])
        return dict(policy_id=kw['policy_id'])
    def score_many(self,rows,*,audio_token_id):
        assert all(set(r)=={'inputs','positions'} for r in rows) # no target leakage
        pid=self.bundle['policy_id'];self.calls.append((pid,len(rows)))
        z=torch.tensor({REF:[.5,-.5],P0:[-.5,.5],P1:[1.,-1.]}[pid])
        records=[]
        for row in rows:
            identity=self.cache.binding(**row);identity['policy_id']=pid
            records.append(dict(policy_id=pid,request_identity=identity,request_id=self.cache.digest(identity),
                prob=z.sigmoid().tolist(),logits=z.tolist(),output_protocol=WIRE))
        if self.fault=='reverse':records.reverse()
        if self.fault=='truncated':records.pop()
        if self.fault=='logit':records[0]['logits'][0]+=1
        if self.fault=='switched':self.bundle['policy_id']=P1
        if self.fault=='saturated_pair':
            for r in records:
                # Canonical pair is (a,b), opposite to this input order.
                z=torch.tensor([-10.,10.]);r.update(logits=z.tolist(),prob=z.sigmoid().tolist())
        return records


def setup(tmp_path,monkeypatch,*,runtime=RUNTIME,policy=P0,start=0):
    monkeypatch.setattr(context_reference_cache,'check_paths',lambda _:None)
    cache=RuntimeReferenceCache(tmp_path,policy_id=REF,base_id=BASE,contract=CONTRACT,runtime_identity=runtime)
    engine=Engine(cache,policy)
    session=DualContextRolloutSession(engine,cache,runtime_identity=runtime,candidate_ids=['b','a'],
        reference=spec(REF),initial_policy=spec(policy),audio_token_id=1,start_update=start)
    return session,engine,cache


def test_bulk_two_updates_fresh_behavior_and_one_shared_reference_fill(tmp_path,monkeypatch):
    s,e,cache=setup(tmp_path,monkeypatch)
    data,options,r=s.prepare([query(P0,0),query(P0,1)],update=0)
    assert e.calls==[(REF,2),(P0,2)] and r['reference_misses']==2
    assert options['algorithm']=='dual_credit_rloo_clipped_v1'
    for p in data:
        verify(p['dual_credit'])
        assert p['dual_credit']['branches']['ranking']['keys']==(('a','b'),)
        assert p['dual_credit']['branches']['ranking']['actions'].shape==(1,16)
    s.accept_actor_update(update=0,next_policy=spec(P1))
    second,_,r=s.prepare([query(P1,0),query(P1,1)],update=1)
    assert e.calls==[(REF,2),(P0,2),(P1,2)] and r['reference_misses']==0
    assert e.switches==[REF,P0,P1]
    assert data[0]['dual_credit']['binding']['probability']!=second[0]['dual_credit']['binding']['probability']
    assert torch.equal(data[0]['reference_probability'],second[0]['reference_probability'])


def test_fresh_session_reuses_reference_and_exact_dual_samples(tmp_path,monkeypatch):
    s,e,_=setup(tmp_path,monkeypatch);s.prepare([query(P0)],update=0)
    s.accept_actor_update(update=0,next_policy=spec(P1))
    expected,_,_=s.prepare([query(P1)],update=1)
    fresh,e2,_=setup(tmp_path,monkeypatch,policy=P1,start=1)
    actual,_,r=fresh.prepare([query(P1)],update=1)
    assert e2.calls==[(P1,1)] and r['reference_misses']==0
    for branch in ('ranking','existence'):
        for field in ('actions','advantages','old_logprob','rewards'):
            assert torch.equal(actual[0]['dual_credit']['branches'][branch][field],
                               expected[0]['dual_credit']['branches'][branch][field])


def test_duplicate_padding_preserves_slots_and_single_reference_entry(tmp_path,monkeypatch):
    s,e,_=setup(tmp_path,monkeypatch);x=query(P0);pad=copy.deepcopy(x)
    pad.update(request_id='padding',real_request=False,request_weight=0.)
    data,_,r=s.prepare([x,pad],update=0)
    assert e.calls==[(REF,1),(P0,2)] and r['reference_misses']==1
    assert not data[1]['real_request'] and data[1]['request_weight']==0
    assert data[0]['dual_credit']['binding']['query_id']!=data[1]['dual_credit']['binding']['query_id']


@pytest.mark.parametrize('fault',['reverse','truncated','logit','switched'])
def test_bad_actual_engine_batch_poisoned_without_actor(tmp_path,monkeypatch,fault):
    s,e,_=setup(tmp_path,monkeypatch);e.fault=fault
    with pytest.raises(ValueError):s.prepare([query(P0,0),query(P0,1)],update=0)
    assert s.phase=='failed' and s.groups.phase=='failed' and s.groups.cursor==0
    with pytest.raises(ValueError):s.prepare([query(P0)],update=0)


@pytest.mark.parametrize('fault',['candidate_order','missing_label','weight','duplicate','stale_policy'])
def test_invalid_training_contract_fails_before_inference(tmp_path,monkeypatch,fault):
    s,e,_=setup(tmp_path,monkeypatch);x=query(P0);rows=[x]
    if fault=='candidate_order':x['candidate_ids'].reverse()
    if fault=='missing_label':x['observed'][0,0]=False
    if fault=='weight':x['request_weight']=2.
    if fault=='duplicate':rows.append(copy.deepcopy(x))
    if fault=='stale_policy':x['policy_id']=P1
    with pytest.raises(ValueError):s.prepare(rows,update=0)
    assert e.calls==[]


def test_reference_runtime_namespace_separates_profiles_and_rejects_mismatch(tmp_path,monkeypatch):
    s,e,cache=setup(tmp_path,monkeypatch);s.prepare([query(P0)],update=0)
    changed=dict(RUNTIME,source_manifest_sha256='4'*64)
    other=RuntimeReferenceCache(tmp_path,policy_id=REF,base_id=BASE,contract=CONTRACT,runtime_identity=changed)
    row=query(P0)
    assert cache.read(row['inputs'],row['positions']) is not None
    assert other.read(row['inputs'],row['positions']) is None and other.namespace!=cache.namespace
    with pytest.raises(ValueError,match='runtime'):
        DualContextRolloutSession(e,other,runtime_identity=changed,candidate_ids=['b','a'],
            reference=spec(REF),initial_policy=spec(P0),audio_token_id=1)


def test_reference_cache_conflicting_probability_never_overwritten(tmp_path,monkeypatch):
    s,e,cache=setup(tmp_path,monkeypatch);s.prepare([query(P0)],update=0)
    row=query(P0);old=cache.read(row['inputs'],row['positions']);changed=copy.deepcopy(old)
    changed['prob']=[.8,.2]
    with pytest.raises(ValueError,match='collision'):cache.write(changed,row['inputs'],row['positions'])
    assert cache.read(row['inputs'],row['positions'])==old


def test_actual_cached_loader_binds_compiled_candidates_without_new_input_pipeline(monkeypatch):
    from test_cached_context_batches import fixture
    from multimodal_jev.cached_context_batches import CachedContextBatches
    from tools.reports.context_dual_session_candidate import bind_candidates
    population,task,batches,preparer=fixture(monkeypatch)
    with CachedContextBatches(batches,population,task,preparer,start_update=0,stop_update=2,global_batch=2) as stream:
        items,metadata=stream.consume(0,P0)
        rows=bind_candidates(items,metadata,['a','b'])
        assert rows[0]['candidate_ids']==['a','b'] and rows[0]['inputs'] is items[0]['inputs']
        with pytest.raises(ValueError):bind_candidates(items,metadata,['b','a'])
        bad=copy.deepcopy(metadata);bad[0]['target']['labels'].reverse()
        with pytest.raises(ValueError):bind_candidates(items,bad,['a','b'])
        with pytest.raises(ValueError):bind_candidates(items,metadata[:-1],['a','b'])


def test_large_finite_pair_gap_keeps_all_queries(tmp_path,monkeypatch):
    s,e,_=setup(tmp_path,monkeypatch);e.fault='saturated_pair'
    data,_,receipt=s.prepare([query(P0,0),query(P0,1)],update=0)
    assert len(data)==len(receipt['queries'])==2 and s.phase=='awaiting_actor'
