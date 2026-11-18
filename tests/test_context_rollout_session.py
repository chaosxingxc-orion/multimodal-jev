import copy,hashlib,json,sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from context_rollout_session import ContextRolloutSession
from context_reference_cache import ReferenceCache
import context_reference_cache

REF='a'*64;P0='b'*64;P1='c'*64;BASE='d'*64
CONTRACT={'base_id':BASE}

def spec(pid):return dict(policy_id=pid,snapshot='/policy/'+pid,adapter='/adapter/'+pid)

def item(pid,i=0):
    return dict(request_id=f'q{i}',policy_id=pid,inputs={'input_ids':torch.tensor([[1,2,3]]),
        'input_features':torch.full((1,2,3),float(i+1))},positions=torch.tensor([[1,2]]),
        labels=torch.tensor([[True,False]]),observed=torch.ones(1,2,dtype=torch.bool),
        real_request=True,request_weight=1.)

class Engine:
    def __init__(self,cache):self.cache=cache;self.bundle=dict(policy_id=P0,contract=CONTRACT);self.calls=[];self.switches=[];self.fail=False
    def switch(self,**kw):
        if self.fail:raise ValueError('worker switch failure')
        self.bundle['policy_id']=kw['policy_id'];self.switches.append(kw['policy_id']);return {'policy_id':kw['policy_id']}
    def score(self,inputs,positions,*,audio_token_id):
        pid=self.bundle['policy_id'];self.calls.append(pid)
        identity=self.cache.binding(inputs,positions);identity['policy_id']=pid
        prob={REF:[.7,.3],P0:[.4,.6],P1:[.8,.2]}[pid]
        return dict(policy_id=pid,request_identity=identity,request_id=self.cache.digest(identity),prob=prob)


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(context_reference_cache,'check_paths',lambda x:None)
    cache=ReferenceCache(tmp_path,policy_id=REF,base_id=BASE,contract=CONTRACT)
    e=Engine(cache);s=ContextRolloutSession(e,cache,reference=spec(REF),initial_policy=spec(P0),audio_token_id=1)
    return s,e,cache


def test_two_batches_one_engine_fresh_behavior_and_fixed_reference(tmp_path,monkeypatch):
    s,e,cache=setup(tmp_path,monkeypatch)
    a,opt,r=s.prepare([item(P0,0),item(P0,1)],update=0)
    assert e.switches==[REF,P0] and e.calls==[REF,REF,P0,P0] and r['reference_misses']==2
    s.accept_actor_update(update=0,next_policy=spec(P1))
    b,opt2,r2=s.prepare([item(P1,0),item(P1,1)],update=1)
    assert e.switches==[REF,P0,P1] and e.calls[-2:]==[P1,P1] and r2['reference_misses']==0
    assert opt2['reference_policy_id']==REF
    assert not torch.equal(a[0]['behavior'][0],b[0]['behavior'][0])
    assert torch.equal(a[0]['reference_policy'][0],b[0]['reference_policy'][0])


def test_resume_new_session_retains_reference_and_sampling(tmp_path,monkeypatch):
    s,e,cache=setup(tmp_path,monkeypatch);s.prepare([item(P0)],update=0)
    s.accept_actor_update(update=0,next_policy=spec(P1));expected,_,_=s.prepare([item(P1)],update=1)
    e2=Engine(cache);e2.bundle['policy_id']=P1
    restored=ContextRolloutSession(e2,cache,reference=spec(REF),initial_policy=spec(P1),audio_token_id=1,start_update=1)
    actual,_,r=restored.prepare([item(P1)],update=1)
    assert torch.equal(actual[0]['actions'],expected[0]['actions']) and e2.calls==[P1] and r['reference_misses']==0


def test_switch_failure_prevents_cursor_advance(tmp_path,monkeypatch):
    s,e,_=setup(tmp_path,monkeypatch);s.prepare([item(P0)],update=0);e.fail=True
    with pytest.raises(ValueError):s.accept_actor_update(update=0,next_policy=spec(P1))
    assert s.phase=='failed' and s.groups.cursor==0
    with pytest.raises(ValueError):s.prepare([item(P1)],update=1)


def test_duplicate_input_slots_share_reference_but_not_behavior(tmp_path,monkeypatch):
    s,e,_=setup(tmp_path,monkeypatch);x=item(P0);y=copy.deepcopy(x);y['request_id']='padding';y['real_request']=False;y['request_weight']=0.
    data,_,r=s.prepare([x,y],update=0)
    assert r['reference_misses']==1 and e.calls==[REF,P0,P0] and data[1]['request_weight']==0.


def test_pending_actor_cannot_prepare_twice_and_failure_stops_session(tmp_path,monkeypatch):
    s,e,_=setup(tmp_path,monkeypatch);s.prepare([item(P0)],update=0)
    with pytest.raises(ValueError):s.prepare([item(P0)],update=0)
    s.fail_actor_update()
    with pytest.raises(ValueError):s.accept_actor_update(update=0,next_policy=spec(P1))
