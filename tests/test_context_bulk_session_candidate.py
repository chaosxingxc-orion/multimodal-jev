import copy
import pytest
import torch
from test_context_rollout_session import Engine,ReferenceCache,CONTRACT,REF,P0,P1,BASE,spec,item
import context_reference_cache
from context_rollout_session import ContextRolloutSession
from tools.reports.context_bulk_session_candidate import BulkContextRolloutSession


def full_item(pid,i=0):
    x=item(pid,i);x['inputs']['attention_mask']=torch.ones(1,3,dtype=torch.long)
    x['inputs']['input_features_mask']=torch.ones(1,3,dtype=torch.long)
    return x


class BulkEngine(Engine):
    def __init__(self,cache):super().__init__(cache);self.transactions=[];self.fault=None
    def score_many(self,rows,*,audio_token_id):
        assert all(set(r)=={'inputs','positions'} for r in rows)
        self.transactions.append((self.bundle['policy_id'],len(rows)))
        records=[self.score(**r,audio_token_id=audio_token_id) for r in rows]
        if self.fault=='short' or (self.fault=='behavior_short' and self.bundle['policy_id']!=REF):return records[:-1]
        if self.fault=='reverse':return records[::-1]
        return records


def setup(path,monkeypatch,cls):
    monkeypatch.setattr(context_reference_cache,'check_paths',lambda paths:None)
    cache=ReferenceCache(path,policy_id=REF,base_id=BASE,contract=CONTRACT)
    engine=BulkEngine(cache)
    session=cls(engine,cache,reference=spec(REF),initial_policy=spec(P0),audio_token_id=1)
    return session,engine,cache


def equal_output(a,b):
    assert a[1]==b[1] and a[2]['queries']==b[2]['queries']
    for x,y in zip(a[0],b[0]):
        assert torch.equal(x['actions'],y['actions'])
        for key in ('behavior','reference_policy'):
            assert all(torch.equal(u,v) for u,v in zip(x[key],y[key]))


def test_full_batch_two_updates_padding_and_restore_match_original(tmp_path,monkeypatch):
    old,oe,oc=setup(tmp_path/'old',monkeypatch,ContextRolloutSession)
    new,ne,nc=setup(tmp_path/'new',monkeypatch,BulkContextRolloutSession)
    items=[full_item(P0,i) for i in range(127)];pad=copy.deepcopy(items[0]);pad.update(request_id='padding',real_request=False,request_weight=0.);items.append(pad)
    equal_output(old.prepare(items,update=0),new.prepare(items,update=0))
    assert ne.transactions==[(REF,127),(P0,128)] and oe.calls==ne.calls
    for s in (old,new):s.accept_actor_update(update=0,next_policy=spec(P1))
    for x in items:x['policy_id']=P1
    expected=old.prepare(items,update=1);actual=new.prepare(items,update=1);equal_output(expected,actual)
    assert ne.transactions[-1]==(P1,128) and actual[2]['reference_misses']==0
    engine=BulkEngine(nc);engine.bundle['policy_id']=P1
    restored=BulkContextRolloutSession(engine,nc,reference=spec(REF),initial_policy=spec(P1),audio_token_id=1,start_update=1)
    replay=restored.prepare(items,update=1)
    assert engine.transactions==[(P1,128)]
    # New session's in-memory hit flag differs; sampling and distributions do not.
    for a,b in zip(actual[0],replay[0]):assert torch.equal(a['actions'],b['actions'])


@pytest.mark.parametrize('fault',['missing','duplicate','weight','policy','empty_real'])
def test_invalid_items_fail_before_bulk_forward(tmp_path,monkeypatch,fault):
    s,e,c=setup(tmp_path,monkeypatch,BulkContextRolloutSession);items=[full_item(P0)]
    if fault=='missing':items[0]['observed'][0,0]=False
    if fault=='duplicate':items*=2
    if fault=='weight':items[0]['request_weight']=2.
    if fault=='policy':items[0]['policy_id']=P1
    if fault=='empty_real':items[0].update(real_request=False,request_weight=0.)
    with pytest.raises(ValueError):s.prepare(items,update=0)
    assert s.phase=='failed' and not e.transactions


@pytest.mark.parametrize('fault',['short','reverse'])
def test_reference_batch_error_does_not_publish_partial_cache(tmp_path,monkeypatch,fault):
    s,e,c=setup(tmp_path,monkeypatch,BulkContextRolloutSession);e.fault=fault
    with pytest.raises(ValueError):s.prepare([full_item(P0,0),full_item(P0,1)],update=0)
    assert not list(c.namespace.glob('*.json')) and s.groups.cursor==0 and s.phase=='failed'


def test_behavior_failure_retains_valid_reference_but_cannot_ack_or_retry(tmp_path,monkeypatch):
    s,e,c=setup(tmp_path,monkeypatch,BulkContextRolloutSession);e.fault='behavior_short'
    items=[full_item(P0,0),full_item(P0,1)]
    with pytest.raises(ValueError):s.prepare(items,update=0)
    assert len(list(c.namespace.glob('*.json')))==2 and s.groups.cursor==0
    with pytest.raises(ValueError):s.prepare(items,update=0)
    with pytest.raises(ValueError):s.accept_actor_update(update=0,next_policy=spec(P1))
