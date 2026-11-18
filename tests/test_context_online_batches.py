import copy
import hashlib,json
import importlib.util
from pathlib import Path

import pytest
import torch

spec=importlib.util.spec_from_file_location('context_online_batches',Path(__file__).parents[1]/'tools/jev/context_online_batches.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.span_request_identity import tensor_identity


class Engine:
    def __init__(self,policy,prob):self.bundle={'policy_id':policy};self.prob=prob;self.calls=0
    def score(self,inputs,positions,*,audio_token_id):
        self.calls+=1
        identity=dict(policy_id=self.bundle['policy_id'],positions=positions[0].tolist(),
                      inputs={k:tensor_identity(v) for k,v in sorted(inputs.items())})
        return dict(policy_id=self.bundle['policy_id'],request_identity=identity,
                    request_id=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),prob=list(self.prob))


def item(policy='p0'):
    return dict(request_id='query::epoch0',policy_id=policy,inputs={'input_ids':torch.tensor([[3,9,9]]),
        'input_features':torch.ones(1,2,3)},positions=torch.tensor([[1,2]]),
        labels=torch.tensor([[True,False]]),observed=torch.ones(1,2,dtype=torch.bool),
        real_request=True,request_weight=1.)


def session():
    reference=Engine('ref',[.7,.3])
    return m.OnlineContextBatches(reference_engine=reference,reference_policy_id='ref',audio_token_id=3),reference


def test_two_updates_refresh_behavior_but_keep_reference_and_decay():
    s,ref=session();e0=Engine('p0',[.2,.8]);a,opts,r=s.prepare([item()],e0,policy_id='p0',update=0)
    assert opts['ranking_coefficient']==pytest.approx(.3)
    assert a[0]['actions'].shape==(1,16,2) and ref.calls==e0.calls==1
    s.acknowledge_update(update=0,next_policy_id='p1')
    e1=Engine('p1',[.9,.1]);b,opts2,r2=s.prepare([item('p1')],e1,policy_id='p1',update=1)
    assert e1.calls==1 and ref.calls==1 and r2[0]['reference_cache_hit']
    assert not torch.equal(a[0]['behavior'][0],b[0]['behavior'][0])
    assert all(torch.equal(x,y) for x,y in zip(a[0]['reference_policy'],b[0]['reference_policy']))
    assert opts2['ranking_coefficient']<opts['ranking_coefficient']
    assert r[0]['rollout']['seed']!=r2[0]['rollout']['seed']


def test_changed_input_never_hits_reference_and_actions_reproducible_after_resume():
    s,ref=session();a,_,r=s.prepare([item()],Engine('p0',[.3,.6]),policy_id='p0',update=0)
    fresh,_=session();b,_,r2=fresh.prepare([item()],Engine('p0',[.3,.6]),policy_id='p0',update=0)
    assert torch.equal(a[0]['actions'],b[0]['actions']) and r==r2
    s.acknowledge_update(update=0,next_policy_id='p1')
    changed=item('p1');changed['inputs']['input_features'][0,0,0]=2
    _,_,r=s.prepare([changed],Engine('p1',[.4,.5]),policy_id='p1',update=1)
    assert ref.calls==2 and not r[0]['reference_cache_hit']


@pytest.mark.parametrize('fault',['old_policy','wrong_activation','missing_gold','saturation','duplicate','bad_weight'])
def test_invalid_batch_poisoned_before_actor(fault):
    s,ref=session();s.prepare([item()],Engine('p0',[.2,.8]),policy_id='p0',update=0)
    s.acknowledge_update(update=0,next_policy_id='p1')
    data=[item('p1')];engine=Engine('p1',[.2,.8]);pid='p1'
    if fault=='old_policy':data[0]['policy_id']='p0'
    if fault=='wrong_activation':engine=Engine('p0',[.2,.8]);pid='p0';data=[item('p0')]
    if fault=='missing_gold':data[0]['observed'][0,0]=False
    if fault=='saturation':engine.prob=[1.,.5]
    if fault=='duplicate':data*=2
    if fault=='bad_weight':data[0]['request_weight']=2.
    with pytest.raises(ValueError):s.prepare(data,engine,policy_id=pid,update=1)
    assert s.phase=='failed'
    with pytest.raises(ValueError):s.prepare([item('p1')],Engine('p1',[.5,.5]),policy_id='p1',update=1)


def test_no_second_rollout_before_ack_and_no_skipped_cursor():
    s,_=session();engine=Engine('p0',[.2,.8]);s.prepare([item()],engine,policy_id='p0',update=0)
    with pytest.raises(ValueError):s.prepare([item()],engine,policy_id='p0',update=0)
    with pytest.raises(ValueError):s.acknowledge_update(update=1,next_policy_id='p1')
    s.fail_update()
    with pytest.raises(ValueError):s.acknowledge_update(update=0,next_policy_id='p1')


def test_decay_endpoints_and_monotonicity():
    values=[m.auxiliary_coefficient(i) for i in range(634)]
    assert values[0]==pytest.approx(.3) and values[507:]==[.1]*127
    assert all(a>=b for a,b in zip(values,values[1:]))
    for i in (-1,634,True):
        with pytest.raises(ValueError):m.auxiliary_coefficient(i)


def test_reference_engine_cannot_switch_and_initial_shared_engine_runs_once():
    ref=Engine('ref',[.7,.3])
    s=m.OnlineContextBatches(reference_engine=ref,reference_policy_id='ref',audio_token_id=3)
    data,_,_=s.prepare([item('ref')],ref,policy_id='ref',update=0)
    assert ref.calls==1
    s.acknowledge_update(update=0,next_policy_id='p1')
    ref.bundle['policy_id']='p1'
    with pytest.raises(ValueError,match='reference engine changed'):
        s.prepare([item('p1')],Engine('p1',[.4,.6]),policy_id='p1',update=1)
    assert s.phase=='failed'


def test_padding_keeps_group_but_zero_weight_and_only_padding_rejected():
    s,_=session();pad=item();pad.update(request_id='padding',real_request=False,request_weight=0.)
    data,_,_=s.prepare([item(),pad],Engine('p0',[.2,.8]),policy_id='p0',update=0)
    assert len(data)==2 and data[1]['request_weight']==0. and data[1]['actions'].shape==(1,16,2)
    s,_=session()
    with pytest.raises(ValueError,match='nonempty real population'):
        s.prepare([pad],Engine('p0',[.2,.8]),policy_id='p0',update=0)
