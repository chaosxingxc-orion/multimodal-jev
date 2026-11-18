import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from context_reference_cache import ReferenceCache
import context_reference_cache
from tools.reports.context_attribute_schedule_candidate import AttributeScheduledBatches, AttributeRolloutSession
from tools.reports.context_attribute_actor_candidate import generate,REFERENCE_ID
from tools.reports.build_context_attribute_candidate import request,targets,digest
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports.probe_context_attribute_parity_v2 import content_hash


def adapter():
    menu=json.loads(Path('.local/decision-objective-research/slurp-native-attribute-prob-candidate-v1/catalog.json').read_text())
    _,contract=generate(menu,'q_credit',source_root=Path.cwd())
    class Adapter:
        arm='q_credit'
        def __call__(self,job):
            req,h=job
            assert len(req.questions)==124 and h=='a'*64
            return SimpleNamespace(inputs=dict(input_ids=torch.ones(1,130,dtype=torch.long),
                input_features=torch.zeros(1,4,100)),positions=torch.arange(124)[None],
                cache_hit=True,audio_receipt={'audio_sha256':h})
        def actor_item(self,asset,target,encoded,*,policy_id,slot_id,padding):
            return dict(request_id=slot_id,policy_id=policy_id,inputs=encoded.inputs,
                positions=encoded.positions,labels=torch.tensor([target['labels']]),
                observed=torch.tensor([target['supervised']]),real_request=not padding,
                request_weight=0. if padding else 1.,attribute_contract_id=self.contract['contract_id'],
                decision_ids=self.contract['decision_ids'])
    obj=Adapter();obj.menu=menu;obj.contract=contract
    return obj


def population(a):
    assets=[dict(id='a',audio='a.wav',audio_sha256='a'*64,catalog_sha256=digest(a.menu)),
            dict(id='b',audio='b.wav',audio_sha256='a'*64,catalog_sha256=digest(a.menu))]
    labels=[dict(id=x['id'],input_sha256=digest(x),**targets(a.menu,a.menu['events'][i]['id']))
            for i,x in enumerate(assets)]
    slots=[dict(id='a' if i%2==0 else 'b',padding=i==127,weight=0. if i==127 else 1.) for i in range(128)]
    batches=[dict(epoch=0,index=k,slots=copy.deepcopy(slots)) for k in range(2)]
    return assets,labels,batches


def test_frozen_schedule_slots_padding_and_new_cursor():
    a=adapter();assets,labels,batches=population(a)
    with AttributeScheduledBatches(batches,assets,labels,a,source_root=Path.cwd(),start_update=0,stop_update=2) as stream:
        items,meta=stream.consume(0,'p0')
        assert len(items)==128 and len({i['request_id'] for i in items})==128
        assert sum(i['request_weight'] for i in items)==127
        assert not items[-1]['real_request'] and items[-1]['labels'].shape==(1,124)
        assert stream.last_timing['cache_hits']==128
        with pytest.raises(ValueError):stream.consume(0,'p0')
        expected,_=stream.consume(1,'p1')
    with AttributeScheduledBatches(batches,assets,labels,a,source_root=Path.cwd(),start_update=1,stop_update=2) as fresh:
        actual,_=fresh.consume(1,'p1')
        assert [x['request_id'] for x in actual]==[x['request_id'] for x in expected]
        assert all(torch.equal(x['labels'],y['labels']) for x,y in zip(actual,expected))


@pytest.mark.parametrize('fault',['target','padding','short','unknown'])
def test_bad_schedule_rejected_before_prefetch(fault):
    a=adapter();assets,labels,batches=population(a)
    if fault=='target':labels[0]['input_sha256']='bad'
    if fault=='padding':batches[0]['slots'][-1]['weight']=1.
    if fault=='short':batches[0]['slots'].pop()
    if fault=='unknown':batches[0]['slots'][0]['id']='missing'
    with pytest.raises(ValueError):AttributeScheduledBatches(batches,assets,labels,a,
        source_root=Path.cwd(),start_update=0,stop_update=1)


class Engine:
    def __init__(self):self.bundle=dict(policy_id='p0',contract={'base_id':'b'*64});self.calls=[];self.switches=[]
    def switch(self,**kw):
        self.bundle['policy_id']=kw['policy_id'];self.switches.append(kw['policy_id']);return kw
    def score(self,inputs,positions,*,audio_token_id):
        pid=self.bundle['policy_id'];self.calls.append(pid)
        identity=dict(policy_id=pid,positions=positions[0].tolist(),
                      inputs={k:tensor_identity(v) for k,v in sorted(inputs.items())})
        return dict(policy_id=pid,request_identity=identity,request_id=content_hash(identity),
                    prob=[.25 if pid==REFERENCE_ID else .4]*124)


def spec(pid):return dict(policy_id=pid,snapshot='/policy/'+pid,adapter='/adapter/'+pid)


def test_attribute_session_refresh_and_restored_sampling(tmp_path,monkeypatch):
    monkeypatch.setattr(context_reference_cache,'check_paths',lambda x:None)
    a=adapter();assets,labels,batches=population(a);e=Engine()
    cache=ReferenceCache(tmp_path,policy_id=REFERENCE_ID,base_id='b'*64,contract=e.bundle['contract'])
    def session(engine,start):return AttributeRolloutSession(engine,cache,input_adapter=a,source_root=Path.cwd(),
        reference=spec(REFERENCE_ID),initial_policy=spec(engine.bundle['policy_id']),audio_token_id=151676,start_update=start)
    with AttributeScheduledBatches(batches,assets,labels,a,source_root=Path.cwd(),start_update=0,stop_update=2) as stream:
        s=session(e,0);items,_=stream.consume(0,'p0');data,options,receipt=s.prepare(items,update=0)
        assert options['attribute_contract_id']==a.contract['contract_id'] and receipt['reference_misses']==1
        assert e.switches==[REFERENCE_ID,'p0'] and len(e.calls)==129
        s.accept_actor_update(update=0,next_policy=spec('p1'))
        items,_=stream.consume(1,'p1');expected,_,r=s.prepare(items,update=1)
        assert r['reference_misses']==0
        other=Engine();other.bundle['policy_id']='p1'
        restored=session(other,1);actual,_,r=restored.prepare(items,update=1)
        assert other.calls==['p1']*128 and r['reference_misses']==0
        assert all(torch.equal(x['actions'],y['actions']) for x,y in zip(actual,expected))
        bad=session(Engine(),0);items[0]['attribute_contract_id']='wrong'
        with pytest.raises(ValueError):bad.prepare(items,update=0)
        assert bad.engine.calls==[] and bad.phase=='failed'
