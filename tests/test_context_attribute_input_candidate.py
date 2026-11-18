"""Real processor/audio/cache tests; only probability engine is a test double."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from tools.reports.context_attribute_input_candidate import AttributeInputs, AttributeOnlineBatches
from tools.reports.context_attribute_actor_candidate import load, REFERENCE_ID
from tools.reports.context_attribute_compact_candidate import encode
from tools.reports.build_context_attribute_candidate import request, targets, digest
from multimodal_jev.audio_policy import load_audio
from multimodal_jev.encoded_audio_cache import EncodedAudioCache, key_for
from multimodal_jev.cached_context_preparation import CachedContextPreparer
from multimodal_jev.span_request_identity import tensor_identity


class Engine:
    def __init__(self, policy):self.bundle={'policy_id':policy};self.calls=0
    def score(self,inputs,positions,*,audio_token_id):
        assert audio_token_id==151676
        assert set(inputs)=={'input_ids','attention_mask','input_features','input_features_mask'}
        self.calls+=1
        identity=dict(policy_id=self.bundle['policy_id'],positions=positions[0].tolist(),
            inputs={k:tensor_identity(v) for k,v in sorted(inputs.items())})
        uid=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return dict(policy_id=self.bundle['policy_id'],request_identity=identity,request_id=uid,
                    prob=torch.linspace(.15,.75,positions.shape[1]).tolist())


@pytest.fixture(scope='module')
def real_inputs(tmp_path_factory):
    from transformers import Qwen3ASRProcessor
    root=Path.cwd();fixture=root/'input-fixture'
    if not fixture.exists():fixture=root/'.local/decision-objective-research/context-attribute-compact-processor-cpu-v1-package'
    menu=json.loads((fixture/'catalog.json').read_text())
    assets=json.loads((fixture/'audio.json').read_text())
    processor=Qwen3ASRProcessor.from_pretrained(fixture/'processor',local_files_only=True)
    processor_sha=digest({str(p.relative_to(fixture/'processor')):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (fixture/'processor').rglob('*') if p.is_file()})
    cache=EncodedAudioCache(tmp_path_factory.mktemp('attribute-cache'))
    calls=[]
    def factory():calls.append(True);return processor
    kwargs=dict(source_root=root,processor_factory=factory,cache=cache,processor_sha256=processor_sha,
        decision_token_id=151706,audio_token_id=151676,max_tokens=4096)
    adapters={a:AttributeInputs(menu,a,**kwargs) for a in ('direct','q_input','q_credit')}
    records=[]
    for asset in assets:
        row=dict(id=asset['id'],audio=str(fixture/asset['path']),audio_sha256=asset['audio_sha256'],catalog_sha256=digest(menu))
        target=dict(id=row['id'],input_sha256=digest(row),**targets(menu,menu['events'][0]['id']))
        records.append((row,target))
    return menu,records,adapters,processor,calls,kwargs


def test_actual_audio_cold_warm_and_q_arm_cache_sharing(real_inputs):
    menu,records,adapters,processor,calls,kwargs=real_inputs
    for asset,target in records:
        wave=load_audio(asset['audio'])
        for arm in ('direct','q_input'):
            adapter=adapters[arm];req=request(asset['audio'],menu,arm=arm);job=(req,asset['audio_sha256'])
            first=adapter(job);second=adapter(job)
            assert not first.cache_hit and second.cache_hit
            expected,pos=encode(processor,req,wave,decision_token_id=151706,audio_token_id=151676,max_tokens=4096)
            for k,v in expected.items():
                assert torch.equal(first.inputs[k],v.to(torch.bfloat16) if v.is_floating_point() else v)
                assert torch.equal(first.inputs[k],second.inputs[k])
            assert torch.equal(first.positions,pos)
            original=CachedContextPreparer(**{k:v for k,v in kwargs.items() if k!='source_root'})
            assert key_for(original.identity(req,asset['audio_sha256']))!=key_for(adapter.identity(req,asset['audio_sha256']))
            if arm=='q_input':
                other=adapters['q_credit'];assert other(job).cache_hit
                assert key_for(other.identity(req,asset['audio_sha256']))==key_for(adapter.identity(req,asset['audio_sha256']))
                assert other.contract['contract_id']!=adapter.contract['contract_id']
    assert len(calls)==2  # Q-credit consumed cache without constructing a processor.


@pytest.mark.parametrize('arm',['direct','q_input','q_credit'])
def test_actual_encoded_inputs_to_online_group_and_native_transport(real_inputs,arm):
    menu,records,adapters,_,_,_=real_inputs;adapter=adapters[arm];asset,target=records[0]
    req=request(asset['audio'],menu,arm=arm);prepared=adapter((req,asset['audio_sha256']))
    ref=Engine(REFERENCE_ID);session=AttributeOnlineBatches(adapter,reference_engine=ref,
        reference_policy_id=REFERENCE_ID,audio_token_id=151676)
    actor,_=load(menu,arm,source_root=Path.cwd())
    for step in range(2):
        policy=f'policy-{step}';engine=Engine(policy)
        item=adapter.actor_item(asset,target,prepared,policy_id=policy,slot_id=f'{asset["id"]}:{step}')
        items,options,receipts=session.prepare([item],engine,policy_id=policy,update=step)
        batch=actor.make_batch(items,**options);payload=batch.non_tensor_batch['multi_modal_inputs'][0]
        assert payload['actions'].shape==(1,16,len(req.questions))
        assert payload['attribute_contract_id']==adapter.contract['contract_id']
        assert ref.calls==1 and engine.calls==1
        assert receipts[0]['reference_cache_hit']==bool(step)
        assert payload['labels'].sum()==(1 if arm=='direct' else 3)
        session.acknowledge_update(update=step,next_policy_id=f'policy-{step+1}')


@pytest.mark.parametrize('fault',['target_binding','node_order','missing','wrong_catalog','float_label','prepared_input','padding'])
def test_late_target_binding_rejects_faults(real_inputs,fault):
    menu,records,adapters,_,_,_=real_inputs;adapter=adapters['q_input']
    asset,target=copy.deepcopy(records[0]);req=request(asset['audio'],menu,arm='q_input')
    prepared=adapter((req,asset['audio_sha256']));padding=False
    if fault=='target_binding':target['input_sha256']='0'*64
    if fault=='node_order':target['candidate_ids']=target['candidate_ids'][::-1]
    if fault=='missing':target['supervised'][0]=False
    if fault=='wrong_catalog':asset['catalog_sha256']='0'*64
    if fault=='float_label':target['labels'][0]=0.
    if fault=='prepared_input':prepared.inputs['input_ids'][0,prepared.positions[0,0]]=1
    if fault=='padding':padding=1
    with pytest.raises(ValueError):adapter.actor_item(asset,target,prepared,policy_id='p',slot_id='slot',padding=padding)


def test_wrong_online_arm_rejected_before_engine_access(real_inputs):
    menu,records,adapters,_,_,_=real_inputs;adapter=adapters['q_input'];asset,target=records[0]
    prepared=adapter((request(asset['audio'],menu,arm='q_input'),asset['audio_sha256']))
    item=adapter.actor_item(asset,target,prepared,policy_id='p',slot_id='slot')
    item['attribute_contract_id']=adapters['q_credit'].contract['contract_id']
    ref=Engine(REFERENCE_ID);engine=Engine('p')
    session=AttributeOnlineBatches(adapter,reference_engine=ref,reference_policy_id=REFERENCE_ID,audio_token_id=151676)
    with pytest.raises(ValueError):session.prepare([item],engine,policy_id='p',update=0)
    assert ref.calls==engine.calls==0 and session.phase=='failed'
