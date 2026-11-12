import copy
import math
import hashlib
import json
import pytest
from multimodal_jev.intent_binding_inputs import select,bind_behavior
from multimodal_jev.span_candidate_intervention import VARIANTS,from_input
from multimodal_jev.intent_binding_training import expand
from test_intent_binding_training import fixture


def rows():
    result=[]
    for i in range(9):
        item=fixture();item['id']=str(i)
        for r,t in expand(item):result.append(dict(r,request=from_input(r['request']),target=t,
            gate_id=r['source_id']+'::Q::'+r['variant']))
    return result


def test_schedule_groups_all_views_and_preserves_source_budget():
    selected,receipt=select(rows(),317,0)
    assert len(selected)==32 and len(set(receipt['source_ids']))==8
    assert receipt['request_weight_sum']==16
    assert select(rows(),317,1)[1]['source_ids']!=receipt['source_ids']
    for i in range(0,32,4):assert [r['variant'] for r in selected[i:i+4]]==list(VARIANTS)


def test_partial_view_never_becomes_new_source():
    items=rows();items.pop()
    with pytest.raises(ValueError):select(items,317,0)


def evidence():
    item=rows()[1];identity=dict(policy_id='p',request=item['request'].model_input(),inputs={'hash':'real'})
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    record=dict(id=item['gate_id'],request_id=digest,request_identity=identity,
        candidate_logprobs=[[-math.log(len(q.values))]*len(q.values) for q in item['request'].questions])
    return item,identity,record,copy.deepcopy(record)


def test_complete_transformed_behavior_matches():
    item,identity,hf,vl=evidence()
    probs,audit=bind_behavior(item,hf['request_id'],identity,hf,vl,'p')
    assert audit['passed'] and probs==vl['candidate_logprobs']


@pytest.mark.parametrize('bad',['native','policy','input','identity','support','probability'])
def test_rejects_original_template_or_false_evidence(bad):
    item,identity,hf,vl=evidence()
    if bad=='native':vl['id']=item['source_id']+'::Q::native'
    if bad=='policy':identity['policy_id']='other'
    if bad=='input':identity['request']['audio']='different'
    if bad=='identity':vl['request_id']='different'
    if bad=='support':vl['candidate_logprobs'].pop()
    if bad=='probability':vl['candidate_logprobs'][0][0]=0.
    with pytest.raises(ValueError):bind_behavior(item,hf['request_id'],identity,hf,vl,'p')


def test_prepare_transports_external_targets_without_encoding_them(monkeypatch):
    import torch
    import multimodal_jev.intent_binding_inputs as module
    item,identity,hf,vl=evidence();seen=[]
    def encoded(row,processor,policy):
        seen.append(row['request'].model_input())
        return dict(request_id=hf['request_id'],policy_id=policy,inputs={},positions=None,candidates=[]),identity
    monkeypatch.setattr(module,'encode',encoded)
    prepared,receipt=module.prepare_item(item,None,'p',hf,vl)
    assert prepared['targets'].dtype==torch.long
    assert prepared['targets'].tolist()==[item['target']['targets']]
    assert prepared['request_weight']==item['target']['request_weight']
    assert receipt['parity']['passed']
    assert len(seen)==1 and 'targets' not in seen[0]
    changed=copy.deepcopy(item);changed['target']['targets'][0]=(changed['target']['targets'][0]+1)%60
    with pytest.raises(ValueError,match='reviewed semantic'):
        module.prepare_item(changed,None,'p',hf,vl)


def test_encode_rejects_changed_audio_before_processor(monkeypatch):
    import multimodal_jev.intent_binding_inputs as module
    monkeypatch.setattr(module,'file_hash',lambda path:'changed')
    with pytest.raises(ValueError,match='audio changed'):
        module.encode(rows()[0],None,'p')
