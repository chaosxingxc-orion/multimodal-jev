import json,math,copy
import pytest
from test_slurp_intent_structure import fixture
from multimodal_jev.slurp_intent_structure import training_targets
from multimodal_jev.intent_policy_metrics import score


def inputs():
    task,example=fixture();request,target=training_targets(example,task,'P');target['request_weight']=1.
    item=dict(id=example.example_id,request=request,target=target)
    row=dict(id=item['id']+'::Q',request_identity={'request':json.loads(json.dumps(request.model_input()))},
        candidate_logprobs=[[math.log(.8),math.log(.2)],[math.log(.1),math.log(.9)],[math.log(.1),math.log(.8),math.log(.1)]])
    return item,row


def test_final_and_compatibility_are_distinct():
    item,row=inputs();r=score([item],[row],'Q')
    assert r['final_accuracy']==1 and r['component_final_compatibility']==1
    assert r['observed_classes']==1 and r['candidate_classes']==3
    row['candidate_logprobs'][0].reverse();r=score([item],[row],'Q')
    assert r['final_accuracy']==1 and r['component_final_compatibility']==0
    assert r['final_expected_reward']==pytest.approx(.8)


def test_ties_and_missing_rows_are_not_hidden():
    item,row=inputs();row['candidate_logprobs'][-1]=[-math.log(3)]*3
    assert score([item],[row],'Q')['final_accuracy']==0
    with pytest.raises(ValueError):score([item],[],'Q')
    row['request_identity']['request']['requirements']=['leaked answer']
    with pytest.raises(ValueError,match='identity'):score([item],[row],'Q')
