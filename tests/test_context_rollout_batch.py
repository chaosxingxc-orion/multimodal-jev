import copy,hashlib,json
import pytest
import torch
from multimodal_jev.context_rollout_batch import bind_rollout
from multimodal_jev.span_request_identity import tensor_identity


def fixture():
    inputs={'input_ids':torch.tensor([[3,9,9]]),'input_features':torch.ones(1,2,3)}
    positions=torch.tensor([[1,2]])
    identity=dict(policy_id='policy',positions=[1,2],inputs={k:tensor_identity(v) for k,v in sorted(inputs.items())})
    rid=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(policy_id='policy',request_id=rid,request_identity=identity,prob=[.25,.75]),inputs,positions


def test_complete_group_reproducible_without_audio_reexecution():
    record,inputs,positions=fixture()
    a=bind_rollout(record,inputs,positions,policy_id='policy',group_size=16,seed=317)
    b=bind_rollout(record,inputs,positions,policy_id='policy',group_size=16,seed=317)
    assert torch.equal(a['actions'],b['actions']) and a['actions'].shape==(1,16,2)
    for p in a['behavior']:torch.testing.assert_close(p.exp().sum(-1),torch.ones(1))
    assert a['receipt']==b['receipt']


@pytest.mark.parametrize('fault',['policy','audio','positions','rid','saturated','nan'])
def test_no_stale_policy_input_or_silent_probability_clamp(fault):
    record,inputs,positions=fixture();record=copy.deepcopy(record)
    if fault=='policy':record['policy_id']='old'
    if fault=='audio':inputs['input_features'][0,0,0]=2
    if fault=='positions':positions[0,0]=0
    if fault=='rid':record['request_id']='unbound'
    if fault=='saturated':record['prob'][0]=1.
    if fault=='nan':record['prob'][0]=float('nan')
    with pytest.raises(ValueError):bind_rollout(record,inputs,positions,policy_id='policy',group_size=16,seed=317)
