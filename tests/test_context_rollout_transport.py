"""Input/response boundary checks; fake engine is not backend acceptance."""
from types import SimpleNamespace as N

import pytest
import torch

from multimodal_jev.context_resident_rollout import FixedContextRollout


def fixture():
    owner = object.__new__(FixedContextRollout)
    owner.bundle = dict(policy_id='a'*64,contract=dict(token=dict(id=99)))
    owner.max_model_len = 4096
    owner.max_decisions = 2
    owner.lora = object()
    owner.phase = 'ready'
    ids = torch.tensor([[1,98,98,2,99,3,99]])
    inputs = dict(input_ids=ids,attention_mask=torch.ones_like(ids),
        input_features=torch.arange(24,dtype=torch.float32).reshape(1,3,8),
        input_features_mask=torch.tensor([[1,1,1,1,1,0,0,0]]))
    calls = []
    def encode(prompts,**kwargs):
        calls.append((prompts,kwargs))
        return [N(prompt_token_ids=ids[0].tolist(),outputs=N(data=torch.tensor([[.2],[.8]])))]
    owner.engine = N(encode=encode)
    return owner,inputs,torch.tensor([[4,6]]),calls


def test_compact_audio_identity_and_native_lora_are_bound():
    owner,inputs,pos,calls = fixture()
    result = owner._score(inputs,pos,98,N)
    prompts,kwargs = calls[0]
    assert prompts[0]['prompt_token_ids']==[1,98,2,99,3,99]
    audio = prompts[0]['multi_modal_data']['audio']
    assert torch.equal(audio['input_audio_features'],inputs['input_features'][0,:,:5])
    assert audio['audio_feature_lengths'].tolist()==[5]
    assert kwargs['lora_request'] is owner.lora
    assert kwargs['pooling_params'].extra_kwargs==dict(policy_id='a'*64,
        marker_positions=[4,6],expected_input_ids=inputs['input_ids'][0].tolist())
    assert prompts[0]['multi_modal_uuids']['audio']==[result['request_id']]
    modified = {k:v.clone() for k,v in inputs.items()}
    modified['input_features'][0,0,0]+=1
    assert owner._score(modified,pos,98,N)['request_id']!=result['request_id']


@pytest.mark.parametrize('fault',['label','markers','mask','frames','features','budget'])
def test_bad_input_rejected_before_engine(fault):
    owner,inputs,pos,calls = fixture()
    if fault=='label':inputs['labels']=torch.tensor([0])
    if fault=='markers':pos[0,1]=5
    if fault=='mask':inputs['attention_mask'][0,0]=0
    if fault=='frames':inputs['input_features_mask'][0,1]=0
    if fault=='features':inputs['input_features'][0,0,0]=float('nan')
    if fault=='budget':owner.max_decisions=1
    with pytest.raises(ValueError):owner._score(inputs,pos,98,N)
    assert not calls


@pytest.mark.parametrize('fault',['prompt','shape','nan','range'])
def test_bad_backend_response_rejected(fault):
    owner,inputs,pos,_ = fixture()
    ids = inputs['input_ids'][0].tolist()
    prob = torch.tensor([[.2],[.8]])
    if fault=='prompt':ids[0]=7
    if fault=='shape':prob=prob[:,0]
    if fault=='nan':prob[0,0]=float('nan')
    if fault=='range':prob[0,0]=1.1
    owner.engine=N(encode=lambda *a,**kw:[N(prompt_token_ids=ids,outputs=N(data=prob))])
    with pytest.raises(ValueError):owner._score(inputs,pos,98,N)
