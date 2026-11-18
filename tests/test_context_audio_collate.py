import copy
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.context_audio_collate import collate_context_audio,forward_context_audio_batch
from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.context_decision_model import ContextDecisionModel


CONFIG = dict(decision_token_id=99,policy_id='policy',pad_token_id=0,audio_token_id=98,
              audio_chunk_size=100,audio_infer_window=200)


def requests():
    result = []
    for i,frames in enumerate([100,190]):
        count = 13 if frames==100 else 25
        ids = torch.tensor([[1]+[98]*count+[2,99,3,99]])
        width = ((frames+99)//100)*100
        features = torch.randn(1,4,width,dtype=torch.float64)
        mask = (torch.arange(width)<frames).long()[None]
        result.append(dict(request_id=str(i),policy_id='policy',inputs=dict(input_ids=ids,
            attention_mask=torch.ones_like(ids),input_features=features,input_features_mask=mask),
            positions=torch.tensor([[count+2,count+4]])))
    return result


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(100,8)
        self.audio = torch.nn.Linear(4,8,bias=False)
        self.calls = 0

    def forward(self,input_ids,attention_mask,input_features,input_features_mask,
                position_ids,use_cache,return_dict):
        self.calls += 1
        assert not use_cache and return_dict
        masked = input_features*input_features_mask[:,None]
        audio = self.audio(masked.sum(-1)/input_features_mask.sum(-1)[:,None])
        hidden = (self.embedding(input_ids)+audio[:,None])*attention_mask[:,:,None]
        return SimpleNamespace(last_hidden_state=hidden.cumsum(1))


def test_one_forward_matches_serial_logits_and_gradients_with_unequal_audio():
    torch.manual_seed(317)
    items = requests()
    model = ContextDecisionModel(Backbone(),ContextDecisionHead(8,2,16),99).double()
    serial = copy.deepcopy(model)
    batch = forward_context_audio_batch(model,items,device='cpu',dtype=torch.float64,**CONFIG)
    refs = torch.cat([forward_context_audio_batch(serial,[item],device='cpu',dtype=torch.float64,**CONFIG)
                      for item in items])
    assert model.backbone.calls==1 and serial.backbone.calls==2
    torch.testing.assert_close(batch,refs,atol=1e-12,rtol=1e-12)
    coefficients = torch.tensor([[.3,-.2],[-.7,.5]],dtype=torch.float64)
    (batch*coefficients).sum().backward();(refs*coefficients).sum().backward()
    for p,q in zip(model.parameters(),serial.parameters(),strict=True):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad,q.grad,atol=1e-12,rtol=1e-12)
    packed,pos = collate_context_audio(items,**CONFIG)
    for i,item in enumerate(items):
        length = item['inputs']['input_ids'].shape[-1]
        assert torch.equal(packed['input_ids'][i,:length],item['inputs']['input_ids'][0])
        assert not packed['attention_mask'][i,length:].any()
        assert torch.equal(pos[i],item['positions'][0])


def test_batch_logits_enters_wrapper_forward():
    class Wrapper(torch.nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child
            self.calls = 0

        def forward(self, inputs, positions, *, return_logits):
            self.calls += 1
            assert return_logits is True
            return self.child(inputs, positions, return_logits=return_logits)

    model=Wrapper(ContextDecisionModel(Backbone(),ContextDecisionHead(8,2,16),99).double())
    result=forward_context_audio_batch(model,requests(),device='cpu',dtype=torch.float64,**CONFIG)
    assert model.calls == 1 and result.shape == (2,2)
    result.sum().backward()
    assert model.child.decision_head.projection.weight.grad is not None


@pytest.mark.parametrize('fault',['policy','duplicate','marker','missing','unequal','label','audio_count'])
def test_incompatible_or_label_bearing_requests_rejected(fault):
    items = requests()
    if fault=='policy': items[1]['policy_id']='stale'
    if fault=='duplicate': items[1]['request_id']='0'
    if fault=='marker': items[1]['positions'][0,0]-=1
    if fault=='missing': items[1]['inputs']['input_ids'][0,1]=99
    if fault=='unequal': items[1]['positions']=items[1]['positions'][:,:1]
    if fault=='label': items[1]['gold']=1
    if fault=='audio_count':
        # Features imply seven audio tokens, but the prompt still has thirteen.
        x=items[0];x['inputs']['input_features_mask'][:,50:]=0
    with pytest.raises(ValueError): collate_context_audio(items,**CONFIG)
