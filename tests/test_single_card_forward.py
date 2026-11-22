import torch
from multimodal_jev import single_card_forward as mod
from multimodal_jev.context_decision_contract import render_prefix_context_request

def test_forward_isolates_cards_but_loss_uses_both(monkeypatch):
    def execute(model,rows,**kw):
        assert len(rows)==2
        assert [r['inputs']['input_ids'].tolist() for r in rows]==[[[6,9,9,7]],[[5,9,9,7]]]
        assert all(r['positions'].tolist()==[[3]] for r in rows)
        return model(torch.tensor([[1.],[-1.]]))
    monkeypatch.setattr(mod,'forward_context_audio_batch',execute)
    f=mod.IndependentForward(dict(labels=['x','y'],arms={'rules':[0,1]},catalog=[
        dict(id='x',prefix=[6],suffix=[7]),dict(id='y',prefix=[5],suffix=[7])]),{'sample':['rules','rules']})
    m=torch.nn.Linear(1,1,bias=False)
    scores=f(m,[dict(request_id='sample::epoch0-batch0-slot0',policy_id='p',inputs=dict(
        input_ids=torch.tensor([[111,9,9,222,7,7]]),input_features=torch.ones(1),input_features_mask=torch.ones(1)))],
        audio_token_id=9,decision_token_id=7)
    assert scores.shape==(1,2)
    torch.nn.functional.cross_entropy(scores,torch.tensor([0])).backward()
    assert m.weight.grad.item()<0

def test_single_card_text_contains_only_the_selected_definition():
    c=dict(version='jev-single-card-v1',instruction='Listen.',question='Does this apply?')
    a=mod.request(dict(id='music',definition='Play music.'),[],c)
    text,history,readout=render_prefix_context_request(a)
    assert text=='Listen.\nmusic: Play music.\nDoes this apply?' and not history and readout=='<|jev_decision|>'
