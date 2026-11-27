import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from multimodal_jev.card_contrast import Directions, bind_sample_counts, components
from multimodal_jev.card_training_acoustics import collate
from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.context_decision_model import ContextDecisionModel
from multimodal_jev.context_row_head import install_row_head


def objective():
    cfg = json.loads(Path('configs/portable/operations/card-stage1-speech-alignment-211.json').read_bytes())
    bind_sample_counts(cfg)
    return cfg['window']['triplet_objective']


def test_three_softmaxes_and_probability_units_against_independent_classification():
    obj = objective()
    obj['speech'].update(fixed_card_temperature=.25, speech_temperature=.5)
    d = Directions.from_objective(obj)
    z = torch.tensor([[5., 4., 3., 2., 2., 3., 4., 5.]], requires_grad=True)
    torch.manual_seed(42)
    e = torch.randn(1, 8, 4, requires_grad=True)
    total, card, audio, speech, margin = components(z, d, e).flatten()
    target = torch.zeros(1, dtype=torch.long)
    ce = lambda values, tau: F.cross_entropy(values/tau, target)
    rr = torch.stack([ce(z[:, [j,4,5,6]],.25) for j in [0,7]]).mean()
    aa = ce(z[:, :4], .1)
    ss = ce(F.cosine_similarity(e[:, :1], e[:, [7,4,5,6]], dim=-1), .5)
    # Raw-logit gap is one, but deployment probability gap is only ~.011.
    mm = .3 - (torch.sigmoid(z[:,0]) - torch.sigmoid(z[:,1])).squeeze()
    torch.testing.assert_close(torch.stack([card,audio,speech,margin]), torch.stack([aa,rr,ss,mm]))
    torch.testing.assert_close(total, .5*(aa+rr)+d.speech.speech_weight*ss+d.speech.margin_weight*mm)
    zz = z.detach().clone(); zz[:,[0,7]]=2.; zz[:,1:7]=-2.
    assert components(zz,d,e)[0,-1].item()==0
    with pytest.raises(ValueError):
        components(z,d)
    for bogus in (0., float('nan')):
        with pytest.raises(ValueError):
            Directions.from_objective(dict(obj, speech=dict(obj['speech'],speech_temperature=bogus)))


class CountingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_tower = nn.Identity()
        self.multi_modal_projector = nn.Identity()
        self.embed = nn.Embedding(32,4)
        self.calls = []
        self.language_calls = 0

    def get_audio_features(self, x, mask, return_dict):
        self.calls.append(len(x))
        return SimpleNamespace(pooler_output=torch.cat([v[:,:int(m.sum()):8].T for v,m in zip(x,mask)]))

    def get_input_embeddings(self):
        return self.embed

    def get_placeholder_mask(self, ids, embeds, audio):
        return (ids==8).unsqueeze(-1)

    def forward(self, inputs_embeds, **kwargs):
        self.language_calls += 1
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1))


def test_joint_forward_full_context_card_invariance_audio_swap_and_real_gradients():
    torch.manual_seed(307)
    backbone = CountingBackbone().eval()
    head = ContextDecisionHead(4,2,12)
    install_row_head(head)
    model = ContextDecisionModel(backbone,head,9)
    owners = [torch.randn(1,4,100) for _ in range(5)]
    masks = [(torch.arange(100)<n)[None].long() for n in (8,16,8,16,8)]
    order = (0,0,0,0,1,2,3,4)
    def forward(cards, order, auxiliary):
        rows = []
        for i,(uid, card) in enumerate(zip(order,cards,strict=True)):
            n = (int(masks[uid].sum())+7)//8
            ids=torch.tensor([[1,card,*([8]*n),9]])
            rows.append(dict(request_id=str(i),policy_id='p',positions=torch.tensor([[ids.numel()-1]]),
                             inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                                         input_features=owners[uid],input_features_mask=masks[uid])))
        inputs,pos=collate(rows,'p',speech_features=auxiliary,audio_token_id=8,decision_token_id=9,
                           pad_token_id=0,audio_chunk_size=100,audio_infer_window=100)
        return model(inputs,pos,return_logits=True,return_speech=auxiliary)
    cards = (10,11,12,13,10,10,10,10)
    logits, e = forward(cards, order, True)
    assert backbone.calls == [5] and backbone.language_calls==1
    torch.testing.assert_close(logits,forward(cards, order, False),rtol=0,atol=0)
    swapped, unchanged = forward(tuple(14+i for i in range(8)),order,True)
    assert not torch.equal(logits,swapped)
    torch.testing.assert_close(unchanged,e,rtol=0,atol=0)
    _, switched = forward(cards,(4,4,4,4,1,2,3,0),True)
    torch.testing.assert_close(switched[0], e[7])
    torch.testing.assert_close(switched[7], e[0])
    assert not torch.allclose(e[0], e[7])
    terms=components(logits.reshape(1,8),Directions.from_objective(objective()),e.reshape(1,8,-1))
    shared = list(head.ffn.parameters())
    aux_grad = torch.autograd.grad(terms[0,3],shared,retain_graph=True)
    member_grad = torch.autograd.grad(terms[0,1],shared)
    assert all(g.isfinite().all() and g.abs().max()>0 for g in (*aux_grad,*member_grad))
