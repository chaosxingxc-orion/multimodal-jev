import json
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.context_decision_contract import (
    ApplicabilityQuestion, ContextDecisionRequest, DECISION_TOKEN,
    encode_context_request, render_context_request,
)
from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.context_decision_model import ContextDecisionModel


def request():
    return ContextDecisionRequest('/audio.wav', (
        ApplicabilityQuestion('candidate_a', 'Rule A', 'Does rule A apply?'),
        ApplicabilityQuestion('candidate_b', 'Rule B', 'Does rule B apply?')),
        ('Use current rules.',), ('Earlier supplied context.',), version='custom-protocol-v7')


class Tokenizer:
    all_special_tokens = [DECISION_TOKEN]
    def get_vocab(self): return {DECISION_TOKEN: 998}
    def encode(self, text, **kwargs):
        ids = []
        for i, part in enumerate(text.split(DECISION_TOKEN)):
            if i: ids.append(998)
            for c in part: ids.extend([999] * 7 if c == '@' else [ord(c)])
        return ids


class Processor:
    tokenizer = Tokenizer()
    drop = False
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content'][0]['text'] + '@' + ('' if self.drop else messages[2]['content'])
    def __call__(self, text, **kwargs):
        ids = torch.tensor([self.tokenizer.encode(text[0])])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


def test_full_definitions_history_version_precede_markers_and_audio_expansion():
    r = request()
    system, content = render_context_request(r)
    header = json.loads(content.splitlines()[0])
    assert header['version'] == r.version and header['history'] == list(r.history)
    assert header['questions'] == [q.__dict__ for q in r.questions]
    assert content.index('Rule B') < content.index(DECISION_TOKEN)
    batch, pos = encode_context_request(Processor(), r, [0.], decision_token_id=998,
                                        audio_token_id=999, max_tokens=4096)
    assert (batch['input_ids'][0, pos[0]] == 998).all()
    assert (batch['input_ids'][0, :pos[0, 0]] == 999).sum() == 7
    assert pos.shape == (1, 2)


@pytest.mark.parametrize('fault', ['limit', 'id', 'drop'])
def test_no_silent_truncation_token_substitution_or_template_loss(fault):
    p = Processor()
    p.drop = fault == 'drop'
    with pytest.raises(ValueError):
        encode_context_request(p, request(), [0.], decision_token_id=997 if fault == 'id' else 998,
                               audio_token_id=999, max_tokens=5 if fault == 'limit' else 4096)


def test_request_rejects_marker_injection_and_duplicate_ids():
    with pytest.raises(ValueError, match='reserved'):
        ContextDecisionRequest('/a', request().questions, (DECISION_TOKEN,))
    with pytest.raises(ValueError, match='duplicate'):
        ContextDecisionRequest('/a', (request().questions[0],) * 2, ('rule',))
    with pytest.raises(ValueError,match='reserved'):
        ContextDecisionRequest('/a',request().questions,('rule',),('<|jev_probe_reserved_id_151705|>',))


def test_processor_cannot_inject_timestamp_reservation():
    from multimodal_jev.context_decision_contract import RESERVED_TIMESTAMP_TOKEN
    class ReservedTokenizer(Tokenizer):
        def get_vocab(self):return dict(super().get_vocab(),**{RESERVED_TIMESTAMP_TOKEN:42})
    p=Processor();p.tokenizer=ReservedTokenizer()
    original=p.apply_chat_template
    p.apply_chat_template=lambda *a,**kw:original(*a,**kw)+'*'
    with pytest.raises(ValueError,match='timestamp reservation'):
        encode_context_request(p,request(),[0.],decision_token_id=998,audio_token_id=999,max_tokens=4096)


def test_cached_inputs_cannot_bypass_timestamp_guard():
    backbone=AudioBackbone()
    model=ContextDecisionModel(backbone,ContextDecisionHead(12,3,24),151706)
    inputs=dict(input_ids=torch.tensor([[151705,151706]]),attention_mask=torch.ones(1,2,dtype=torch.long),
                input_features=torch.zeros(1,3))
    with pytest.raises(ValueError,match='timestamp reservation'):
        model(inputs,torch.tensor([[1]]))
    assert backbone.calls==0


class AudioBackbone(torch.nn.Module):
    """Synthetic differentiable audio/backbone wiring; not Qwen acceptance."""
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(20, 12)
        self.audio = torch.nn.Linear(3, 12)
        self.calls = 0
    def forward(self, input_ids, attention_mask, input_features, use_cache, return_dict):
        self.calls += 1
        assert not use_cache and return_dict
        h = self.emb(input_ids) + self.audio(input_features)[:, None, :]
        return SimpleNamespace(last_hidden_state=(h * attention_mask[..., None]).cumsum(1))


def test_single_audio_call_gradient_and_marker_self_readout():
    torch.manual_seed(317)
    backbone = AudioBackbone()
    model = ContextDecisionModel(backbone, ContextDecisionHead(12, 3, 24), 19)
    features = torch.randn(1, 3, requires_grad=True)
    inputs = dict(input_ids=torch.tensor([[1, 2, 19, 3, 19]]),
                  attention_mask=torch.ones(1, 5, dtype=torch.long), input_features=features)
    prob = model(inputs, torch.tensor([[2, 4]]))
    assert backbone.calls == 1 and prob.shape == (1, 2)
    prob[0, 1].backward()
    assert features.grad.abs().sum() > 0
    assert backbone.emb.weight.grad[19].abs().sum() > 0
    assert model.decision_head.attention.in_proj_weight.grad.abs().sum() > 0
    for positions in (torch.tensor([[1, 3]]), torch.tensor([[2]]), torch.tensor([[2, 2]])):
        with pytest.raises(ValueError): model(inputs, positions)
    with pytest.raises(ValueError, match='label-free'):
        model(dict(inputs, labels=torch.tensor([1])), torch.tensor([[2, 4]]))
    assert backbone.calls == 1
