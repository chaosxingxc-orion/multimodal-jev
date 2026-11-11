from types import SimpleNamespace
import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.span_model import AudioSpanModel, span_supervised_loss
from multimodal_jev.span_contract import SpanRequest, SpanQuestion
from multimodal_jev.task_schema import DecisionSchema


class CausalBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.emb = torch.nn.Embedding(12,4); self.calls = 0
    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        self.calls += 1
        assert use_cache is False
        return SimpleNamespace(last_hidden_state=(self.emb(input_ids)*attention_mask[...,None]).cumsum(1))


def test_one_forward_and_final_gradient_reaches_earlier_latent_position():
    torch.manual_seed(17)
    core = CausalBackbone(); model = AudioSpanModel(core,4,3)
    inputs = {'input_ids':torch.tensor([[1,2,3,4,5,6]]),'attention_mask':torch.ones(1,6,dtype=torch.long)}
    logits = model(inputs,torch.tensor([[1,3,5]]),torch.tensor([[2,3,2]]))
    assert core.calls == 1 and logits.shape == (1,3,3)
    assert torch.isneginf(logits[0,0,2]) and torch.isneginf(logits[0,2,2])
    loss = span_supervised_loss(logits,torch.tensor([[-999,-999,1]]),torch.tensor([[False,False,True]]))
    loss.backward()
    assert core.emb.weight.grad[2].abs().sum() > 0
    # This synthetic causal fixture verifies wiring, not real ASR CoT quality.


def test_padding_and_invalid_target_fail_closed():
    model = AudioSpanModel(CausalBackbone(),4,3)
    with pytest.raises(ValueError,match='padding'):
        model({'input_ids':torch.tensor([[1,2]]),'attention_mask':torch.tensor([[1,0]])},
              torch.tensor([[1]]),torch.tensor([[2]]))
    with pytest.raises(ValueError,match='masked class'):
        span_supervised_loss(torch.tensor([[[1.,2.,-torch.inf]]]),torch.tensor([[2]]),torch.tensor([[True]]))


def test_request_has_no_targets_and_typed_value_order():
    q = SpanQuestion('final',DecisionSchema('s','noul','Is condition met?',allow_unknown=True))
    r = SpanRequest('/a.wav',(q,),'final',('Use audio evidence.',))
    assert q.values == (False,True,None) and 'target' not in str(r.model_input())
    with pytest.raises(ValueError,match='last span'):
        SpanRequest('/a.wav',(q,),'other',('Use audio.',))


def test_native_head_same_positions_no_extra_backbone_call():
    from multimodal_jev.span_model import NativeAudioSpanModel
    core = CausalBackbone(); head = torch.nn.Linear(4,12,bias=False)
    model = NativeAudioSpanModel(SimpleNamespace(model=core,lm_head=head))
    inputs={'input_ids':torch.tensor([[1,2,3,4,5,6]]),'attention_mask':torch.ones(1,6,dtype=torch.long)}
    vocab, logp=model(inputs,torch.tensor([[1,3,5]]),[[2,4],[3,5,7],[1,9]])
    assert core.calls == 1
    expected=head(core.emb(inputs['input_ids']).cumsum(1)[:,[1,3,5]]).float()
    torch.testing.assert_close(vocab,expected)
    for p in logp:torch.testing.assert_close(p.exp().sum(-1),torch.ones(1))
    loss=span_supervised_loss(vocab,torch.tensor([[2,-999,9]]),torch.tensor([[True,False,True]]))
    loss.backward()
    assert core.emb.weight.grad[2].abs().sum() > 0


def test_marker_predecessors_after_audio_expansion_and_template_loss():
    from multimodal_jev.span_contract import encode_span_request
    class Tokenizer:
        def encode(self,text,**kwargs):return [ord(c) for c in text]
        def __call__(self,text,**kwargs):
            return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}
        def convert_tokens_to_ids(self,token):return 999
    class Processor:
        tokenizer=Tokenizer()
        drop=False
        def apply_chat_template(self,messages,**kwargs):
            # Mimic ASR: user text discarded, assistant context preserved.
            return messages[0]['content'][0]['text']+'@'+('' if self.drop else messages[2]['content'])
        def __call__(self,text,**kwargs):
            ids=[]
            for c in text[0]:ids.extend([999]*7 if c=='@' else [ord(c)])
            return {'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long)}
    p=Processor()
    questions=tuple(SpanQuestion(s,DecisionSchema(s,'noul','Question '+s,allow_unknown=False)) for s in ('q1','final'))
    request=SpanRequest('/a.wav',questions,'final',('Use audio.',))
    batch,pos=encode_span_request(p,request,[0.])
    for i,j in enumerate(pos[0].tolist()):
        following=batch['input_ids'][0,j+1:j+1+len(f'[JEV_SPAN_{i+1}]')].tolist()
        assert following==p.tokenizer.encode(f'[JEV_SPAN_{i+1}]')
    assert (batch['input_ids'][0,:pos[0,0]]==999).sum()==7
    p.drop=True
    with pytest.raises(ValueError,match='dropped task'):encode_span_request(p,request,[0.])


def test_candidate_ce_matches_manual_masked_nll_and_ignores_non_candidates():
    from multimodal_jev.span_model import span_candidate_loss
    logits=torch.tensor([[[.2,99.,.7,-.1],[.3,-.4,88.,.2]],
                         [[.8,77.,-.1,.3],[.2,.5,66.,-.7]]],requires_grad=True)
    options=[[0,2],[1,3]]
    logp=[logits[:,j,ids].log_softmax(-1) for j,ids in enumerate(options)]
    targets=torch.tensor([[2,-100],[0,3]])
    mask=torch.tensor([[True,False],[True,True]])
    loss=span_candidate_loss(logp,targets,mask,options)
    expected=(-logp[0][0,1]-logp[0][1,0]-logp[1][1,1])/3
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert logits.grad[0,1].abs().sum()==0
    assert logits.grad[:,0,1].abs().sum()==0
    assert logits.grad[:,1,2].abs().sum()==0
    assert logits.grad[:,0,[0,2]].abs().sum()>0
    with pytest.raises(ValueError,match='absent'):
        span_candidate_loss(logp,torch.tensor([[1,-100],[0,3]]),mask,options)
    with pytest.raises(ValueError,match='normalized'):
        span_candidate_loss([p+1 for p in logp],targets,mask,options)
