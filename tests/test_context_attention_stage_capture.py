import sys
from types import ModuleType

import pytest
import torch

from tools.reports.context_attention_stage_capture import attach
from tools.reports.context_attention_stage_probe import namespace


class Rotary(torch.nn.Module):
    def forward(self,positions,q,k):return q+1,k+1


class Attention(torch.nn.Module):
    def __init__(self,rotary):
        super().__init__();self.q_size=4;self.kv_size=2
        self.qkv_proj=torch.nn.Linear(4,8,bias=False)
        self.q_norm=torch.nn.Identity();self.k_norm=torch.nn.Identity()
        self.rotary_emb=rotary;self.o_proj=torch.nn.Identity()
    def forward(self,positions,hidden):
        q,k,v=self.qkv_proj(hidden).split([4,2,2],-1)
        q=self.q_norm(q.reshape(-1,2,2)).reshape(-1,4)
        k=self.k_norm(k.reshape(-1,1,2)).reshape(-1,2)
        q,k=self.rotary_emb(positions,q,k)
        return self.o_proj(q)


class Language(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.ModuleList();rotary=Rotary()
        for _ in range(2):
            l=torch.nn.Module();l.input_layernorm=torch.nn.Identity()
            l.post_attention_layernorm=torch.nn.Identity();l.self_attn=Attention(rotary)
            l.mlp=torch.nn.Module();l.mlp.down_proj=torch.nn.Identity();self.layers.append(l)
        self.norm=torch.nn.Identity()
    def forward(self,x,positions):
        for l in self.layers:
            x=l.input_layernorm(x);x=l.self_attn(positions,x)
            x=l.post_attention_layernorm(x);x=l.mlp.down_proj(x)
        return self.norm(x)


def test_shared_rotary_only_records_first_layer_and_hooks_do_not_change_outputs():
    model=Language();x=torch.randn(7,4);pos=torch.arange(7)
    expected=model(x,pos);hooks,values=attach(model,backend='vllm')
    actual=model(x,pos)
    assert torch.equal(expected,actual)
    for name in ('q_norm','k_norm','q_rope','k_rope','kernel_output'):
        assert len(values['layer_00.'+name])==1
        assert values['layer_00.'+name][0].shape[0]==7
    assert torch.equal(values['layer_00.q_rope'][0],values['layer_00.q_norm'][0]+1)
    assert torch.equal(values['layer_00.kernel_output'][0],values['layer_00.q_rope'][0])
    for h in hooks:h.remove()
    model(x,pos)
    assert len(values['layer_00.q_rope'])==1


def test_hf_capture_preserves_kernel_and_token_major_layout(monkeypatch):
    # Explicit backend fixture: real hooks/wrapper, no transformers/NPU compute.
    qwen=ModuleType('transformers.models.qwen3.modeling_qwen3')
    calls=[]
    def original(module,q,k,v,*a,**kw):calls.append(module);return q,None
    qwen.eager_attention_forward=original
    parents=['transformers','transformers.models','transformers.models.qwen3']
    for name in parents:monkeypatch.setitem(sys.modules,name,ModuleType(name))
    for name in parents[1:]+[qwen.__name__]:
        obj=qwen if name==qwen.__name__ else sys.modules[name]
        monkeypatch.setattr(sys.modules[name.rsplit('.',1)[0]],name.rsplit('.',1)[1],obj,raising=False)
    monkeypatch.setitem(sys.modules,qwen.__name__,qwen)
    model=Language();att=model.layers[0].self_attn
    for n,width in [('q',4),('k',2),('v',2)]:setattr(att,n+'_proj',torch.nn.Linear(4,width))
    hooks,values=attach(model,backend='hf')
    q=torch.randn(1,2,7,2);k=torch.randn(1,1,7,2)
    out=qwen.eager_attention_forward(att,q,k,k)
    assert out[0] is q and calls==[att]
    assert torch.equal(values['layer_00.q_rope'][0],q.transpose(1,2)[0].reshape(7,4))
    qwen.eager_attention_forward(model.layers[1].self_attn,q,k,k)
    assert len(values['layer_00.q_rope'])==1
    for h in hooks:h.remove()
    assert qwen.eager_attention_forward is original


def test_phase_uses_new_hooks_and_same_existing_runner():
    ns=namespace();constants=str(ns['phase'].__code__.co_consts)
    assert 'context_attention_stage_worker' in constants
    assert 'context_audio_stage_worker' not in constants
    assert ns['compare'].__module__=='tools.reports.context_language_stage_probe'
