import copy
from types import SimpleNamespace

import pytest
import torch

from tools.reports.context_language_stage_capture import attach,compare
from tools.reports.context_language_stage_probe import namespace


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__();self.q_size=4;self.kv_size=2
        self.qkv_proj=torch.nn.Linear(4,8,bias=False)
        self.o_proj=torch.nn.Identity()

    def forward(self,positions,hidden):
        self.qkv_proj(hidden)
        return self.o_proj(hidden)


class Language(torch.nn.Module):
    def __init__(self):
        super().__init__();layer=torch.nn.Module()
        layer.input_layernorm=torch.nn.Identity();layer.post_attention_layernorm=torch.nn.Identity()
        layer.self_attn=Attention();layer.mlp=torch.nn.Module();layer.mlp.down_proj=torch.nn.Identity()
        self.layers=torch.nn.ModuleList([layer]);self.norm=torch.nn.Identity()

    def forward(self,x,positions):
        layer=self.layers[0];x=layer.input_layernorm(x)
        x=layer.self_attn(positions=positions,hidden=x)
        x=layer.post_attention_layernorm(x);x=layer.mlp.down_proj(x)
        return self.norm(x)


def test_real_hooks_preserve_output_split_qkv_and_remove():
    model=Language();x=torch.randn(7,4);positions=torch.arange(7)
    expected=model(x,positions).clone()
    hooks,values=attach(model,backend='vllm')
    actual=model(x,positions);assert torch.equal(actual,expected)
    projected=model.layers[0].self_attn.qkv_proj(x)
    assert torch.equal(values['layer_00.v'][0],projected[:,6:])
    actual.add_(100)
    assert torch.equal(values['final_norm'][0],expected)
    for hook in hooks:hook.remove()
    model(x,positions)
    assert len(values['final_norm'])==1 and len(values['positions'])==1


def tensors():
    names=['layer_00.input_norm','layer_00.q','layer_00.k','layer_00.v',
           'layer_00.attention_output','layer_00.post_attention_norm','layer_00.mlp_output',
           'layer_01.input_norm','final_norm']
    refs=[{name:[torch.randn(1,n,4,dtype=torch.bfloat16)] for name in names} for n in (3,5)]
    batch={name:[torch.cat([r[name][0][0] for r in refs])] for name in names}
    batch['positions']=[torch.cat([torch.arange(3),torch.arange(5)])]
    return refs,batch


def test_independent_stage_diff_finds_single_bf16_perturbation():
    refs,batch=tensors()
    assert compare(refs,batch,lengths=[3,5])['all_captured_stages_exact']
    batch['layer_00.q'][0][4,1]+=1
    batch['layer_01.input_norm'][0][4,2]+=1
    result=compare(refs,batch,lengths=[3,5])
    assert result['stages']['layer_00.q']['count']==1
    assert result['first_normalized_hidden_difference']=='layer_01.input_norm'
    assert not result['full_model_parity_verified']


@pytest.mark.parametrize('fault',['missing','order','length','nan','width'])
def test_incomplete_reordered_or_invalid_trace_fails(fault):
    refs,batch=tensors();lengths=[3,5]
    if fault=='missing':batch.pop('final_norm')
    if fault=='order':batch['positions']=[torch.arange(8)]
    if fault=='length':lengths=[4,4]
    if fault=='nan':batch['layer_00.q'][0][0,0]=float('nan')
    if fault=='width':batch['layer_00.q']=[torch.ones(8,3)]
    with pytest.raises(ValueError):compare(refs,batch,lengths=lengths)


def test_generated_phase_uses_only_language_hooks_and_same_shared_runner():
    ns=namespace()
    constants=str(ns['phase'].__code__.co_consts)
    assert 'context_language_stage_worker' in constants
    assert 'context_audio_stage_worker' not in constants
    assert ns['compare'].__module__=='tools.reports.context_language_stage_probe'
