from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.omni_compact_attention import compact_attention,compact_text_attention


def sdpa(module,q,k,v,mask,**kwargs):
    return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,
        dropout_p=kwargs.get('dropout',0.)).transpose(1,2),None


def inputs():
    g=torch.Generator().manual_seed(7)
    tensors=[torch.randn((2,2,7,4),generator=g,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    mask=torch.zeros((2,1,7,7),dtype=torch.bool)
    # Different noncontiguous excluded keys per row; padded query rows can be
    # all false. This prevents a left-slice-only implementation from passing.
    for row,indices in enumerate(([1,3,4,6],[0,2,3])):
        for i,qpos in enumerate(indices):mask[row,0,qpos,indices[:i+1]]=True
    return *tensors,mask


def test_attention_edges_outputs_and_gradients_preserved():
    q,k,v,mask=inputs();module=SimpleNamespace(layer_idx=0);records=[]
    expected=sdpa(module,q,k,v,mask)[0]
    actual=compact_attention(sdpa,module,q,k,v,mask,records=records)[0]
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    ref_grad=torch.autograd.grad(expected.square().sum(),(q,k,v),retain_graph=True)
    got_grad=torch.autograd.grad(actual.square().sum(),(q,k,v))
    for a,b in zip(got_grad,ref_grad):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    assert records[0]['key_lengths']==[4,3]
    for row in range(2):
        excluded=~mask[row].any(dim=(0,1))
        assert torch.count_nonzero(got_grad[1][row,:,excluded])==0
        assert torch.count_nonzero(got_grad[2][row,:,excluded])==0


@pytest.mark.parametrize('fault',['float_mask','heads','empty','dropout','cache','causal_override','position_bias'])
def test_unsupported_attention_fails_explicitly(fault):
    q,k,v,mask=inputs();kw={}
    if fault=='float_mask':mask=mask.double()
    if fault=='heads':mask=mask.expand(2,2,7,7)
    if fault=='empty':mask.zero_()
    if fault=='dropout':kw['dropout']=.1
    if fault=='cache':q=q[:,:,:1];mask=mask[:,:,:1]
    if fault=='causal_override':kw['is_causal']=True
    if fault=='position_bias':kw['position_bias']=torch.zeros(1)
    with pytest.raises(ValueError):compact_attention(sdpa,SimpleNamespace(layer_idx=0),q,k,v,mask,records=[],**kw)


def test_registry_restored_after_failure_and_other_attention_untouched():
    class Registry:
        fn=staticmethod(sdpa)
        def get_interface(self,*a):return self.fn
        def register(self,key,fn):self.fn=fn
    class Text:
        layer_idx=0
    registry=Registry();hf=SimpleNamespace(ALL_ATTENTION_FUNCTIONS=registry,
        eager_attention_forward=sdpa,Qwen3OmniMoeThinkerTextAttention=Text)
    q,k,v,mask=inputs()
    with pytest.raises(ValueError,match='reachable key'):
        with compact_text_attention(hf) as records:
            with pytest.raises(RuntimeError,match='nested'):
                with compact_text_attention(hf):pass
            registry.fn(object(),q,k,v,mask)
            assert records==[]
            registry.fn(Text(),q,k,v,None)
            assert records[0]['mode']=='unchanged_no_mask'
            registry.fn(Text(),q,k,v,torch.zeros_like(mask))
    assert registry.fn is sdpa
