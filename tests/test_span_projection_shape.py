from types import SimpleNamespace
import pytest
import torch
from multimodal_jev.span_projection_shape import full_width_local_linear, full_width_qkv_apply


def test_padding_selects_correct_outputs_for_every_rank():
    torch.manual_seed(7)
    x=torch.randn(2,5,8,dtype=torch.bfloat16);w=torch.randn(16,8,dtype=torch.bfloat16)
    expected=torch.nn.functional.linear(x,w)
    actual=torch.cat([full_width_local_linear(x,s,rank) for rank,s in enumerate(w.chunk(4,0))],-1)
    assert torch.equal(actual,expected)


def test_qkv_retains_native_lora_and_local_order():
    x=torch.ones(3,4,dtype=torch.bfloat16)
    w=torch.arange(32,dtype=torch.bfloat16).reshape(8,4)
    seen=[]
    def add(original,base):
        assert original is x;seen.append(base);return base+1
    module=SimpleNamespace(base_layer=SimpleNamespace(weight=w),output_slices=(4,2,2),
        output_ids=(2,2,2),_jev_full_width_calls=0,_apply_lora_to_output=add)
    result=full_width_qkv_apply(module,x)
    assert module._jev_full_width_calls==1 and len(seen)==1
    assert torch.equal(result,torch.nn.functional.linear(x,w)+1)
    with pytest.raises(ValueError,match='unbiased'):
        full_width_qkv_apply(module,x,torch.zeros(8))


@pytest.mark.parametrize('rank,size,dtype', [(4,4,torch.bfloat16),(True,4,torch.bfloat16),(0,2,torch.bfloat16),(0,4,torch.float32)])
def test_invalid_shape_execution_rejected(rank,size,dtype):
    with pytest.raises(ValueError):full_width_local_linear(torch.ones(3,4,dtype=dtype),torch.ones(8,4,dtype=dtype),rank,size)
