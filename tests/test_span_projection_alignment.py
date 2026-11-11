from types import SimpleNamespace
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_projection_alignment import split_qkv_base,split_base_native_lora_apply


def test_separate_projection_preserves_gradients_and_bias_slice_mapping():
    torch.manual_seed(17)
    x=torch.randn(2,7,4,requires_grad=True)
    w=torch.randn(10,4,requires_grad=True)
    b=torch.randn(10,requires_grad=True)
    y=split_qkv_base(x,w,(6,2,2),b)
    expected=torch.nn.functional.linear(x,w,b)
    torch.testing.assert_close(y,expected)
    gradients=torch.autograd.grad(y.square().sum(),(x,w,b),retain_graph=True)
    reference=torch.autograd.grad(expected.square().sum(),(x,w,b))
    for a,c in zip(gradients,reference):torch.testing.assert_close(a,c)
    with pytest.raises(ValueError,match='aligned'):
        split_qkv_base(x,w,(4,4,4))


def test_native_lora_add_receives_original_input_and_split_base():
    x=torch.ones(3,2);w=torch.arange(12.).reshape(6,2)
    calls=[]
    def add(original,output):
        assert original is x
        calls.append(output.clone())
        return output+7
    module=SimpleNamespace(base_layer=SimpleNamespace(weight=w),output_slices=(2,2,2),_apply_lora_to_output=add)
    actual=split_base_native_lora_apply(module,x)
    assert len(calls)==1
    torch.testing.assert_close(actual,torch.nn.functional.linear(x,w)+7)


def test_mlp_uses_bf16_activation_materialization():
    from multimodal_jev.span_projection_alignment import separate_mlp_forward
    torch.manual_seed(31)
    x=torch.randn(5,8,dtype=torch.bfloat16)
    w=torch.randn(24,8,dtype=torch.bfloat16)
    module=SimpleNamespace(gate_up_proj=SimpleNamespace(weight=w),down_proj=lambda y:(y,None))
    actual=separate_mlp_forward(module,x)
    gate=torch.nn.functional.linear(x,w[:12])
    up=torch.nn.functional.linear(x,w[12:])
    expected=torch.nn.functional.silu(gate)*up
    assert torch.equal(actual,expected)
    fused=(torch.nn.functional.silu(gate.float())*up.float()).to(torch.bfloat16)
    assert not torch.equal(actual,fused)


def test_qkv_partition_must_slice_each_projection_before_packing():
    torch.manual_seed(29)
    x=torch.randn(5,8); w=torch.randn(32,8); b=torch.randn(32)
    widths=(16,8,8)
    reference=split_qkv_base(x,w,widths,b).split(widths,-1)
    local=[]
    for rank in range(4):
        shard=torch.cat([v.chunk(4,0)[rank] for v in w.split(widths,0)],0)
        bias=torch.cat([v.chunk(4,0)[rank] for v in b.split(widths,0)],0)
        local.append(split_qkv_base(x,shard,(4,2,2),bias).split((4,2,2),-1))
    for i,expected in enumerate(reference):
        torch.testing.assert_close(torch.cat([s[i] for s in local],-1),expected)
