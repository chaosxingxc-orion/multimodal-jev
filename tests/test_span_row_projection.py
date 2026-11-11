import sys
from types import SimpleNamespace
import pytest
import torch
from multimodal_jev.span_row_projection import output_weight_shard, complete_k_forward


def test_output_partition_reconstructs_complete_matmul_and_changes_partial_rounding():
    torch.manual_seed(0)
    x=torch.randn(9,64,dtype=torch.bfloat16)
    w=torch.randn(32,64,dtype=torch.bfloat16)
    full=torch.nn.functional.linear(x,w)
    shards=[output_weight_shard(w,r,4) for r in range(4)]
    assert sum(s.numel() for s in shards)==w.numel()
    assert all(s.untyped_storage().nbytes()==s.numel()*s.element_size() for s in shards)
    result=torch.cat([torch.nn.functional.linear(x,s) for s in shards],-1)
    assert torch.equal(result,full)
    rounded=torch.stack([torch.nn.functional.linear(a,b) for a,b in zip(x.chunk(4,-1),w.chunk(4,-1))]).float().sum(0).to(x.dtype)
    assert not torch.equal(rounded,full)


def test_forward_gathers_inputs_then_outputs_without_summing(monkeypatch):
    torch.manual_seed(3)
    x=torch.randn(3,16,dtype=torch.bfloat16);w=torch.randn(8,16,dtype=torch.bfloat16)
    full=torch.nn.functional.linear(x,w);calls=[]
    def gather(value,dim):
        calls.append(value)
        if len(calls)==1:
            assert torch.equal(value,x[:,:4]);return x
        assert torch.equal(value,full[:,:2]);return full
    monkeypatch.setitem(sys.modules,'vllm.distributed',SimpleNamespace(tensor_model_parallel_all_gather=gather))
    module=SimpleNamespace(weight=output_weight_shard(w,0,4),return_bias=True,
        _jev_complete_k_calls=0,_jev_complete_k_max_input_bytes=0,_jev_complete_k_max_output_bytes=0)
    actual,bias=complete_k_forward(module,x[:,:4])
    assert torch.equal(actual,full) and bias is None and len(calls)==2
    assert module._jev_complete_k_calls==1
    with pytest.raises(ValueError,match='BF16'):
        complete_k_forward(module,x[:,:4].float())


@pytest.mark.parametrize('rank,size', [(4,4),(-1,4),(0,2),(True,4)])
def test_invalid_partition_rejected(rank,size):
    with pytest.raises(ValueError):output_weight_shard(torch.ones(8,16),rank,size)
