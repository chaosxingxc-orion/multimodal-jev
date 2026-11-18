import pytest
import torch

from test_context_policy_artifacts import fixture
from multimodal_jev.context_policy_snapshot import _parameters
from multimodal_jev.context_fsdp_export import canonical_execution,load_assembled_execution


def test_assembled_execution_preserves_frozen_embedding(tmp_path,monkeypatch):
    model,_,_,_=fixture(tmp_path,monkeypatch)
    original=model.backbone.embedding.weight.detach().clone()
    tensors={'_fsdp_wrapped_module.'+name.replace('.weight','._fsdp_wrapped_module.weight'):
             value.detach().cpu().clone()+.125 for name,value in _parameters(model).items()}
    result=load_assembled_execution(model,tensors)
    assert result['exact_bf16'] and not result['optimizer_restored']
    assert torch.equal(model.backbone.embedding.weight,original)
    assert all(torch.equal(v,canonical_execution(tensors)[k]) for k,v in _parameters(model).items())


@pytest.mark.parametrize('fault',['duplicate','missing','shape','fp32','nan','extra'])
def test_bad_assembled_payload_never_partially_loads(tmp_path,monkeypatch,fault):
    model,_,_,_=fixture(tmp_path,monkeypatch)
    before={k:p.detach().clone() for k,p in model.named_parameters()}
    tensors={k:v.detach().clone()+.125 for k,v in _parameters(model).items()}
    key=list(tensors)[-1]
    if fault=='duplicate':tensors['_fsdp_wrapped_module.'+key]=tensors[key]
    if fault=='missing':del tensors[key]
    if fault=='shape':tensors[key]=tensors[key].reshape(-1)[:0]
    if fault=='fp32':tensors[key]=tensors[key].float()
    if fault=='nan':tensors[key].fill_(float('nan'))
    if fault=='extra':tensors['decision_token_row']=torch.zeros(8,dtype=torch.bfloat16)
    with pytest.raises(ValueError):load_assembled_execution(model,tensors)
    assert all(torch.equal(v,before[k]) for k,v in model.named_parameters())
