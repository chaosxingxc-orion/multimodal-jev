import copy
import pytest
import torch
from test_context_policy_artifacts import fixture
import multimodal_jev.context_checkpoint_policy as module
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.context_policy_snapshot import load_policy,_parameters


def test_joint_export_loads_new_head_and_lora_but_keeps_marker(tmp_path,monkeypatch):
    model,source,policy_id,contract=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'check_paths',lambda *a:[])
    marker=model.backbone.embedding.weight[19].detach().clone()
    full={'_fsdp_wrapped_module.'+k:v.detach().clone()+.125 for k,v in _parameters(model).items()}
    out=tmp_path/'updated'
    receipt=module.export_checkpoint_policy(source,out,full,policy_id=policy_id,contract=contract,base_id='a'*64)
    assert receipt['policy_id']!=policy_id and receipt['contract']==contract
    bundle=read_execution_bundle(out,expected_policy_id=receipt['policy_id'],expected_contract=contract,expected_base_id='a'*64)
    assert torch.equal(bundle['tensors']['decision_token_row'],marker)
    load_policy(out,model,expected_policy_id=receipt['policy_id'],expected_contract=contract,expected_base_id='a'*64)
    assert all(torch.equal(p,full['_fsdp_wrapped_module.'+n]) for n,p in _parameters(model).items())
    assert torch.equal(model.backbone.embedding.weight[19],marker)
    with pytest.raises(FileExistsError):
        module.export_checkpoint_policy(source,out,full,policy_id=policy_id,contract=contract,base_id='a'*64)


@pytest.mark.parametrize('fault',['missing','marker','shape','dtype','contract'])
def test_invalid_snapshot_cannot_be_published(tmp_path,monkeypatch,fault):
    model,source,policy_id,contract=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'check_paths',lambda *a:[])
    full={n:p.detach().clone() for n,p in _parameters(model).items()};key=next(iter(full))
    if fault=='missing':del full[key]
    if fault=='marker':full['decision_token_row']=torch.zeros(8,dtype=torch.bfloat16)
    if fault=='shape':full[key]=full[key].flatten()[:0]
    if fault=='dtype':full[key]=full[key].float()
    if fault=='contract':contract=copy.deepcopy(contract);contract['request_version']='wrong'
    with pytest.raises(ValueError):
        module.export_checkpoint_policy(source,tmp_path/'bad',full,policy_id=policy_id,contract=contract,base_id='a'*64)
    assert not (tmp_path/'bad').exists()
