import copy,json
import pytest
import torch
from test_context_policy_artifacts import fixture
import multimodal_jev.context_policy_admission as module
from multimodal_jev.context_policy_snapshot import export_policy
from multimodal_jev.context_runtime_profile import PROFILE
from multimodal_jev.span_request_identity import tensor_identity


def setup(tmp_path,monkeypatch):
    model,old,_,contract=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'check_paths',lambda *a:[])
    # Tokenizer extension is independently tested against real assets. These
    # tests isolate evidence binding and preservation of initialization bytes.
    monkeypatch.setattr(module,'validate_processor_extension',lambda *a:None)
    model.backbone.embedding=torch.nn.Embedding(151707,8,dtype=torch.bfloat16).requires_grad_(False)
    model.decision_token_id=151706;contract=copy.deepcopy(contract);contract['token']['id']=151706
    (old/'processor/tokenizer.json').write_text(json.dumps(dict(added_tokens=[dict(id=151706,content='<|jev_decision|>',special=True)])))
    source=tmp_path/'source';aid=export_policy(source,model,old/'processor',contract)
    contract['runtime_profile']=PROFILE;contract['request_version']='jev-context-exclusive_event-v1'
    checked=tmp_path/'checked'
    with torch.no_grad():model.decision_head.projection.weight.add_(.125)
    bid=export_policy(checked,model,source/'processor',contract)
    ok=dict(state=dict(ExitCode=0,OOMKilled=False))
    proof=dict(parity=dict(ok,joint_policy_id=bid,independent_probability_checks=[dict(id=str(i),probability_max_error=0.,threshold_half_disagreements=0) for i in range(256)]),
        prefix=dict(ok,result=dict(passed=True),rank_updates=[{}]*8),
        replay=dict(ok,fresh_process_replay=dict(exact_four_rank_masters_adam=True,exact_128_bf16_execution=True)),
        token_assets=dict(source_policy_id=aid,original_tokenizer_preserved=True,
            embedding_initialization=dict(exact=True,marker=tensor_identity(model.backbone.embedding.weight[151706].detach()))))
    return source,checked,proof


def test_admission_changes_contract_not_initialization(tmp_path,monkeypatch):
    source,checked,proof=setup(tmp_path,monkeypatch)
    out=tmp_path/'admitted';r=module.admit_initial_policy(source,checked,out,base='/base',**proof)
    assert (source/'execution.pt').read_bytes()==(out/'execution.pt').read_bytes()
    assert r['contract']['token']['registration_status']=='admitted'
    assert r['contract']['request_version']=='jev-context-exclusive_event-v1'
    assert r['policy_id']!=r['source_policy_id']


@pytest.mark.parametrize('fault',['policy','partial','gap','flip','failed','restore','marker'])
def test_evidence_rejection_never_publishes(fault,tmp_path,monkeypatch):
    source,checked,p=setup(tmp_path,monkeypatch)
    if fault=='policy':p['parity']['joint_policy_id']='wrong'
    if fault=='partial':p['parity']['independent_probability_checks'].pop()
    if fault=='gap':p['parity']['independent_probability_checks'][0]['probability_max_error']=1e-6
    if fault=='flip':p['parity']['independent_probability_checks'][0]['threshold_half_disagreements']=1
    if fault=='failed':p['prefix']['state']['ExitCode']=1
    if fault=='restore':p['replay']['fresh_process_replay']['exact_four_rank_masters_adam']=False
    if fault=='marker':p['token_assets']['embedding_initialization']['marker']['sha256']='bad'
    with pytest.raises(ValueError):module.admit_initial_policy(source,checked,tmp_path/'bad',base='/base',**p)
    assert not (tmp_path/'bad').exists()
