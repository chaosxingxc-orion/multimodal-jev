"""Candidate stage transition authenticates the real complete parity verifier."""
import json
from pathlib import Path
import pytest
pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from multimodal_jev.batched_stage2_initialization import authenticate
from multimodal_jev.span_batch_training_config import PROFILE
from multimodal_jev.slurp_instruction_resume import FIELDS
from multimodal_jev.resident_service_admission import IMAGE
from test_slurp_batched_selection_verifier import fixture, m


def source(tmp_path,monkeypatch):
    audit_path,gate,put=fixture(tmp_path,monkeypatch)
    audit=json.loads(audit_path.read_text());root=tmp_path/'train'
    previous=json.loads((root/'config.json').read_text())
    for key in FIELDS:previous.setdefault(key,'fixed')
    previous.update(global_batch=128,microbatch_per_rank=8,batch_order='length_grouped',loss_reduction='real_request_mean')
    put(root/'config.json',previous)
    state_path=Path(audit['checkpoint_evidence']);state=json.loads(state_path.read_text())
    state.update(ranks=4,tensors=112,candidate_profile=PROFILE)
    assets=[root/'run'/f'rank-{r}/checkpoint-0634.pt' for r in range(4)]
    assets += [Path(state['adapter'])/name for name in ('adapter_config.json','adapter_model.safetensors')]
    for i,p in enumerate(assets):put(p,{'asset':i})
    state['files']={str(p):file_hash(p) for p in assets};put(state_path,state)
    for key in ('training_launch','backend_launch','training_observed','backend_observed'):
        p=Path(audit[key]);v=json.loads(p.read_text());v['image']=IMAGE
        if key.endswith('launch'):
            v['config_sha256']=file_hash((root if key=='training_launch' else gate)/'config.json')
        put(p,v)
    audit['pins']={p:file_hash(p) for p in audit['pins']};put(audit_path,audit)
    config=dict(previous,stage2_initialization=dict(mode='final_stage1_fresh_optimizer',
        algorithm='grpo_v1',start_update=0,total_updates=634,stop_update=634,reset_optimizer=True,
        candidate_parity_audit_config=str(audit_path)),pins={str(p):file_hash(p) for p in
        [audit_path,gate/'config.json',Path(audit['backend_launch']),Path(audit['backend_observed'])]})
    return config,audit,put


def test_real_full_selection_to_candidate_stage_boundary(tmp_path,monkeypatch):
    config,_,_=source(tmp_path,monkeypatch)
    result=authenticate(config,verify_parity=m.verify)
    assert result['policy_id']==result['reference_policy_id']=='b:a'
    assert result['completed_updates']==0 and result['source_updates']==634
    assert result['candidate_profile']==PROFILE and len(result['shard_paths'])==4


@pytest.mark.parametrize('fault',['micro','order','reset','algorithm','missing_shard','state_profile','last_probability'])
def test_invalid_candidate_stage_boundary_rejected(tmp_path,monkeypatch,fault):
    config,audit,put=source(tmp_path,monkeypatch)
    if fault=='micro':config['microbatch_per_rank']=16
    elif fault=='order':config['batch_order']='original'
    elif fault=='reset':config['stage2_initialization']['reset_optimizer']=False
    elif fault=='algorithm':config['stage2_initialization']['algorithm']='other'
    elif fault=='last_probability':
        p=Path(audit['backend_root'])/'run/vllm/result.json';v=json.loads(p.read_text())
        v['records'][-1]['candidate_logprobs'][0][0]+=1;put(p,v)
    else:
        p=Path(audit['checkpoint_evidence']);v=json.loads(p.read_text())
        if fault=='missing_shard':v['files'].pop(next(k for k in v['files'] if k.endswith('checkpoint-0634.pt')))
        else:v['candidate_profile']='other'
        put(p,v);audit['pins'][str(p)]=file_hash(p)
        ap=Path(config['stage2_initialization']['candidate_parity_audit_config']);put(ap,audit)
        config['pins'][str(ap)]=file_hash(ap)
    with pytest.raises(ValueError):authenticate(config,verify_parity=m.verify)
