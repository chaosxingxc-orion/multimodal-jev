"""Candidate source checks use real four-rank saved state and export fixtures."""
import json
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from multimodal_jev.span_batch_training_config import PROFILE,batch_layout
from multimodal_jev.batched_continuous_resume import authenticate
from test_simple_continuous_resume import formal_fixture


def fixture(tmp_path,monkeypatch):
    c,root,put=formal_fixture(tmp_path,monkeypatch)
    c.update(candidate_profile=PROFILE,global_batch=128,microbatch_per_rank=8,batch_order='length_grouped')
    c['stage2_initialization']['candidate_parity_audit_config']=c['stage2_initialization'].pop('gate')
    train=json.loads((root/'config.json').read_text())
    train.update({k:c[k] for k in ('candidate_profile','global_batch','microbatch_per_rank','batch_order','stage2_initialization')})
    put(root/'config.json',train)
    launch=json.loads((root/'launch.json').read_text());launch['config_sha256']=file_hash(root/'config.json')
    external=tmp_path/'external-training-launch.json';put(external,launch);(root/'launch.json').unlink()
    c['continuous_resume']['training_launch']=str(external)
    layout=batch_layout(c)
    for rank in range(4):
        p=root/'run'/f'loop-rank-{rank}/checkpoint-0003.pt'
        state=torch.load(p,weights_only=True);state.update(candidate_profile=PROFILE,batch_layout=layout,batch_order='length_grouped');torch.save(state,p)
    p=root/'run/checkpoint-0003-committed.json';v=json.loads(p.read_text());v['files']={p:file_hash(p) for p in v['files']}
    from multimodal_jev.slurp_training import content_hash
    v['cursor']['checkpoint_id']=content_hash(v['files']);put(p,v)
    p=Path(c['continuous_resume']['audit']);v=json.loads(p.read_text())
    v.update(candidate_profile=PROFILE,batch_layout=layout,batch_order='length_grouped',exact_length_permutation=True)
    v['files']={str(p):file_hash(p) for p in root.rglob('*') if p.is_file()}
    for q in (external,Path(c['continuous_resume']['training_observed'])):v['files'][str(q)]=file_hash(q)
    put(p,v);c['pins'][str(p)]=file_hash(p);c['pins'][str(external)]=file_hash(external)
    return c,root,put


def test_candidate_recovery_keeps_exact_cursor_and_all_shards(tmp_path,monkeypatch):
    c,_,_=fixture(tmp_path,monkeypatch);r=authenticate(c)
    assert r['completed_updates']==3 and len(r['states'])==4
    assert all(s['batch_layout']['microbatch_per_rank']==8 for s in r['states'])
    assert r['reference_policy_id']=='base:reference'


@pytest.mark.parametrize('fault',['profile','micro','order','audit_profile','permutation','missing_shard','reference'])
def test_candidate_recovery_rejects_changed_execution_or_state(tmp_path,monkeypatch,fault):
    c,root,put=fixture(tmp_path,monkeypatch)
    if fault=='profile':c['candidate_profile']='other'
    elif fault=='micro':c['microbatch_per_rank']=16
    elif fault=='order':c['batch_order']='original'
    elif fault=='reference':c['continuous_resume']['reference_policy_id']='base:rl'
    else:
        p=Path(c['continuous_resume']['audit']);v=json.loads(p.read_text())
        if fault=='audit_profile':v['candidate_profile']='other'
        elif fault=='permutation':v['exact_length_permutation']=False
        else:v['files'].pop(str(root/'run/loop-rank-0/checkpoint-0003.pt'))
        put(p,v);c['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError):authenticate(c)


def test_candidate_restored_next_adam_update_matches_uninterrupted():
    from test_simple_stage2_resume import tensors,plan
    from multimodal_jev.batched_stage2_resume import restore,continuous_plan
    original,old_optimizer,loaded,fresh,recovered=tensors()
    _,receipt=restore(loaded,fresh,recovered,rank=0)
    assert receipt['exact_state']['exact']
    assert continuous_plan(plan(),recovered,fresh,base_id='base',adapter_id='rl')['start_update']==1
    for model,opt in ((original,old_optimizer),(loaded,fresh)):
        opt.zero_grad(set_to_none=True);model(torch.ones(1,3)).square().sum().backward();opt.step()
    assert torch.equal(original.weight,loaded.weight)
    for k,v in old_optimizer.state_dict()['state'][0].items():assert torch.equal(v,fresh.state_dict()['state'][0][k])
