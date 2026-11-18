"""Actual candidate permutation, sampling, saved tensors and probability audit."""
import importlib.util
import json
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.span_batch_training_config import PROFILE,batch_layout,reorder_global_items
from test_simple_resident_train_audit import fixture as base_fixture


def fixture(tmp_path,monkeypatch):
    _,c,root,put=base_fixture(tmp_path,monkeypatch,formal_prefix=True,formal_stop=2)
    import multimodal_jev.simple_stage2_audit as old
    import multimodal_jev.batched_stage2_audit as candidate
    monkeypatch.setattr(candidate,'source',old.source);monkeypatch.setattr(candidate,'selections',old.selections)
    p=root/'config.json';train=json.loads(p.read_text());train.update(candidate_profile=PROFILE,
        global_batch=128,microbatch_per_rank=8,batch_order='length_grouped');put(p,train)
    layout=batch_layout(train)
    from multimodal_jev.simple_rl_schedule import SimpleRLSchedule
    from multimodal_jev.simple_audio_batch import compile_batch
    from multimodal_jev.simple_sampled_rollout import actor_batch
    from multimodal_jev.slurp_official_training import load
    from transformers import Qwen3ASRProcessor
    rows,task,_=load('data','release','learning',purpose='optimize');processor=Qwen3ASRProcessor.from_pretrained('base')
    for update in (1,2):
        step=update-1
        behavior=json.loads((root/'run'/f'behavior-{step:04d}.json').read_text())
        reference=json.loads((root/'run'/f'reference-{step:04d}.json').read_text())
        policy=behavior['base_id']+':'+behavior['adapter_id']
        slots,_=SimpleRLSchedule().at(step)
        entries,_,_=compile_batch({r['id']:r for r in rows},task,slots,processor,policy_id=policy,global_batch=128)
        items,_=actor_batch(entries,behavior,reference,policy_id=policy,reference_policy_id='base:reference',seed=317,update=step)
        _,perm=reorder_global_items(items,train)
        p=root/'run'/f'inputs-{step:04d}.json';v=json.loads(p.read_text());v.update(candidate_batch_layout=layout,candidate_rank_indices=perm);put(p,v)
        flat=[]
        for rank in range(4):flat+=json.loads((root/'run'/f'loop-rank-{rank}/after-{update:04d}.json').read_text())
        for rank in range(4):
            out=root/'run'/f'loop-rank-{rank}'
            put(out/f'after-{update:04d}.json',[flat[i] for i in perm[rank]])
            p=out/f'checkpoint-{update:04d}.pt';s=torch.load(p,weights_only=True)
            s.update(candidate_profile=PROFILE,batch_layout=layout,batch_order='length_grouped');torch.save(s,p)
        p=root/'run'/f'checkpoint-{update:04d}-committed.json';v=json.loads(p.read_text())
        v['files']={p:file_hash(p) for p in v['files']};v['cursor']['checkpoint_id']=content_hash(v['files']);put(p,v)
        for rank in range(4):
            p=root/'run'/f'loop-rank-{rank}/update-{update:04d}.json';r=json.loads(p.read_text());r['checkpoint']=v;put(p,r)
    import verify_batched_stage2_training as m
    return m,c,root,put


def test_every_candidate_update_independently_replayed(tmp_path,monkeypatch):
    m,c,_,_=fixture(tmp_path,monkeypatch);r=m.run(c,tmp_path/'out')
    assert r['exact_length_permutation'] and r['candidate_profile']==PROFILE
    assert len(r['checks'])==2 and r['updates']==2 and not r['complete_training']


@pytest.mark.parametrize('fault',['permutation','rank_order','probability','layout'])
def test_candidate_audit_rejects_plausible_but_misbound_records(tmp_path,monkeypatch,fault):
    m,c,root,put=fixture(tmp_path,monkeypatch)
    p=root/'run/inputs-0001.json' if fault in ('permutation','layout') else root/'run/loop-rank-0/after-0002.json'
    v=json.loads(p.read_text())
    if fault=='permutation':v['candidate_rank_indices'][0].reverse()
    elif fault=='layout':v['candidate_batch_layout']['microbatch_per_rank']=16
    elif fault=='rank_order':v.reverse()
    else:v[0]['candidate_logprobs'][0][0]+=.1
    put(p,v)
    with pytest.raises(ValueError):m.run(c,tmp_path/'out')


@pytest.mark.parametrize('fault', [None, 'missing', 'probability'])
def test_resumed_audit_requires_recomputed_pre_update_probabilities(tmp_path, monkeypatch, fault):
    import copy
    m,c,root,put=fixture(tmp_path,monkeypatch)
    import multimodal_jev.batched_stage2_audit as candidate
    from multimodal_jev.batched_stage2_recovery_policy import prepare
    from transformers import Qwen3ASRProcessor
    source=candidate.source(c,verify_parity=None)
    first=json.loads((root/'run/loop-rank-0/update-0001.json').read_text())
    train=json.loads((root/'config.json').read_text())
    train['continuous_resume']=dict(training_root=str(root));put(root/'config.json',train)
    recovered=dict(completed_updates=1,policy_id=first['policy_id'],reference_policy_id='base:reference',
        checkpoint_id='accepted-1',selection_history=[],
        prefix_audit=dict(checks=[first['acceptance']],policies={'0':'base:initial','1':first['policy_id']},
                          initial_state_checks=source['checks']),
        files={str(p):file_hash(p) for p in root.rglob('*') if p.is_file()})
    source['recovery']=recovered
    monkeypatch.setattr(candidate,'source',lambda *a,**k:source)
    replay=prepare(train,recovered,Qwen3ASRProcessor.from_pretrained('base'))
    for rank in range(4):
        report=dict(rank=rank,binding=copy.deepcopy(replay['binding']),modes=[
            dict(microbatch=micro,records=copy.deepcopy(replay['original'][rank])) for micro in (1,8)])
        if fault=='missing' and rank==3:continue
        if fault=='probability' and rank==0:
            report['passed']=True
            report['modes'][1]['records'][0]['candidate_logprobs'][0][0]+=.1
        put(root/'run'/f'rank-{rank}'/'restored-policy.json',report)
    if fault:
        with pytest.raises((ValueError,FileNotFoundError)):m.run(c,tmp_path/'out')
    else:
        result=m.run(c,tmp_path/'out')
        assert result['exact_pre_update_recovery'] and result['restored_policy_check']['comparisons']==256
        assert result['segment_start_update']==1 and result['segment_new_updates']==1
