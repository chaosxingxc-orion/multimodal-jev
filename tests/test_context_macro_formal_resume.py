"""Sparse resume integration; CPU optimizer evidence, not NPU acceptance."""
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_training_checkpoint import capture, restore, validate
from tools.reports import context_macro_formal_resume as resume
from tools.reports.context_macro_formal_candidate import contract, generate as fresh_source
from test_context_macro_formal_candidate import config as fresh_config


def configuration(root, step=64):
    c = fresh_config()
    c.update(start_update=step, source_root=str(root), resume_root=str(root/'resume'),
        initial_adapter=str(root/'resume'/f'adapter-{step:04d}'), physical_devices=[4,5],
        global_batch=128, microbatch=16, base_id='base', source_policy_id='p0',
        source_contract={}, runtime_profile='fixture', release_sha256='release',
        schedule_sha256='schedule', behavior_bounds={}, dual_runtime_identity={'fixture':True},
        macro_client=dict(max_queries=128, rollout_queries=32, runtime_identity={'fixture':True}),
        pins={})
    return c


@pytest.mark.parametrize('step',[64,128,192,256,320,384,448,512,576])
def test_only_committed_nonterminal_cursors_preserve_experiment_identity(tmp_path,step):
    from tools.reports.audit_context_macro_training import expected_contract as audited_contract
    c=configuration(tmp_path,step)
    assert resume.resume_contract(c)==contract(fresh_config())
    assert resume.expected_contract(c)==audited_contract(c)
    with pytest.raises(ValueError):contract(c)


@pytest.mark.parametrize('change',[
    {'start_update':0}, {'start_update':1}, {'start_update':317}, {'start_update':634},
    {'start_update':64.0}, {'resume_root':'/other'}, {'initial_adapter':'/other'},
    {'seed':318}, {'stop_update':100}, {'microbatch':16,'dual_arm':'ranking_only'},
])
def test_invalid_recovery_cannot_weaken_fresh_contract(tmp_path,change):
    with pytest.raises(ValueError):resume.resume_contract(dict(configuration(tmp_path),**change))


def sealed(tmp_path,monkeypatch):
    c=configuration(tmp_path);r=tmp_path/'resume';r.mkdir()
    shards={}
    for rank in range(2):
        p=r/f'rank-{rank}/checkpoint-0064.pt';p.parent.mkdir();p.write_bytes(b'fixture')
        shards[str(p.relative_to(r))]=file_hash(p);c['pins'][str(p)]=file_hash(p)
    policy=r/'policy-0064';policy.mkdir();manifest=policy/'MANIFEST.json';manifest.write_text('{}')
    c['pins'][str(manifest)]=file_hash(manifest)
    experiment=resume.expected_contract(c);experiment['formal_experiment']=resume.resume_contract(c)
    commit=dict(updates=64,contract=experiment,rank_shards=shards,joint_policy_directory='policy-0064',
        joint_policy_manifest_sha256=file_hash(manifest),joint_policy={'policy_id':'p64'})
    path=r/'checkpoint-0064-committed.json'
    path.write_text(json.dumps(commit));c['pins'][str(path)]=file_hash(path)
    calls=[]
    def bundle(path,**kw):
        calls.append((path,kw));return {'fixture':True}
    monkeypatch.setattr(resume,'read_execution_bundle',bundle)
    monkeypatch.setattr(resume,'verify_native_adapter',lambda p,b:calls.append((p,b)))
    return c,commit,path,calls


def test_recovery_binds_all_rank_shards_policy_and_native_adapter(tmp_path,monkeypatch):
    c,_,_,calls=sealed(tmp_path,monkeypatch)
    assert resume.validate_resume(c)==contract(fresh_config())
    assert calls[0][1]['expected_policy_id']=='p64'
    assert calls[1][0]==c['initial_adapter']


@pytest.mark.parametrize('fault',['bytes','missing_pin','objective','cursor','rank','path','policy_hash'])
def test_resealed_bad_recovery_is_rejected(tmp_path,monkeypatch,fault):
    c,commit,path,calls=sealed(tmp_path,monkeypatch)
    shard=tmp_path/'resume/rank-1/checkpoint-0064.pt'
    if fault=='bytes':shard.write_bytes(b'changed')
    if fault=='missing_pin':c['pins'].pop(str(shard))
    if fault=='objective':commit['contract']['optimizer']['lr']=2e-4
    if fault=='cursor':commit['updates']=128
    if fault=='rank':commit['rank_shards'].pop('rank-1/checkpoint-0064.pt')
    if fault=='path':commit['joint_policy_directory']='../policy'
    if fault=='policy_hash':commit['joint_policy_manifest_sha256']='0'*64
    path.write_text(json.dumps(commit));c['pins'][str(path)]=file_hash(path)
    with pytest.raises(ValueError):resume.validate_resume(c)
    assert not calls


def test_actual_generated_restore_replays_sparse_adam_boundary(tmp_path):
    # Execute the unchanged generated restore block with real CPU Adam tensors.
    # Collective calls are substitutes; actual DP2/NPU replay is a separate gate.
    torch.set_num_threads(1)
    def model_and_optimizer():
        m=torch.nn.Linear(2,1,bias=False)
        with torch.no_grad():m.weight.copy_(torch.tensor([[.2,-.3]]))
        return m,torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=0.,foreach=False,fused=False)
    def update(m,opt):
        opt.zero_grad();m(torch.tensor([[.3,.7]])).square().sum().backward();opt.step()
    model,opt=model_and_optimizer()
    for _ in range(64):update(model,opt)
    c=configuration(tmp_path);r=tmp_path/'resume';r.mkdir()
    experiment=resume.expected_contract(c);experiment['formal_experiment']=resume.resume_contract(c)
    shards={}
    for rank in range(2):
        state=capture(model.named_parameters(),opt,rank=rank,world_size=2,completed_updates=64,contract=experiment)
        p=r/f'rank-{rank}/checkpoint-0064.pt';p.parent.mkdir();torch.save(state,p)
        shards[str(p.relative_to(r))]=file_hash(p)
    commit=dict(contract=experiment,updates=64,rank_shards=shards)
    path=r/'checkpoint-0064-committed.json';path.write_text(json.dumps(commit));c['pins'][str(path)]=file_hash(path)
    before=fresh_source(Path.cwd());source=resume.generate(Path.cwd())
    assert fresh_source(Path.cwd())==before
    assert 'formal=validate_resume(c)' in source and 'actor.update_policy(batch)' in source
    block=source[source.index('        if start_update:\n'):source.index('        # Bind resumed/live actor')]
    restored,newopt=model_and_optimizer();observed=[]
    def gather(peers,value):peers[:]=[value,value]
    env=dict(start_update=64,c=c,Path=Path,file_hash=file_hash,json=json,contract=experiment,
        world=2,torch=torch,rank=0,named=list(restored.named_parameters()),opt=newopt,
        validate=validate,restore=restore,dist=SimpleNamespace(all_gather_object=gather),out=tmp_path,
        atomic_json=lambda p,v:observed.append(v))
    exec(compile(textwrap.dedent(block),'<generated-sparse-restore>','exec'),env)
    assert observed==[dict(exact_masters=True,exact_adam=True,completed_updates=64)]
    update(model,opt);update(restored,newopt)
    assert torch.equal(model.weight,restored.weight)
    a=opt.state_dict();b=newopt.state_dict()
    assert a['param_groups']==b['param_groups']
    for k in a['state']:
        for field in a['state'][k]:assert torch.equal(a['state'][k][field],b['state'][k][field])
