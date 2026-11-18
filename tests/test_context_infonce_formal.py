import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from tools.reports import context_infonce_formal as formal
from tools.reports.audit_context_infonce_training import checkpoint_steps, check_artifact_inventory, check_update
from tools.reports.context_infonce_trainer import generate
from multimodal_jev.context_warmup import learning_rate

ROOT=Path(__file__).resolve().parents[1]


def config():
    design=json.loads((ROOT/'configs/experiments/context-two-stage-v0.0.4.json').read_text())
    c=dict(design['stage1'],physical_devices=list(range(8)),microbatch=8,
           source_root='/old/source',schedule='/old/source/schedule',
           parent_schedule='/old/source/parent-schedule',pins={'/old/source/code':'hash'})
    return formal.formal_config(c,'/old','/new',design)


def inventory(root,c):
    root.mkdir()
    for step in checkpoint_steps(c):
        (root/f'checkpoint-{step:04d}-committed.json').touch()
        (root/f'execution-{step:04d}.pt').touch()
        (root/f'policy-{step:04d}').mkdir()
    for index in range(c['start_update'],c['stop_update']):
        (root/f'batch-{index:04d}.json').touch()
    for rank in range(8):
        d=root/f'rank-{rank}';d.mkdir()
        for step in checkpoint_steps(c):(d/f'checkpoint-{step:04d}.pt').touch()
        for step in range(c['start_update']+1,c['stop_update']+1):(d/f'update-{step:04d}.json').touch()


def test_full_budget_sparse_inventory_and_no_extra_saves(tmp_path):
    c=config();run=tmp_path/'run';inventory(run,c)
    assert check_artifact_inventory(run,c,8)==set(range(16,161,16))
    (run/'rank-7/checkpoint-0001.pt').touch()
    with pytest.raises(ValueError,match='unexpected rank'):check_artifact_inventory(run,c,8)
    (run/'rank-7/checkpoint-0001.pt').unlink()
    (run/'rank-7/update-0031.json').unlink()
    with pytest.raises(ValueError,match='unexpected rank'):check_artifact_inventory(run,c,8)


@pytest.mark.parametrize('changes',[
    {'evaluation_steps':[0,160]}, {'stop_update':100}, {'reference_endpoint':144},
    {'resume_root':'/previous/run'}, {'engineering_prefix':True},
    {'training_dispatch_allowed':False}, {'checkpoint_interval':0},
])
def test_reject_changed_formal_window(changes):
    with pytest.raises(ValueError):checkpoint_steps(dict(config(),**changes))


@pytest.mark.parametrize('fault',[None,'lr','checkpoint','cursor','backbone','ownership','time','cache'])
def test_intermediate_updates_have_evidence_without_tensor_restore(fault):
    c=config();scope=dict(tensors=1,base_model_saved=False,local_numel=dict(lora=0,head=2))
    record=dict(update=3,real_queries=512,checkpoint=False,learning_rate=learning_rate(c['lr_schedule'],2,160),
        input_service=dict(cache_hits=512),actor_seconds=1.,total_seconds=2.,preparation_seconds=.5,
        optimizer_audit={'decision_head.projection.weight':dict(numel=2,state='initialized',step=3)})
    if fault=='lr':record['learning_rate']=1e-4
    if fault=='checkpoint':record['checkpoint']=True
    if fault=='cursor':record['optimizer_audit']['decision_head.projection.weight']['step']=2
    if fault=='backbone':record['optimizer_audit']={'backbone.embed_tokens.weight':dict(numel=2,state='initialized',step=3)}
    if fault=='ownership':scope['local_numel']['head']=3
    if fault=='time':record['total_seconds']=float('nan')
    if fault=='cache':record['input_service']['cache_hits']=511
    if fault:
        with pytest.raises(ValueError):check_update(record,c,step=3,real=512,checkpoint=False,scope=scope)
    else:assert check_update(record,c,step=3,real=512,checkpoint=False,scope=scope)==2.


def test_formal_dispatch_requires_own_explicit_gate():
    with pytest.raises(ValueError,match='explicit formal'):formal.launch({'finite_training_dispatch_allowed':True},{})


def test_accepted_trainer_supports_sparse_checkpoint_without_source_change():
    source=generate(ROOT)
    assert "save_now=(index+1)%c['checkpoint_interval']==0 or index+1==stop_update" in source
    assert "if save_now:" in source
    assert "if c['engineering_prefix'] is not True and not c['training_dispatch_allowed']:" in source
    compile(source,'<formal-runtime>','exec')


@pytest.mark.parametrize('fault',[None,'runtime','objective','design','evidence','runner','manifest'])
def test_validator_rejects_resealed_changes_to_accepted_runtime(tmp_path,monkeypatch,fault):
    design=json.loads((ROOT/'configs/experiments/context-two-stage-v0.0.4.json').read_text())
    original=dict(design['stage1'],physical_devices=list(range(8)),microbatch=8,
        source_root='/old/source',schedule='/old/source/schedule',parent_schedule='/old/source/parent-schedule',
        pins={'/old/source/code.py':'unused'},release='/release')
    package=tmp_path/'package';(package/'source').mkdir(parents=True)
    (package/'source/code.py').write_text('accepted')
    (package/'run.py').write_text(formal.shared.RUNNER)
    parent_files={str(p.relative_to(package)):file_hash(p) for p in package.rglob('*') if p.is_file()}
    pm=dict(remote_root='/old',files=parent_files,image='fixed',training_devices=list(range(8)),
        inference_services=[],batch_plan={},layout={})
    evidence={'replay':'exact','tail':'65_then_512'}
    monkeypatch.setattr(formal,'parent',lambda c:({'manifest_sha256':'parent'},pm,original,None,{}))
    monkeypatch.setattr(formal,'admission',lambda c,m:evidence)
    monkeypatch.setattr(formal.shared,'generate',lambda p:None)
    monkeypatch.setattr(formal.shared.lifecycle,'release_evidence',lambda c,t:None)
    design_path=tmp_path/'design.json';design_path.write_text(json.dumps(design))
    c=dict(package=str(package),host='npu',tag='formal-test',remote_root='/new',design=str(design_path),
           parent_audit_sha256='audit')
    t=formal.formal_config(original,'/old','/new',design)
    m=dict(pm,status='formal_infonce_prepared_only',remote_root='/new',parent_manifest_sha256='parent',
           parent_audit_sha256='audit',admission=dict(evidence),design_sha256=file_hash(design_path))
    if fault=='runtime':(package/'source/code.py').write_text('changed')
    if fault=='objective':t['instruction_objective']['probability_coefficient']=.3
    if fault=='design':design_path.write_text('{}')
    if fault=='evidence':m['admission']['tail']='other'
    if fault=='runner':(package/'run.py').write_text('changed')
    (package/'config.json').write_text(json.dumps(t))
    m['files']={str(p.relative_to(package)):file_hash(p) for p in package.rglob('*') if p.is_file()}
    (package/'PREPARED.json').write_text(json.dumps(m));c['manifest_sha256']=file_hash(package/'PREPARED.json')
    if fault=='manifest':(package/'PREPARED.json').write_text(json.dumps(dict(m,extra=True)))
    if fault:
        with pytest.raises(ValueError):formal.validate(c)
    else:assert formal.validate(c)[1]==t
