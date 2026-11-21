import copy
import json
from pathlib import Path

import pytest

from tools.reports import context_infonce_stage2_source as module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def setup(tmp_path,monkeypatch):
    from test_context_policy_artifacts import fixture
    from multimodal_jev.context_stage_lora import from_policy,bind_design
    from multimodal_jev import local_checkpoint,local_runtime_storage
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(local_runtime_storage,'check_paths',lambda _:None)
    origin=tmp_path/'origin';origin.mkdir()
    _,policy,pid,contract=fixture(origin,monkeypatch)
    design=json.loads((Path.cwd()/'configs/experiments/context-two-stage-v0.0.11.json').read_text())
    for stage in ('stage1','stage2'):design[stage]['lora']=from_policy(contract)
    write(origin/'design.json',design)
    s=design['stage1']
    training=dict(base_id=contract['base_id'],source_policy_id=pid,source_contract=contract,
        runtime_profile='test-bf16',release_sha256='c'*64,schedule_sha256='d'*64,
        parent_schedule_sha256='e'*64,schedule_release_sha256='c'*64,
        global_batch=256,microbatch=8,stage='instruction',batch_config=s['batch_config'],
        instruction_objective=s['instruction_objective'],lr_schedule=s['lr_schedule'])
    write(origin/'training/config.json',training)
    commit=dict(updates=318,contract=module.expected_contract(training),
        joint_policy={'policy_id':pid},joint_policy_manifest_sha256=module.file_hash(policy/'MANIFEST.json'))
    write(origin/'commit.json',commit)
    points=[dict(step=step,snapshot=str(policy),policy_id=pid,manifest_sha256=module.file_hash(policy/'MANIFEST.json'))
            for step in [0,*range(32,318,32),318]]
    points[-1]['commit']=dict(path=str(origin/'commit.json'),sha256=module.file_hash(origin/'commit.json'))
    plan=dict(schema='jev-infonce-source-evaluation-plan-v1',
        status='audited_source_policies_bound_evaluation_pending',algorithm='candidate_infonce',
        source_audit=dict(status='formal_infonce_artifacts_audited',world_size=4,real_exposures=81026,
            updates=list(range(318))),checkpoints=points,initial_policy_id=pid,
        reference_endpoint=dict(step=318,policy_id=pid),
        experiment=dict(task_stage={'updates':318},selection_recordings=9226,
            evaluation=dict(checkpoints=[p['step'] for p in points],primary='query_macro_set_f1',set_threshold=.5,
                secondary=['candidate_binary_nll','candidate_brier','top1_event_accuracy','top1_macro_f1'])),
        target_tuning_allowed=False,model_selection_performed=False,base_id=contract['base_id'],
        source_contract=contract,source_operations={'package':str(origin/'training')},
        training_config_sha256=module.file_hash(origin/'training/config.json'),
        stage_lora_transition=bind_design(design,contract))
    write(origin/'plan.json',plan)
    c=dict(source_plan=str(origin/'plan.json'),source_plan_sha256=module.file_hash(origin/'plan.json'),
        design=str(origin/'design.json'),design_sha256=module.file_hash(origin/'design.json'),
        output=str(tmp_path/'source'))
    # Full NPU training origin is a fixture; checkpoint, LoRA/head bytes, native
    # export, source validation and sealed-package audit are exercised for real.
    monkeypatch.setattr(module,'reconstruct_plan',lambda c:copy.deepcopy(plan))
    return c,plan


def test_real_endpoint_export_and_independent_reconstruction(tmp_path,monkeypatch):
    c,plan=setup(tmp_path,monkeypatch)
    result=module.prepare(c,{})
    proof=module.audit(dict(c,manifest_sha256=result['manifest_sha256']),{})
    assert proof['source_step']==318 and proof['training_dispatch_allowed'] is False
    assert not proof['optimizer_state_copied'] and not proof['base_weights_copied']
    bound=module.read(Path(c['output'])/'binding.json')
    assert bound['source_policy_id']==bound['reference_policy_id']==plan['reference_endpoint']['policy_id']
    assert bound['stage_lora_transition']['reuse_stage1_adapter'] is True
    assert bound['reset_optimizer_for_task_stage'] is True and not bound['selection_best_used']
    monkeypatch.setattr(module,'reconstruct_plan',lambda c:dict(plan,training_config_sha256='0'*64))
    with pytest.raises(ValueError,match='formal source configuration'):
        module.audit(dict(c,manifest_sha256=result['manifest_sha256']),{})


@pytest.mark.parametrize('fault',['commit','snapshot','design','budget','reuse','lora','prefix','ancestor'])
def test_wrong_source_or_transition_rejected_before_export(tmp_path,monkeypatch,fault):
    c,plan=setup(tmp_path,monkeypatch)
    if fault=='commit':write(Path(plan['checkpoints'][-1]['commit']['path']),{})
    elif fault=='snapshot':(Path(plan['checkpoints'][-1]['snapshot'])/'execution.pt').write_bytes(b'bad')
    elif fault=='design':write(Path(c['design']),{})
    elif fault in ('budget','reuse','lora'):
        design=module.read(c['design'])
        if fault=='budget':design['total_device_budget']=8
        elif fault=='reuse':design['stage2']['reuse_stage1_adapter']=False
        else:design['stage2']['lora']['rank']=8
        write(Path(c['design']),design);c['design_sha256']=module.file_hash(c['design'])
    elif fault=='prefix':plan['reference_endpoint']['step']=288
    else:c['output']=str(tmp_path)
    with pytest.raises(ValueError):module.prepare(c,{})
    if fault!='ancestor':assert not Path(c['output']).exists()


def test_rehashed_extra_checkpoint_is_not_allowed_in_source_package(tmp_path,monkeypatch):
    c,_=setup(tmp_path,monkeypatch);module.prepare(c,{})
    root=Path(c['output']);extra=root/'optimizer.pt';extra.write_bytes(b'not part of Stage2 initialization')
    m=module.read(root/'PACKAGE.json');m['files'][extra.name]=module.file_hash(extra)
    write(root/'PACKAGE.json',m)
    with pytest.raises(ValueError,match='unexpected assets'):
        module.audit(dict(c,manifest_sha256=module.file_hash(root/'PACKAGE.json')), {})


def test_new_design_endpoint_drives_source_view_and_reference(tmp_path,monkeypatch):
    from multimodal_jev.stage_experiment_contract import source_curve
    from multimodal_jev.context_topology_batch_plan import derive
    from tools.reports.prepare_context_reference_job import reference_source
    c,plan=setup(tmp_path,monkeypatch)
    d=module.read('configs/experiments/context-two-stage-margin-r64-dp8-v1.json')
    # Real small adapter tensors from fixture, independent of the batch geometry.
    for key in ('stage1','stage2'):
        d[key]['lora']=plan['stage_lora_transition'][key]
    curve=source_curve(d);s=d['stage1'];steps=curve['steps']
    write(Path(c['design']),d);c['design_sha256']=module.file_hash(c['design'])
    t=Path(plan['source_operations']['package'])/'config.json';training=module.read(t)
    training.update(batch_config=s['batch_config'],instruction_objective=s['instruction_objective'],
                    lr_schedule=s['lr_schedule'],global_batch=s['global_batch'])
    write(t,training);plan['training_config_sha256']=module.file_hash(t)
    plan['source_curve']=curve
    for p,step in zip(plan['checkpoints'],steps,strict=True):p['step']=step
    cp=plan['checkpoints'][-1]['commit'];commit=module.read(cp['path'])
    commit.update(updates=steps[-1],contract=module.expected_contract(training))
    write(Path(cp['path']),commit);cp['sha256']=module.file_hash(cp['path'])
    plan['reference_endpoint']['step']=steps[-1]
    plan['source_audit'].update(world_size=curve['world_size'],updates=list(range(steps[-1])))
    plan['experiment']['task_stage']['updates']=steps[-1]
    plan['experiment']['evaluation']['checkpoints']=steps
    bound,_,_=module.binding(c,plan)
    assert bound['source_step']==160 and bound['objective_config']==d['stage2']['objective_config']
    assert derive(bound['batch_config'])['global_batch']==384
    assert d['stage2']['batch_config']['source_queries'] is None
    root=tmp_path/'job';write(root/'initial/endpoint-commit.json',commit)
    origin=dict(source_policy_id=bound['source_policy_id'],reference_source=dict(endpoint=160,population=40513,max_tokens=4096))
    assert reference_source(root,origin)==origin['reference_source']
    with pytest.raises(ValueError,match='declared reference source'):
        reference_source(root,dict(origin,reference_source=dict(origin['reference_source'],endpoint=318)))
