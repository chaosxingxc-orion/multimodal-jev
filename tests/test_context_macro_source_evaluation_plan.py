"""Full curve identity tests with synthetic tensor/runtime audit substitutes."""
import json
from pathlib import Path

import pytest

from test_context_macro_formal_candidate import config
from tools.reports import context_macro_source_evaluation_plan as module
from tools.reports.prepare_context_macro_replay import write


def fixture(tmp_path,monkeypatch):
    package=tmp_path/'prepared/package';package.mkdir(parents=True)
    ops=tmp_path/'ops';run=ops/'collected/run';run.mkdir(parents=True)
    training=dict(config(),source_policy_id='stage1',source_contract={'test':True},base_id='base',
        release_sha256='release',dual_runtime_identity={'profile':'fixture'})
    write(package/'training.json',training)
    (package/'policy').mkdir();write(package/'policy/MANIFEST.json',{'policy_id':'stage1'})
    for step in module.STEPS[1:]:
        p=run/f'policy-{step:04d}';p.mkdir();write(p/'MANIFEST.json',{'policy_id':f'p{step}'})
        write(run/f'checkpoint-{step:04d}-committed.json',dict(updates=step,
            contract=dict(formal_experiment=module.contract(training),source_policy_id='stage1',
                algorithm=training['algorithm'],release_sha256='release'),
            joint_policy_directory=p.name,joint_policy_manifest_sha256=module.file_hash(p/'MANIFEST.json'),
            joint_policy={'policy_id':f'p{step}'}))
    def reseal():
        write(package/'FILES.json',{str(p.relative_to(package)):module.file_hash(p)
            for p in package.rglob('*') if p.is_file() and p.name!='FILES.json'})
        write(ops/'transport.json',{'files':{str(p.relative_to(ops/'collected')):module.file_hash(p)
            for p in run.rglob('*') if p.is_file()}})
        operations=dict(prepared=str(package.parent),operations_root=str(ops),manifest_sha256=module.verify(package))
        proof=dict(status='formal_dp2_outputs_audited',files_sha256=operations['manifest_sha256'],
            transport_sha256=module.file_hash(ops/'transport.json'),updates=[{} for _ in range(634)])
        return operations,proof
    calls=[]
    def read(path,**kw):
        assert json.loads((path/'MANIFEST.json').read_text())['policy_id']==kw['expected_policy_id']
        calls.append((path,kw));return kw
    monkeypatch.setattr(module,'read_execution_bundle',read)
    monkeypatch.setattr(module,'verify_native_adapter',lambda path,bundle:None)
    operations,proof=reseal()
    return operations,proof,reseal,calls


def test_complete_plan_binds_actual_points_without_selecting_or_scoring(tmp_path,monkeypatch):
    ops,proof,_,calls=fixture(tmp_path,monkeypatch)
    plan=module.build(ops,proof)
    assert len(calls)==11
    assert [p['step'] for p in plan['checkpoints']]==module.STEPS
    assert plan['initial_policy_id']=='stage1'
    assert plan['experiment']['selection_recordings']==9226
    assert plan['experiment']['evaluation']['primary']=='query_macro_set_f1'
    assert not plan['model_selection_performed'] and not plan['target_tuning_allowed']
    assert plan['checkpoints'][-1]['policy_id']=='p634'


@pytest.mark.parametrize('fault',['status','count','transport','step','algorithm','reference','directory','manifest','tensor'])
def test_refuses_wrong_run_or_mutated_source_checkpoint(tmp_path,monkeypatch,fault):
    ops,proof,reseal,_=fixture(tmp_path,monkeypatch)
    if fault=='status':proof['status']='finite_dp2_efficiency_audited'
    if fault=='count':proof['updates'].pop()
    if fault=='transport':proof['transport_sha256']='changed'
    path=Path(ops['operations_root'])/'collected/run/checkpoint-0064-committed.json'
    if fault in ('step','algorithm','reference','directory','manifest'):
        commit=json.loads(path.read_text())
        if fault=='step':commit['updates']=63
        if fault=='algorithm':commit['contract']['algorithm']='rloo_v1'
        if fault=='reference':commit['contract']['source_policy_id']='other'
        if fault=='directory':commit['joint_policy_directory']='../other'
        if fault=='manifest':commit['joint_policy_manifest_sha256']='changed'
        write(path,commit);ops,proof=reseal()
    if fault=='tensor':
        def reject(*a,**kw):raise ValueError('actual tensor invalid')
        monkeypatch.setattr(module,'read_execution_bundle',reject)
    with pytest.raises(ValueError):module.build(ops,proof)


def test_prepare_reaudits_runtime_before_writing_any_plan(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'load_config',lambda *a:({},{}))
    def reject(*a):raise ValueError('actual audit fails')
    monkeypatch.setattr(module,'audit_collected',reject)
    c=dict(source_operations_config='config',source_operations_local='local',output=str(tmp_path/'plan.json'))
    with pytest.raises(ValueError,match='actual audit'):module.prepare(c,{})
    assert not Path(c['output']).exists()
