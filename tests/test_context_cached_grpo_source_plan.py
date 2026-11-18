"""Actual commit-layout binding; model and full NPU audit explicitly substituted."""
import copy
from pathlib import Path

import pytest

from tools.reports import context_cached_grpo_source_plan as module
from test_context_cached_formal import fixture as formal_fixture
from test_prepare_context_cached_training import write


def setup(tmp_path,monkeypatch):
    spec=formal_fixture(tmp_path,monkeypatch)
    spec['manifest_sha256']=module.formal.package_action(spec,{})['manifest_sha256']
    package=Path(spec['output']);training=module.read(package/'config.json')
    endpoint=dict(updates=318,joint_policy={'policy_id':training['source_policy_id']})
    write(Path(spec['parent']['output'])/'initial/endpoint-commit.json',endpoint)
    training['reference_dataset']['identity']['policy_id']=training['source_policy_id']
    training['stage_lora_transition']={'fixture':'explicit same-adapter reuse'}
    write(package/'config.json',training)
    write(package/'initial/policy/MANIFEST.json',{'policy_id':training['source_policy_id']})
    contract={'fixture':'full-real-contract'}
    monkeypatch.setattr(module.formal,'expected_contract',lambda *a:contract)
    collection=tmp_path/'collected';run=collection/'run';transport=tmp_path/'transport.json'
    for step in module.formal.STEPS[1:]:
        policy=run/f'policy-{step:04d}';write(policy/'MANIFEST.json',{'policy_id':f'p{step}'})
        published=dict(updates=step,contract=copy.deepcopy(contract),joint_policy_directory=policy.name,
            joint_policy_manifest_sha256=module.file_hash(policy/'MANIFEST.json'),joint_policy={'policy_id':f'p{step}'})
        write(run/f'policy-{step:04d}-committed.json',published)
        write(run/f'checkpoint-{step:04d}-committed.json',dict(published,rank_shards={'0':'shard0','1':'shard1'}))
    operations=dict(training_package=spec,collection=str(collection),transport=str(transport))
    path=tmp_path/'audit-config.json';write(path,operations)
    c=dict(training_audit_config=str(path),training_audit_config_sha256=module.file_hash(path))
    proof=dict(status='formal_cached_reference_training_outputs_audited',updates=[{}]*634,
        real_query_exposures=81026,online_reference_calls=0,manifest_sha256=spec['manifest_sha256'])
    def reseal():
        write(transport,dict(files={str(p.relative_to(collection)):module.file_hash(p) for p in collection.rglob('*') if p.is_file()}))
        proof['transport_sha256']=module.file_hash(transport)
    reseal();calls=[]
    def audit(*a):calls.append('audit');return copy.deepcopy(proof)
    def bundle(path,**kw):
        if module.read(path/'MANIFEST.json')['policy_id']!=kw['expected_policy_id']:raise ValueError('actual policy mismatch')
        calls.append(str(path));return {'policy_id':kw['expected_policy_id']}
    monkeypatch.setattr(module.formal,'audit',audit)
    monkeypatch.setattr(module,'read_execution_bundle',bundle)
    monkeypatch.setattr(module,'verify_native_adapter',lambda *a:None)
    return c,spec,proof,reseal,calls


def test_full_curve_uses_actual_endpoint_and_committed_policies(tmp_path,monkeypatch):
    c,spec,proof,_,calls=setup(tmp_path,monkeypatch);plan=module.build(c,{})
    assert calls[0]=='audit' and len(calls)==12
    assert [p['step'] for p in plan['checkpoints']]==module.formal.STEPS
    assert plan['checkpoints'][0]['source_step']==318
    assert plan['stage1_reference_endpoint']['policy_id']==plan['initial_policy_id']
    assert plan['grpo_endpoint']=={'step':634,'policy_id':'p634'}
    assert plan['target_tuning_allowed'] is False and plan['model_selection_performed'] is False
    assert plan['experiment']['selection_recordings']==9226
    path=tmp_path/'source-plan.json';write(path,plan)
    assert module.reconstruct(dict(source_plan=str(path),source_plan_sha256=module.file_hash(path),source_finalization=c))==plan


@pytest.mark.parametrize('fault',['audit-config','partial','exposure','online','transport','endpoint','reference',
    'step','contract','directory','manifest','publication','shards','tensor','adapter','audit'])
def test_rejects_wrong_origin_or_rehashed_bad_commits(tmp_path,monkeypatch,fault):
    c,spec,proof,reseal,_=setup(tmp_path,monkeypatch);package=Path(spec['output']);run=tmp_path/'collected/run'
    if fault=='audit-config':c['training_audit_config_sha256']='wrong'
    elif fault=='partial':proof['updates'].pop()
    elif fault=='exposure':proof['real_query_exposures']=12800
    elif fault=='online':proof['online_reference_calls']=1
    elif fault=='transport':proof['transport_sha256']='wrong'
    elif fault=='endpoint':write(Path(spec['parent']['output'])/'initial/endpoint-commit.json',dict(updates=288,joint_policy={'policy_id':'selected'}))
    elif fault=='reference':
        cfg=module.read(package/'config.json');cfg['reference_dataset']['identity']['policy_id']='other';write(package/'config.json',cfg)
    elif fault in ('step','contract','directory','manifest','shards'):
        path=run/'checkpoint-0064-committed.json';v=module.read(path)
        if fault=='step':v['updates']=63
        elif fault=='contract':v['contract']={}
        elif fault=='directory':v['joint_policy_directory']='../other'
        elif fault=='manifest':v['joint_policy_manifest_sha256']='wrong'
        else:v['rank_shards']={}
        write(path,v);reseal()
    elif fault=='publication':write(run/'policy-0064-committed.json',{});reseal()
    elif fault in ('tensor','adapter','audit'):
        def reject(*a,**kw):raise ValueError('independent evidence invalid')
        target=module.formal if fault=='audit' else module
        monkeypatch.setattr(target,{'tensor':'read_execution_bundle','adapter':'verify_native_adapter','audit':'audit'}[fault],reject)
    with pytest.raises(ValueError):module.build(c,{})


def test_modified_plan_cannot_be_used_even_after_rehash(tmp_path,monkeypatch):
    c,_,_,_,_=setup(tmp_path,monkeypatch);plan=module.build(c,{})
    plan['checkpoints'][0]['source_step']=288
    path=tmp_path/'plan.json';write(path,plan)
    with pytest.raises(ValueError,match='reconstruction'):
        module.reconstruct(dict(source_plan=str(path),source_plan_sha256=module.file_hash(path),source_finalization=c))
