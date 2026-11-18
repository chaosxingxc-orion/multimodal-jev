import copy
import json
from pathlib import Path

import pytest

from tools.reports import context_infonce_selection as module
from tools.reports import collect_context_infonce_curve as collector
from tools.reports import collect_context_macro_source_curve as shared_curve
from multimodal_jev import local_checkpoint
from test_prepare_context_source_point_macro import setup as old_setup
from test_collect_context_macro_source_curve import setup as curve_setup


def plan_from(old):
    plan=copy.deepcopy(old)
    plan.update(schema='jev-infonce-source-evaluation-plan-v1',
        status='audited_source_policies_bound_evaluation_pending',algorithm='candidate_infonce',
        source_audit=dict(status='formal_infonce_artifacts_audited',world_size=4,real_exposures=81026,
                          updates=list(range(318))),target_tuning_allowed=False,model_selection_performed=False)
    for p,step in zip(plan['checkpoints'],module.STEPS,strict=True): p['step']=step
    plan['initial_policy_id']=plan['checkpoints'][0]['policy_id']
    plan['reference_endpoint']=dict(step=318,policy_id=plan['checkpoints'][-1]['policy_id'])
    plan['experiment'].update(task_stage=dict(updates=318),selection_recordings=9226,
        evaluation=dict(checkpoints=module.STEPS,primary='query_macro_set_f1',set_threshold=.5,
            secondary=['candidate_binary_nll','candidate_brier','top1_event_accuracy','top1_macro_f1']))
    return plan


def write(path,value):
    path.write_text(json.dumps(value))


def test_infonce_validator_accepts_own_origin_rejects_grpo(tmp_path,monkeypatch):
    from tools.reports import prepare_context_selection_macro as prep
    from tools.reports import context_source_point_macro_operations as grpo
    c,point,old,ops=old_setup(tmp_path,monkeypatch)
    r=prep.prepare(c,{},source_point=point,source_plan=old)
    package=Path(c['output'])/'package';plan=plan_from(old)
    write(package/'SOURCE-PLAN.json',plan);write(package/'TRAINING-AUDIT.json',plan['source_audit'])
    r.update(source_step=32,source_plan_sha256=module.file_hash(package/'SOURCE-PLAN.json'))
    for name in ('evaluation.json','service-0.json','service-1.json'):
        value=module.read(package/name)
        if name.startswith('service-'):
            value['policy'].update(max_batch_queries=32,max_batch_tokens=131072,max_decisions=60)
        for dep in ('SOURCE-PLAN.json','TRAINING-AUDIT.json'):
            value['pins'][c['remote_root']+'/package/'+dep]=module.file_hash(package/dep)
        write(package/name,value)
    write(package/'FILES.json',{str(p.relative_to(package)):module.file_hash(p)
        for p in package.rglob('*') if p.is_file() and p.name!='FILES.json'})
    r['files_sha256']=module.verify(package);write(package.parent/'PREPARED.json',r)
    operations=dict(ops,prepared=c['output'],manifest_sha256=r['files_sha256'],remote_root=c['remote_root'],
        cache_root=c['cache_root'],release=c['release'],release_sha256=c['release_sha256'])
    assert module.validate(operations)['source_step']==32
    with pytest.raises(ValueError):grpo.validate(operations)
    for fault in ('budget','endpoint','population','threshold','algorithm','origin','missing'):
        bad=copy.deepcopy(plan)
        if fault=='budget':bad['source_audit']['world_size']=8
        if fault=='endpoint':bad['reference_endpoint']['step']=288
        if fault=='population':bad['experiment']['selection_recordings']=10
        if fault=='threshold':bad['experiment']['evaluation']['set_threshold']=.6
        if fault=='algorithm':bad['algorithm']='dual_credit_grpo_floor_clipped_v1'
        if fault=='origin':bad['initial_policy_id']='other'
        if fault=='missing':bad['checkpoints'].pop()
        with pytest.raises(ValueError):module.validate_plan(bad)


def test_complete_infonce_curve_preserves_endpoint_separately_from_best(tmp_path,monkeypatch):
    c,old,operations,audits,calls=curve_setup(tmp_path,monkeypatch)
    plan=plan_from(old)
    new_entries=[]
    for entry,checkpoint in zip(c['evaluations'],plan['checkpoints'],strict=True):
        entry['step']=checkpoint['step'];new_entries.append(entry)
        package=Path(operations[entry['local']]['prepared'])/'package'
        receipt=module.read(package.parent/'PREPARED.json');receipt['source_step']=entry['step']
        write(package.parent/'PREPARED.json',receipt);write(package/'SOURCE-PLAN.json',plan)
    write(Path(c['source_plan']),plan);c['source_plan_sha256']=module.file_hash(c['source_plan'])
    monkeypatch.setattr(collector,'reconstruct_plan',lambda c:copy.deepcopy(plan))
    monkeypatch.setattr(collector,'audit_collected',shared_curve.audit_point)
    monkeypatch.setattr(collector,'check_paths',lambda p:None)
    result=collector.collect(c,{})
    assert result['roles']['source_selected']==dict(step=64,policy_id='p128')
    assert result['roles']['stage2_initial_and_reference']==dict(step=318,policy_id='p634')
    assert result['points']==11 and not result['target_inference_performed']
    source=module.read(result['source_evaluation']['path'])
    assert source['algorithm']=='candidate_infonce' and 'GRPO' not in str(source['roles'])


def test_source_plan_must_match_fresh_independent_reconstruction(tmp_path,monkeypatch):
    c,old,*_=curve_setup(tmp_path,monkeypatch);plan=plan_from(old);plan['config_provenance']={}
    write(Path(c['source_plan']),plan);c['source_plan_sha256']=module.file_hash(c['source_plan'])
    c['source_finalization']={'test':'explicit actual origin'}
    calls=[]
    def build(config,provenance):
        calls.append(config);return copy.deepcopy(plan)
    monkeypatch.setattr(module,'build',build)
    assert module.reconstruct_plan(c)==plan and len(calls)==1
    monkeypatch.setattr(module,'build',lambda *a:dict(plan,base_id='changed'))
    with pytest.raises(ValueError,match='independently audited'):module.reconstruct_plan(c)


def test_prepare_exports_real_adapter_without_mutating_template(tmp_path,monkeypatch):
    from test_context_policy_artifacts import fixture as policy_fixture
    from tools.reports import prepare_context_selection_macro as prep
    from multimodal_jev import local_runtime_storage
    c,point,old,ops=old_setup(tmp_path,monkeypatch)
    prepared=prep.prepare(c,{},source_point=point,source_plan=old)
    template=Path(c['output']);package=template/'package'
    # Restore real artifact readers after the legacy packaging fixture substitutes.
    import importlib
    from multimodal_jev import context_policy_artifacts as artifacts
    importlib.reload(artifacts)
    monkeypatch.setattr(module,'read_execution_bundle',artifacts.read_execution_bundle)
    monkeypatch.setattr(module,'verify_native_adapter',artifacts.verify_native_adapter)
    real=tmp_path/'real';real.mkdir()
    _,snapshot,policy_id,contract=policy_fixture(real,monkeypatch)
    plan=plan_from(old)
    for p in plan['checkpoints']:
        p.update(snapshot=str(snapshot),policy_id=policy_id,manifest_sha256=module.file_hash(snapshot/'MANIFEST.json'))
    plan.update(initial_policy_id=policy_id,source_contract=contract,base_id='a'*64,
                reference_endpoint=dict(step=318,policy_id=policy_id))
    training=tmp_path/'formal';training.mkdir()
    write(training/'config.json',dict(release=c['release'],release_sha256=c['release_sha256']))
    plan['source_operations'].update(package=str(training),remote_root='/data/formal')
    plan['training_config_sha256']=module.file_hash(training/'config.json')
    # A template is code/input provenance only; replace its historic receipt type
    # for this fixture, leaving the real policy export and sealing unsimulated.
    receipt=module.read(template/'PREPARED.json');receipt['status']='macro_selection_prepared_only'
    receipt['runtime_identity']['processor_sha256']='same'
    write(template/'PREPARED.json',receipt)
    (package/'SOURCE-PLAN.json').unlink()
    write(package/'FILES.json',{str(p.relative_to(package)):module.file_hash(p)
        for p in package.rglob('*') if p.is_file() and p.name!='FILES.json'})
    receipt['files_sha256']=module.verify(package)
    write(template/'PREPARED.json',receipt)
    binding=Path(c['binding_root'])
    write(binding/'population.json',[]);write(binding/'candidate-ids.json',[])
    monkeypatch.setattr(module,'inputs',lambda c:([],{},[]))
    monkeypatch.setattr(module,'load_evidence',lambda *a,**kw:([],{}))
    monkeypatch.setattr(module,'audit_receipts',lambda *a:{})
    monkeypatch.setattr(module,'processor_cache_identity',lambda p:receipt['runtime_identity']['processor_sha256'])
    monkeypatch.setattr(module,'check_paths',lambda p:None)
    monkeypatch.setattr(local_runtime_storage,'check_paths',lambda p:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda p:None)
    # Only the release manifest is stubbed; export_native_adapter uses real BF16
    # tensors and verify_native_adapter independently rederives every matrix.
    original_read=module.read
    monkeypatch.setattr(module,'read',lambda p: {} if str(p)==c['release']+'/MANIFEST.json' else original_read(p))
    before=module.verify(package)
    conf=dict(c,runtime_template=str(template),runtime_manifest_sha256=before,
        source_step=32,output=str(tmp_path/'infonce'),remote_root='/data/infonce')
    result=module.prepare_point(conf,{},plan)
    assert result['policy_id']==policy_id and result['source_step']==32
    new=Path(conf['output'])/'package'
    bundle=artifacts.read_execution_bundle(new/'policy',expected_policy_id=policy_id,
        expected_contract=contract,expected_base_id='a'*64)
    assert artifacts.verify_native_adapter(new/'adapter',bundle)['policy_id']==policy_id
    assert module.verify(package)==before
    assert (new/'RUNTIME-SOURCES.json').read_bytes()==(package/'RUNTIME-SOURCES.json').read_bytes()
    for i in range(2):
        service=original_read(new/f'service-{i}.json')
        assert service['policy']['policy_id']==policy_id
        assert service['policy']['contract']==contract
        assert service['policy']['max_batch_queries']==32
