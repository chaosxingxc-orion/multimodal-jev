"""Actual shared suffix state machine with explicit model/transport substitutes."""
from pathlib import Path
import json

import pytest

from test_context_macro_formal_audit import fixture
from tools.reports import audit_context_macro_training as shared
from tools.reports import audit_context_macro_formal_recovery as module


def resumed(tmp_path, monkeypatch, *, step=320, fault=None):
    package, collection, checks = fixture(tmp_path, monkeypatch)
    run=collection/'run';journal=collection/'journal/formal'
    for p in run.glob('policy-*-committed.json'):
        if int(p.name.split('-')[1])<=step:p.unlink()
    for p in run.rglob('checkpoint-*'):
        if int(p.name.split('-')[1].split('.')[0])<=step:p.unlink()
    for p in journal.glob('batch-*'):
        if int(p.name.split('-')[1])>=634-step:p.rmdir()
    original=shared.read
    def read(path):
        path=Path(path);name=path.name
        if name=='RECOVERY.json':return dict(initial_behavior_policy_id=f'p{step}')
        if name=='restored.json':return dict(exact_masters=True,exact_adam=fault!='adam',
                                             completed_updates=step)
        value=original(path)
        if path==package/'training.json':
            value=dict(value,start_update=step,source_root=str(package),
                       resume_root=str(package/'resume'),
                       initial_adapter=str(package/'resume'/f'adapter-{step:04d}'))
        elif name=='result.json':value=dict(value,start_update=step)
        elif name=='ready.json':value=dict(value,policy_id='p0' if fault=='startup' else f'p{step}')
        elif name=='update-0634.json' and fault=='tail':value=dict(value,real_queries=128)
        elif name=='activation-0321.json' and fault=='activation':value=dict(value,policy_id='p320')
        return value
    monkeypatch.setattr(shared,'read',read)
    return package,collection,checks


def test_segment_covers_only_remaining_steps_and_sparse_boundaries(tmp_path,monkeypatch):
    package,collection,checks=resumed(tmp_path,monkeypatch)
    result=module.audit_segment(package,collection)
    assert [u['update'] for u in result['updates']]==list(range(320,634))
    assert checks==[384,448,512,576,634]
    assert result['status']=='formal_dp2_resumed_segment_outputs_audited'
    assert result['fresh_process_replay_verified'] is False


@pytest.mark.parametrize('fault',['adam','startup','tail','activation'])
def test_suffix_corruption_rejected(tmp_path,monkeypatch,fault):
    package,collection,_=resumed(tmp_path,monkeypatch,fault=fault)
    with pytest.raises(ValueError):module.audit_segment(package,collection)


def test_segment_rejects_prefix_artifacts_in_new_attempt(tmp_path,monkeypatch):
    package,collection,_=resumed(tmp_path,monkeypatch)
    (collection/'run/policy-0320-committed.json').touch()
    with pytest.raises(ValueError):module.audit_segment(package,collection)


def test_composition_visits_suffix_after_prefix_admission(tmp_path,monkeypatch):
    package,collection,_=resumed(tmp_path,monkeypatch)
    prefix=dict(status='formal_dp2_prefix_outputs_audited',updates=[dict(update=i) for i in range(320)])
    monkeypatch.setattr(module,'bind_prefix',lambda p:(prefix,320))
    result=module.audit_outputs(package,collection)
    assert [u['update'] for u in result['updates']]==list(range(634))
    assert result['prefix_learning_history_verified'] is True
    assert result['training_dispatch_allowed'] is False


@pytest.mark.parametrize('fault',['gap','overlap','status'])
def test_composition_rejects_forged_coverage(monkeypatch,fault):
    prefix=dict(status='formal_dp2_prefix_outputs_audited',updates=[dict(update=i) for i in range(64)])
    segment=dict(status='formal_dp2_resumed_segment_outputs_audited',updates=[dict(update=i) for i in range(64,634)])
    if fault=='gap':prefix['updates'].pop()
    if fault=='overlap':segment['updates'].insert(0,dict(update=63))
    if fault=='status':segment['status']='finite_dp2_training_audited'
    monkeypatch.setattr(module,'bind_prefix',lambda p:(prefix,64))
    monkeypatch.setattr(module,'audit_segment',lambda *a:segment)
    with pytest.raises(ValueError):module.audit_outputs('package','collection')


def sealed(tmp_path,monkeypatch):
    from test_prepare_context_macro_sparse_resume import fixture as prepare_fixture
    from tools.reports import prepare_context_macro_sparse_resume as prepare
    c,parent=prepare_fixture(tmp_path,monkeypatch)
    prepare.prepare(c,{})
    return Path(c['output'])/'package',parent


def test_real_sealed_bundle_rebinds_prefix_and_preserves_configuration(tmp_path,monkeypatch):
    package,_=sealed(tmp_path,monkeypatch)
    prefix,step=module.bind_prefix(package)
    assert step==64 and len(prefix['updates'])==64


@pytest.mark.parametrize('fault',['asset','objective','prefix','policy','source','parent_transport'])
def test_resealed_bundle_cannot_replace_parent_evidence(tmp_path,monkeypatch,fault):
    from multimodal_jev.baseline import file_hash
    package,parent=sealed(tmp_path,monkeypatch)
    if fault=='asset':(package/'resume/adapter-0064/adapter.safetensors').write_bytes(b'other')
    if fault=='objective':
        p=package/'training.json';v=json.loads(p.read_text());v['global_batch']=192;p.write_text(json.dumps(v))
    if fault in ('prefix','policy'):
        p=package/'RECOVERY.json';v=json.loads(p.read_text())
        if fault=='prefix':v['prefix_audit']['updates'].pop()
        else:v['initial_behavior_policy_id']='other'
        p.write_text(json.dumps(v))
    if fault=='source':(package/'model_code.py').write_text('changed')
    if fault=='parent_transport':
        (Path(parent['operations_root'])/'collected/run/rollout-0000.json').write_text('{}')
    (package/'FILES.json').unlink()
    (package/'FILES.json').write_text(json.dumps({str(p.relative_to(package)):file_hash(p)
        for p in package.rglob('*') if p.is_file()}))
    with pytest.raises(ValueError):module.bind_prefix(package)


def test_fixed_report_preserves_inputs_and_refuses_overwrite(tmp_path,monkeypatch):
    from multimodal_jev.local_config import load_config
    local=tmp_path/'macros.json'
    local.write_text(json.dumps({'schema':'jev-local-macros-v1','macros':{
        'FORMAL_RECOVERY_PACKAGE':str(tmp_path/'package'),
        'FORMAL_RECOVERY_COLLECTION':str(tmp_path/'collection'),
        'FORMAL_RECOVERY_OUTPUT_AUDIT':str(tmp_path/'report.json')}}))
    c,provenance=load_config('configs/portable/operations/audit-context-macro-formal-recovery.json',local)
    monkeypatch.setattr(module,'check_paths',lambda *a:None)
    monkeypatch.setattr(module,'audit_outputs',lambda *a:dict(
        status='formal_dp2_recovered_outputs_audited',resumed_runtime_transport_verified=False))
    result=module.report(c,provenance)
    assert json.loads(Path(c['output']).read_text())==result
    assert result['resumed_runtime_transport_verified'] is False
    with pytest.raises(ValueError):module.report(c,provenance)
    c['output']=str(tmp_path/'package/report.json')
    with pytest.raises(ValueError):module.report(c,provenance)
