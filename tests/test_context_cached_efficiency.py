import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.reports import context_cached_efficiency as module
from test_prepare_context_cached_training import write,reseal


def fixture(tmp_path,monkeypatch):
    from test_context_cached_training_operations import fixture as baseline
    ops,_=baseline(tmp_path,monkeypatch)
    # External NPU gates are explicit fixtures. Real parent/reference binding,
    # immutable package reconstruction and generated launcher run below.
    prefix=dict(training_package=ops['training_package'],reference_audit=ops['reference_audit'],deployment=ops)
    parent=Path(ops['training_package']['parent']['output'])
    c=module.read(parent/'config.json')
    c.update(global_batch=128,microbatch=8,batch_config=dict(training_devices=2,inference_instances=2,inference_batch_queries=32))
    write(parent/'config.json',c)
    # Bind the prefix fixture to the same approved settings.
    actual=module.read(Path(ops['package'])/'config.json');actual.update({k:c[k] for k in ('global_batch','microbatch','batch_config')})
    write(Path(ops['package'])/'config.json',actual)
    tail=copy.deepcopy(prefix);tail['training_package']['acceptance_kind']='cached_reference_tail_admission'
    recovery=dict(training_package=dict(parent_audit=prefix))
    proofs={'engineering_prefix':'finite_cached_reference_training_outputs_audited',
        'cached_reference_tail_admission':'finite_cached_reference_tail_outputs_audited'}
    def gate(c,p):return dict(status=proofs[c['training_package'].get('acceptance_kind','engineering_prefix')])
    monkeypatch.setattr(module.lifecycle,'audit',gate)
    monkeypatch.setattr(module.recovery,'audit',lambda *a:dict(status='finite_cached_reference_recovery_audited',fresh_process_npu_resume_verified=True))
    spec=dict(parent=ops['training_package']['parent'],reference_job=ops['training_package']['reference_job'],
        acceptance=dict(prefix=prefix,tail=tail,recovery=recovery),acceptance_kind='efficiency',
        output=str(tmp_path/'efficiency'),remote_root='/data/efficiency')
    return spec,ops


def test_efficiency_reuses_runtime_and_real_readonly_reference(tmp_path,monkeypatch):
    c,ops=fixture(tmp_path,monkeypatch);parent=Path(c['parent']['output'])
    before={str(p):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    result=module.package_action(c,{})
    c['manifest_sha256']=result['manifest_sha256']
    proof=module.package_action(c,{},audit=True)
    assert proof['actor_updates']==100 and proof['npu_executed'] is False
    assert before=={str(p):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    root=Path(c['output']);cfg=module.read(root/'config.json')
    assert cfg['stop_update']==100 and cfg['start_update']==0 and cfg['checkpoint_interval']==1
    assert cfg['experiment_kind']=='efficiency' and cfg['training_dispatch_allowed'] is False
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(parent/'RUNTIME-SOURCES.json').read_bytes()
    calls=[]
    monkeypatch.setitem(sys.modules,'context_process_supervisor',SimpleNamespace(supervise_services=lambda **kw:(calls.append(kw) or {'passed':True})))
    from multimodal_jev import local_runtime_storage
    monkeypatch.setattr(local_runtime_storage,'require_training_storage',lambda _:None)
    script=(root/'run.py').read_text().replace("Path('/jev-tmp')",'Path('+repr(str(tmp_path))+')')
    with pytest.raises(SystemExit) as e:exec(compile(script,'<efficiency>','exec'),{'__file__':str(root/'run.py')})
    assert e.value.code==0 and len(calls)==1
    assert calls[0]['actor_env']['ASCEND_RT_VISIBLE_DEVICES']=='0,1'
    assert [s['env']['ASCEND_RT_VISIBLE_DEVICES'] for s in calls[0]['services']]==['2','3']
    cfg['stop_update']=101;write(root/'config.json',cfg)
    with pytest.raises(ValueError):exec(compile(script,'<efficiency>','exec'),{'__file__':str(root/'run.py')})
    assert len(calls)==1


@pytest.mark.parametrize('fault',['missing','tail','recovery','parent','reference','runtime','width','not-resumed'])
def test_failed_or_unrelated_admission_prevents_package(tmp_path,monkeypatch,fault):
    c,_=fixture(tmp_path,monkeypatch)
    if fault=='missing':del c['acceptance']['tail']
    elif fault=='tail':c['acceptance']['tail']['training_package']['acceptance_kind']='engineering_prefix'
    elif fault=='recovery':c['acceptance']['recovery']['training_package']['parent_audit']={}
    elif fault=='parent':c['parent']=dict(c['parent'],manifest_sha256='wrong')
    elif fault=='reference':c['reference_job']=dict(c['reference_job'],manifest_sha256='wrong')
    elif fault=='runtime':c['acceptance']['tail']['training_package']['parent']['manifest_sha256']='wrong'
    elif fault=='width':
        p=Path(c['acceptance']['prefix']['training_package']['output'])/'config.json';v=module.read(p);v['microbatch']=16;write(p,v)
    else:monkeypatch.setattr(module.recovery,'audit',lambda *a:dict(status='finite_cached_reference_recovery_audited',fresh_process_npu_resume_verified=False))
    with pytest.raises(ValueError):module.package_action(c,{})
    assert not Path(c['output']).exists()


@pytest.mark.parametrize('fault',['steps','objective','reference','source','runner','cadence'])
def test_rehashed_efficiency_changes_are_independently_rejected(tmp_path,monkeypatch,fault):
    c,_=fixture(tmp_path,monkeypatch);module.package_action(c,{})
    root=Path(c['output']);v=module.read(root/'config.json')
    if fault=='steps':v['stop_update']=101
    elif fault=='objective':v['stage2_objective']['binary_nll_coefficient']=0.
    elif fault=='reference':v['reference_dataset']['manifest_sha256']='wrong'
    elif fault=='cadence':v['checkpoint_interval']=100
    elif fault=='source':(root/'source/test.py').write_text('changed')
    else:(root/'run.py').write_text('pass')
    write(root/'config.json',v);reseal(c)
    with pytest.raises(ValueError):module.package_action(c,{},audit=True)


def test_dispatch_requires_efficiency_flag():
    with pytest.raises(ValueError,match='100-step'):module.launch({'finite_training_dispatch_allowed':True},{})


def test_lifecycle_reconstructs_admitted_package_and_refuses_wrong_host(tmp_path,monkeypatch):
    c,prior=fixture(tmp_path,monkeypatch)
    c['manifest_sha256']=module.package_action(c,{})['manifest_sha256']
    ops=dict(prior,training_package=c,package=c['output'],manifest_sha256=c['manifest_sha256'],
        operations_root=str(tmp_path/'efficiency-ops'),tag='efficiency-test',efficiency_dispatch_allowed=True)
    m,config,_=module.validate(ops)
    assert config['stop_update']==100 and m['experiment']['updates']==100
    ops['host']='other'
    with pytest.raises(ValueError,match='same host'):module.validate(ops)


def test_launch_uses_efficiency_gate_before_any_remote_work(tmp_path,monkeypatch):
    calls=[]
    def refuse(c):calls.append(c);raise ValueError('actual finite gates pending')
    monkeypatch.setattr(module,'validate',refuse)
    with pytest.raises(ValueError,match='finite gates pending'):
        module.launch({'efficiency_dispatch_allowed':True},{})
    assert len(calls)==1 and calls[0]['finite_training_dispatch_allowed'] is True


def test_full_auditor_still_rejects_missing_results(tmp_path,monkeypatch):
    # Real cache verification across 100 updates then actual shared audit. It
    # must not pass just because the reference matches and the count is 100.
    from test_audit_context_cached_training import reference
    ref,c,receipt,batch=reference(tmp_path,monkeypatch)
    root=tmp_path/'package';collection=tmp_path/'collection'
    write(root/'config.json',dict(experiment_kind='efficiency',start_update=0,stop_update=100,
        checkpoint_interval=1,engineering_prefix=True,training_dispatch_allowed=False,reference_dataset=c))
    write(root/'PREPARED.json',{})
    for i in range(100):
        write(collection/f'run/rollout-{i:04d}.json',dict(receipt,update=i))
        write(collection/f'run/batch-{i:04d}.json',batch)
    with pytest.raises((KeyError,FileNotFoundError)):
        module.audit_outputs(root,collection,reference_root=ref)
