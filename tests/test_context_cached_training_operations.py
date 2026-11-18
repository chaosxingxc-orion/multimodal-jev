import copy
import json
from pathlib import Path
import shlex
import subprocess

import pytest

from tools.reports import context_cached_training_operations as module


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))


def fixture(tmp_path,monkeypatch):
    from tools.reports import context_release_deployment
    monkeypatch.setattr(context_release_deployment,'audit',lambda *a:{'fixture':'external release mount admission'})
    from test_prepare_context_cached_training import setup
    from tools.reports import prepare_context_cached_training as preparation
    spec,_,_,_=setup(tmp_path,monkeypatch)
    parent=Path(spec['parent']['output']);config=module.read(parent/'config.json')
    config.update(encoded_audio_cache='/data/input/cache',audio_mirror_root='/data/audio/run')
    config['pins']['/shared/model']='a'*64;write(parent/'config.json',config)
    m=module.read(parent/'PREPARED.json');m['image']=module.lifecycle.shared.IMAGE;write(parent/'PREPARED.json',m)
    source=parent/'source/src/multimodal_jev/local_runtime_storage.py';source.parent.mkdir(parents=True)
    source.write_text('from pathlib import Path\ndef check_paths(value):pass\n')
    write(parent/'RUNTIME-SOURCES.json',{str(p.relative_to(parent/'source')):module.file_hash(p)
        for p in (parent/'source').rglob('*.py')})
    for name in ('initial/endpoint-commit.json','initial/policy/MANIFEST.json'):write(parent/name,{'fixture':name})
    job=Path(spec['reference_job']['output'])
    for name in ('RUNTIME-SOURCES.json','initial/endpoint-commit.json','initial/policy/MANIFEST.json'):
        p=job/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((parent/name).read_bytes())
    spec['manifest_sha256']=preparation.prepare(spec,{})['manifest_sha256']
    package=Path(spec['output']);config=module.read(package/'config.json')
    source=dict(spec['reference_job'],parent=spec['parent'])
    reference=dict(reference_job=source,collection=spec['reference_job']['collection'])
    proof=dict(status='reference_job_and_same_endpoint_parity_audited',
        output_audit=dict(manifest_sha256=config['reference_dataset']['manifest_sha256'],
            policy_id=config['source_policy_id'],recordings=40513),
        numerical_admission=dict(source_policy_id=config['source_policy_id'],
            runtime_identity=config['dual_runtime_identity'],parity_container_id='b'*64))
    monkeypatch.setattr(module,'audit_reference',lambda *a:copy.deepcopy(proof))
    lease=tmp_path/'lease';lease.touch()
    c=dict(training_package=spec,reference_audit=reference,package=spec['output'],manifest_sha256=spec['manifest_sha256'],
        host='test-host',tag='cached-prefix-test',shared_root='/shared',driver_root='/usr/local/Ascend/driver',
        operations_root=str(tmp_path/'operations'),lease=str(lease),finite_training_dispatch_allowed=True)
    return c,proof


def test_real_package_admission_and_readonly_consumer_mount(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch)
    proof=module.admission(c);m,config,_=module.validate(c)
    assert proof['actor_updates']==2 and not proof['training_dispatch_allowed']
    argv=module.command(c,m,config);root=config['reference_dataset']['root']
    assert root+':'+root+':ro' in argv
    assert config['encoded_audio_cache']+':'+config['encoded_audio_cache']+':ro' in argv
    assert config['stop_update']==2


@pytest.mark.parametrize('fault',['parent','producer','collection','manifest','population','policy','runtime','source','endpoint'])
def test_mismatched_admission_never_reaches_training(tmp_path,monkeypatch,fault):
    c,proof=fixture(tmp_path,monkeypatch);ref=c['reference_audit']
    if fault=='parent':ref['reference_job']['parent']=dict(ref['reference_job']['parent'],manifest_sha256='wrong')
    elif fault=='producer':ref['reference_job']['manifest_sha256']='wrong'
    elif fault=='collection':ref['collection']='wrong'
    elif fault=='manifest':proof['output_audit']['manifest_sha256']='wrong'
    elif fault=='population':proof['output_audit']['recordings']=32
    elif fault=='policy':proof['numerical_admission']['source_policy_id']='wrong'
    elif fault=='runtime':proof['numerical_admission']['runtime_identity']={}
    else:
        p=Path(ref['reference_job']['output'])/('RUNTIME-SOURCES.json' if fault=='source' else 'initial/endpoint-commit.json')
        write(p,{'changed':True})
    with pytest.raises(ValueError):module.validate(c)


@pytest.mark.parametrize('fault',['formal','long','resume','online','checkpoint'])
def test_finite_boundary_remains_enforced_if_upstream_auditor_changes(tmp_path,monkeypatch,fault):
    c,_=fixture(tmp_path,monkeypatch);p=Path(c['package'])/'config.json';value=module.read(p)
    if fault=='formal':value['experiment_kind']='formal_source';value['training_dispatch_allowed']=True
    elif fault=='long':value['stop_update']=634
    elif fault=='resume':value['resume_root']='/data/prior'
    elif fault=='online':value['reference_cache']='/data/online'
    else:value['checkpoint_interval']=32
    write(p,value)
    monkeypatch.setattr(module,'audit_package',lambda *a:{'actor_updates':2})
    with pytest.raises(ValueError):module.admission(c)


def test_only_explicit_training_flag_authorizes_dispatch():
    with pytest.raises(ValueError,match='two-update'):
        module.launch({'finite_inference_dispatch_allowed':True,'reference_dispatch_allowed':True},{})


@pytest.mark.parametrize('exit_code',[0,7])
def test_shared_launch_keeps_exact_handle_and_readonly_reference(tmp_path,monkeypatch,exit_code):
    c,_=fixture(tmp_path,monkeypatch);shared=module.lifecycle.shared;calls=[];collected=[];cid='c'*64
    monkeypatch.setattr(shared,'check_paths',lambda _:None)
    def remote(argv,**kw):
        calls.append(argv)
        if argv[0]=='scp':return subprocess.CompletedProcess(argv,0,'','')
        args=shlex.split(argv[-1])
        if args[:3]==['docker','image','inspect']:out=json.dumps([{'Id':shared.IMAGE}])
        elif args[:2]==['npu-smi','info']:out='\n'.join(f'No running processes found in NPU {d}' for d in (4,5,6,7))
        elif args[:2]==['docker','create']:out=cid
        elif args[:2]==['docker','inspect']:out=json.dumps([{'Id':cid,'State':{'ExitCode':exit_code}}])
        else:out=''
        return subprocess.CompletedProcess(argv,0,out,'')
    monkeypatch.setattr(shared.subprocess,'run',remote)
    monkeypatch.setattr(shared,'wait_exact',lambda remote,handle:(calls.append(['wait',handle]) or exit_code))
    monkeypatch.setattr(shared,'collect',lambda handle,*a,**kw:(collected.append(handle) or {'fixture':'transport'}))
    import tools.reports.context_dual_probe_operations as runtime
    def check(*a):
        if exit_code:raise ValueError('terminal failure')
    monkeypatch.setattr(runtime,'check_runtime',check)
    if exit_code:
        with pytest.raises(ValueError,match='terminal failure'):module.launch(c,{})
    else:assert module.launch(c,{})['status']=='collected_independent_cached_training_audit_required'
    assert len(collected)==1 and calls.count(['wait',cid])==1
    commands=[shlex.split(v[-1]) for v in calls if v[0]=='ssh']
    creation=next(v for v in commands if v[:2]==['docker','create'])
    config=module.read(Path(c['package'])/'config.json');ref=config['reference_dataset']['root']
    assert ref+':'+ref+':ro' in creation
    preflight=next(v[-1] for v in commands if v[:2]==['python3','-c'])
    assert ref in preflight and "'cache':" not in preflight
    assert sum(v[:2]==['docker','create'] for v in commands)==1


def test_audit_composes_prerequisite_and_actual_training_checks(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(module,'audit_training',lambda *a:(calls.append(a) or {'fixture':'actual-training-check'}))
    result=module.audit(c,{})
    assert len(calls)==1 and not result['formal_training_admitted']
    assert result['prerequisites']['actor_updates']==2
