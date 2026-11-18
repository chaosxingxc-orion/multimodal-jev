"""Execute real host audit entry with simulated SSH/Docker; never uses NPUs."""
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import sys
import pytest
from multimodal_jev.resident_service_admission import IMAGE


def fixture(tmp_path,monkeypatch,*,prefix=True,status='exited',exit_code=137):
    import subprocess
    repo=tmp_path/'repo';target=repo/'tools/jev/launch_simple_stage2_audit.py'
    target.parent.mkdir(parents=True);target.write_text((Path(__file__).parents[1]/'tools/jev'/target.name).read_text())
    (repo/'src').mkdir();(repo/'src/example.py').write_text('VALUE=1\n')
    local=tmp_path/'state';local.mkdir();base='/data/xingchao/multimodal-jev/simple-final-v1'
    source=base+'/stage2-grpo_v1-resident-train-v1';tag='stage2-test-audit';remote=base+'/'+tag+'-v1'
    launch=dict(container_id='source',image=IMAGE,remote_root=source)
    (local/'stage2-test-train-launch.json').write_text(json.dumps(launch))
    (local/'prefix-prepared.json').write_text(json.dumps(dict(argv=[])))
    files={source+'/config.json':json.dumps(dict(stage2_initialization={})).encode(),source+'/launch.json':json.dumps(launch).encode()}
    events=[];created=[];mode={'exists':False}
    def ssh(argv,**kwargs):
        assert argv[:2]==['ssh','npu'];c=shlex.split(argv[2]);events.append(c)
        if c[:2]==['docker','inspect']:
            state=dict(Status=status,ExitCode=exit_code) if c[2]=='source' else dict(Status='running')
            return json.dumps([dict(Id=c[2],Image=IMAGE,State=state)])
        if c[0]=='cat':return files[c[1]].decode()
        if c[0]=='sha256sum':return hashlib.sha256(files[c[1]]).hexdigest()+' '+c[1]
        if c[:2]==['python3','-c']:return str(mode['exists'])
        if c[0] in ('mkdir','tar'):return ''
        if c[:2]==['docker','create']:created.append(c);return 'audit-container'
        if c[:2]==['docker','start']:return c[2]
        raise AssertionError(c)
    def copy(argv,**kwargs):
        assert argv[0]=='scp';events.append(['scp']);files[argv[2][4:]]=Path(argv[1]).read_bytes()
    monkeypatch.setattr(subprocess,'check_output',ssh);monkeypatch.setattr(subprocess,'run',copy)
    monkeypatch.setattr(sys,'argv',[str(target),'--training-tag','stage2-test-train','--audit-tag',tag,
        '--state-dir',str(local)]+(['--completed-updates','64'] if prefix else []))
    return target,local,remote,events,created,files,mode


@pytest.mark.parametrize('prefix',[True,False])
def test_audit_launch_pins_exact_exited_source_and_cpu_snapshot(tmp_path,monkeypatch,prefix):
    target,local,remote,events,created,files,_=fixture(tmp_path,monkeypatch,prefix=prefix,exit_code=137 if prefix else 0)
    monkeypatch.chdir(local);runpy.run_path(str(target),run_name='__main__')
    assert len(created)==1 and '--device' not in created[0]
    assert 'TORCH_DEVICE_BACKEND_AUTOLOAD=0' in created[0]
    assert '/mnt/cephfs/xingchao:/mnt/cephfs/xingchao:ro' in created[0]
    config=json.loads(files[remote+'/config.json'])
    assert config.get('completed_updates')==(64 if prefix else None)
    for p,digest in config['pins'].items():
        if '/source/' not in p:assert hashlib.sha256(files[p]).hexdigest()==digest
    assert 'verify_simple_resident_train.py' in files[remote+'/run.py'].decode()
    receipt=json.loads((local/'stage2-test-audit-launch.json').read_text())
    assert receipt['completed_updates']==(64 if prefix else 634)
    assert receipt['next_regular_observation_not_before']-receipt['created_unix']>=600
    assert events[0]==['docker','inspect','source']


@pytest.mark.parametrize('fault',['live','terminal_failed','existing_handle','existing_root','engineering_source'])
def test_audit_failure_never_creates_or_copies(tmp_path,monkeypatch,fault):
    target,local,_,events,created,files,mode=fixture(tmp_path,monkeypatch,
        prefix=fault!='terminal_failed',status='running' if fault=='live' else 'exited')
    if fault=='existing_handle':(local/'stage2-test-audit-created.json').write_text('{}')
    if fault=='existing_root':mode['exists']=True
    if fault=='engineering_source':
        key=next(k for k in files if k.endswith('/config.json'));files[key]=b'{"sampled_probe":{}}'
    with pytest.raises((ValueError,RuntimeError)):runpy.run_path(str(target),run_name='__main__')
    assert not created and not any(c[0] in ('mkdir','scp') for c in events)
