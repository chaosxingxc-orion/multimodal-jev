import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tools.jev import launch_context_formal_training as module


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    package=tmp_path/'package';package.mkdir()
    c=dict(physical_devices=[4,5,6,7],start_update=0,stop_update=634,
        engineering_prefix=False,training_dispatch_allowed=True,
        encoded_audio_cache=module.PREFIX+'context-learning-cache-v1/run/cache',
        audio_mirror_root=module.PREFIX+'context-learning-audio-mirror-v1/run')
    for name,value in [('config.json',c),('service-config.json',{}),('PREPARED.json',{})]:
        (package/name).write_text(json.dumps(value))
    m=dict(remote_root=module.PREFIX+'context-rl-grpo-fixture',mount_source_container='a'*64,
           release_id='release',algorithm='grpo_v1')
    monkeypatch.setattr(module,'verify',lambda *args: m)
    monkeypatch.setattr(sys,'argv',['launch','--package',str(package),'--manifest-sha256','digest'])
    return package,c,m


def test_readonly_package_and_four_devices(prepared):
    _,c,m=prepared;args=module.command(m,c)
    devices=[args[i+1] for i,x in enumerate(args) if x=='--device']
    assert all('/dev/davinci'+str(i)+':/dev/davinci'+str(i) in devices for i in (4,5,6,7))
    assert not any('/dev/davinci0' in d for d in devices)
    assert m['mount_source_container']+':ro' in args
    assert str(Path(c['encoded_audio_cache']).parent)+':'+str(Path(c['encoded_audio_cache']).parent)+':ro' in args
    assert c['encoded_audio_cache']+':'+c['encoded_audio_cache'] in args
    for name in ('source','initial','evidence','RELEASE.json','config.json'):
        assert m['remote_root']+'/'+name+':'+m['remote_root']+'/'+name+':ro' in args
    assert module.IMAGE in args


def test_default_no_remote(prepared,monkeypatch,capsys):
    def forbidden(*args,**kw):raise AssertionError('remote call')
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    monkeypatch.setattr(module.subprocess,'check_output',forbidden)
    module.main()
    assert json.loads(capsys.readouterr().out)['status']=='validated_only_no_remote_calls'


@pytest.mark.parametrize('field,value',[('physical_devices',[0,1,2,3]),('stop_update',2),
    ('encoded_audio_cache','/mnt/cephfs/cache'),('engineering_prefix',True)])
def test_wrong_topology_or_scope_rejected(prepared,field,value):
    _,c,m=prepared;c[field]=value
    with pytest.raises(ValueError):module.command(m,c)


def test_exact_wait_failure_retains_handle_and_lease(prepared,tmp_path,monkeypatch):
    import fcntl
    _,c,m=prepared
    work=tmp_path/'work';work.mkdir();monkeypatch.chdir(work)
    local=work/'.local/decision-objective-research';local.mkdir(parents=True)
    lease=work/'.local/simple-final-v1/npu175-evaluation-training.lock';lease.parent.mkdir(parents=True)
    sys.argv.append('--launch');created=[];waits=[];sleeps=[]
    def held():
        with lease.open('a') as stream:
            with pytest.raises(BlockingIOError):fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def output(argv,**kw):
        held();cmd=shlex.split(argv[-1])
        if cmd[:3]==['docker','image','inspect']:return json.dumps([{'Id':module.IMAGE}])
        if cmd[:2]==['docker','inspect']:
            return json.dumps([dict(Id='a'*64,Image=module.IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False))])
        if cmd[:2]==['npu-smi','info']:return '\n'.join(f'No running processes found in NPU {i}' for i in (4,5,6,7))
        if cmd[:2]==['python3','-c']:return 'False' if '.exists()' in cmd[-1] else ''
        if cmd[:2]==['docker','create']:created.append(cmd);return 'b'*64
        return ''
    def run(argv,**kw):
        held()
        if argv[0]=='scp':return subprocess.CompletedProcess(argv,0,'','')
        cmd=shlex.split(argv[-1])
        if cmd[:2]==['docker','start']:
            assert json.loads((local/'context-rl-grpo-fixture-launch.json').read_text())['container_id']=='b'*64
            return subprocess.CompletedProcess(argv,255,'','uncertain start observation')
        assert cmd==['docker','wait','b'*64];waits.append(cmd)
        return subprocess.CompletedProcess(argv,255 if len(waits)==1 else 0,'' if len(waits)==1 else '0\n','')
    monkeypatch.setattr(module.subprocess,'check_output',output)
    monkeypatch.setattr(module.subprocess,'run',run)
    monkeypatch.setattr(module.time,'sleep',lambda seconds:(held(),sleeps.append(seconds)))
    module.main()
    assert len(created)==1 and len(waits)==2 and sleeps==[60]*10
    with pytest.raises(ValueError,match='existing attempt'):module.main()
