import fcntl
import json
from pathlib import Path
import shlex
import subprocess

import pytest

from tools.reports import context_train_parity_operations as module


def fixture(tmp_path,monkeypatch):
    from test_prepare_context_train_parity_job import fixture as parent_fixture,write
    from tools.reports import prepare_context_train_parity_job as preparation
    spec=parent_fixture(tmp_path,monkeypatch)
    parent=Path(spec['parent']['output']);m=preparation.read(parent/'PREPARED.json')
    m['image']=module.shared.IMAGE;write(parent/'PREPARED.json',m)
    training=preparation.read(parent/'config.json');training['pins']['/shared/model']='a'*64
    write(parent/'config.json',training)
    spec['manifest_sha256']=preparation.prepare(spec,{})['manifest_sha256']
    root=Path(spec['output'])
    # Source storage code is supplied by the real source closure in production.
    # This lifecycle test separately exercises real generated preflight below.
    source=root/'source/src/multimodal_jev/local_runtime_storage.py';source.parent.mkdir(parents=True)
    source.write_text('from pathlib import Path\ndef check_paths(value):pass\n')
    lease=tmp_path/'lease';lease.touch()
    c=dict(parity_package=spec,package=str(root),manifest_sha256=spec['manifest_sha256'],
        host='test-host',tag='train-parity-test',shared_root='/shared',driver_root='/usr/local/Ascend/driver',
        operations_root=str(tmp_path/'operations'),lease=str(lease),finite_inference_dispatch_allowed=True)
    # Package source is validated by preparation tests; mock only that upstream
    # prerequisite here, leaving deployment scope/storage/lease logic real.
    monkeypatch.setattr(module,'audit_package',lambda *a:{'fixture':'upstream'})
    monkeypatch.setattr(module.shared,'check_paths',lambda _:None)
    return c


@pytest.mark.parametrize('fault',[None,'source','host','tag','image','remote','overlap'])
def test_validation_is_bound_to_exact_package_and_scope(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch)
    if fault=='source':c['manifest_sha256']='wrong'
    elif fault=='host':c['host']='host;bad'
    elif fault=='tag':c['tag']='bad tag'
    elif fault in ('image','remote'):
        p=Path(c['package'])/'PREPARED.json';m=json.loads(p.read_text())
        m['image' if fault=='image' else 'remote_root']='wrong';p.write_text(json.dumps(m))
    elif fault=='overlap':c['operations_root']=c['package']+'/operations'
    if fault:
        with pytest.raises(ValueError):module.validate(c)
    else:
        m,config,_=module.validate(c)
        assert len(m['training_devices'])+len(m['inference_services'])==4
        assert config['macro_client']['rollout_queries']==32


@pytest.mark.parametrize('scenario',['success','failed','start-timeout','occupied','lease','repeat','disabled'])
def test_shared_launch_keeps_locks_exact_handle_collection_and_no_retry(tmp_path,monkeypatch,scenario):
    c=fixture(tmp_path,monkeypatch);calls=[];collected=[];cid='b'*64
    state=dict(Status='exited',ExitCode=7 if scenario=='failed' else 0,OOMKilled=False)
    def run(argv,**kw):
        calls.append(argv)
        if argv[0]=='scp':return subprocess.CompletedProcess(argv,0,'','')
        args=shlex.split(argv[-1])
        if args[:3]==['docker','image','inspect']:out=json.dumps([{'Id':module.shared.IMAGE}])
        elif args[:2]==['npu-smi','info']:
            out='' if scenario=='occupied' else '\n'.join(f'No running processes found in NPU {d}' for d in (4,5,6,7))
        elif args[:2]==['docker','create']:out=cid
        elif args[:2]==['docker','inspect']:out=json.dumps([dict(Id=cid,State=state)])
        elif args[:2]==['docker','start'] and scenario=='start-timeout':raise subprocess.TimeoutExpired(argv,55)
        else:out=''
        return subprocess.CompletedProcess(argv,0,out,'')
    monkeypatch.setattr(module.shared.subprocess,'run',run)
    monkeypatch.setattr(module.shared,'wait_exact',lambda remote,handle:(calls.append(['wait',handle]) or state['ExitCode']))
    monkeypatch.setattr(module.shared,'collect',lambda handle,*a,**kw:(collected.append(handle) or {'fixture':'transport'}))
    import tools.reports.context_dual_probe_operations as runtime
    def check(*a):
        if state['ExitCode']!=0:raise ValueError('failed exact runtime')
    monkeypatch.setattr(runtime,'check_runtime',check)
    lock=None
    if scenario=='lease':
        lock=Path(c['lease']).open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if scenario=='repeat':
        root=Path(c['operations_root']);root.mkdir();(root/(c['tag']+'-attempt.json')).write_text('{}')
    if scenario=='disabled':c['finite_inference_dispatch_allowed']=False
    try:
        if scenario in ('success','start-timeout'):
            result=module.launch(c,{})
            assert result['status']=='collected_independent_train_parity_audit_required'
            assert result['container_id']==cid
        else:
            with pytest.raises((ValueError,BlockingIOError)):module.launch(c,{})
    finally:
        if lock:lock.close()
    if scenario in ('lease','repeat','disabled'):assert not calls
    elif scenario=='occupied':assert not collected and not any(x[0]=='scp' for x in calls)
    else:
        assert len(collected)==1 and collected[0]['container_id']==cid
        assert calls.count(['wait',cid])==1
        remote=[shlex.split(x[-1]) for x in calls if x[0]=='ssh']
        assert sum(x[:2]==['docker','create'] for x in remote)==1
        assert sum(x[:2]==['docker','start'] for x in remote)==1
        assert not any(x[:2] in (['docker','restart'],['docker','rm']) for x in remote)
        preflight=next(x[-1] for x in remote if x[:2]==['python3','-c'])
        assert "'cache':" not in preflight and 'readonly' not in preflight
        assert "Path('/data/cache/input').is_dir()" not in preflight  # declared loop, not a writable cache check
        assert "['/data/cache/input', '/data/mirror/run']" in preflight
        base=Path(c['operations_root'])
        assert json.loads((base/(c['tag']+'-wait.json')).read_text())==dict(container_id=cid,exit_code=state['ExitCode'])
        assert (base/(c['tag']+'-collection.json')).exists()
