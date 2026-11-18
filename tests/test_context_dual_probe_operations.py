import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tools.reports import context_dual_probe_operations as m
from test_context_batch_probe_launcher_candidate import package


def write(p,value):
    Path(p).write_text(json.dumps(value))


def fixture(tmp_path):
    prepared=package(tmp_path);p=prepared/'package'
    storage=p/'src/multimodal_jev/local_runtime_storage.py'
    storage.parent.mkdir(parents=True);storage.write_text('# fixture preflight source\n')
    c=dict(prepared=str(prepared),remote_root='/local/dual',tag='dual-v1',host='npu',
        model='/shared/model',driver_root='/driver',source=str(Path(__file__).resolve().parents[1]),
        priority_root=str(tmp_path/'priority'),operations_root=str(tmp_path/'operations'),
        lease=str(tmp_path/'lease'),finite_inference_dispatch_allowed=False)
    conf=json.loads((p/'config.json').read_text())
    conf['policy'].update(model=c['model'],snapshot=c['remote_root']+'/package/policy',
        adapter=c['remote_root']+'/package/adapter',candidate_runtime=m.RUNTIME,output_protocol=m.WIRE)
    conf.update(source_root=c['remote_root']+'/package',inputs=c['remote_root']+'/package/inputs.pt')
    conf['pins']={c['remote_root']+'/package/'+str(f.relative_to(p)):m.sha(f) for f in p.rglob('*')
                  if f.is_file() and f.name not in ('FILES.json','config.json')}
    write(p/'config.json',conf)
    seal(c)
    return c


def seal(c):
    p=Path(c['prepared'])/'package'
    write(p/'FILES.json',{str(f.relative_to(p)):m.sha(f) for f in p.rglob('*') if f.is_file() and f.name!='FILES.json'})
    c['package_manifest_sha256']=m.sha(p/'FILES.json')
    receipt=dict(status='sealed_not_launched',image=m.IMAGE,remote_root=c['remote_root'],
        tag=c['tag'],host=c['host'],training_dispatch_allowed=False,files_sha256=m.sha(p/'FILES.json'),
        argv=m.rebind_argv(m.command(c['model']),m.ROOT,c['remote_root'],c['model'],c['model'],
                          c['driver_root'],7,c['tag']))
    write(p.parent/'PREPARED.json',receipt)


def priority(c):
    root=Path(c['priority_root']);root.mkdir()
    lock=root/'chain.lock';lock.touch()
    result=root/'result.json';write(result,dict(passed=True))
    stages=[dict(name='minds',results=[dict(path=str(result),equals=dict(passed=True))])]
    write(root/'input.json',dict(stages=stages,owner_locks=[str(lock)]))
    write(root/'status.json',dict(phase='complete_predeclared_chain',completed=['minds']))
    write(root/'minds-complete.json',dict(outputs={str(result):m.sha(result)}))
    return root


def test_exact_package_command_and_pins(tmp_path):
    c=fixture(tmp_path);r,conf=m.validate(c)
    assert conf['policy']['output_protocol']==m.WIRE
    assert r['files_sha256']==m.verify(Path(c['prepared'])/'package')


@pytest.mark.parametrize('fault',['command','protocol','device','pins','inputs','host','root'])
def test_tampered_or_incomplete_release_rejected(tmp_path,fault):
    c=fixture(tmp_path);p=Path(c['prepared'])/'package'
    conf=json.loads((p/'config.json').read_text())
    if fault=='protocol':conf['policy']['output_protocol']='prob-only'
    if fault=='device':conf['policy']['physical_devices']=[6]
    if fault=='pins':conf['pins'].pop(next(iter(conf['pins'])))
    if fault=='inputs':conf['input_sha256']='f'*64
    write(p/'config.json',conf);seal(c)
    if fault=='command':
        r=json.loads((p.parent/'PREPARED.json').read_text());r['argv'].insert(2,'--privileged')
        write(p.parent/'PREPARED.json',r)
    if fault=='host':c['host']='-ProxyCommand=bad'
    if fault=='root':c['remote_root']='/local/../old'
    with pytest.raises(ValueError):m.validate(c)


def test_real_predecessor_locks_and_outputs_required(tmp_path):
    c=fixture(tmp_path);root=priority(c);m.check_priority(c)
    with (root/'chain.lock').open('r') as held:
        m.fcntl.flock(held,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):m.check_priority(c)
    write(root/'result.json',dict(passed=False))
    with pytest.raises(ValueError,match='evidence changed'):m.check_priority(c)


def test_idle_window_skips_chain_but_never_skips_exclusive_lease(tmp_path,monkeypatch):
    c=fixture(tmp_path);c.update(priority_mode='idle_resource_window',finite_inference_dispatch_allowed=True)
    m.check_priority(c)
    Path(c['lease']).touch()
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**kw:pytest.fail('remote call while training owns lease'))
    with Path(c['lease']).open('r') as held:
        m.fcntl.flock(held,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):m.launch(c,{})


def test_unknown_priority_mode_rejected(tmp_path):
    c=fixture(tmp_path);c['priority_mode']='ignore_all'
    with pytest.raises(ValueError,match='unknown resource'):m.check_priority(c)


def test_no_remote_calls_before_admission_and_priority(tmp_path,monkeypatch):
    c=fixture(tmp_path)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**kw:pytest.fail('unexpected remote call'))
    with pytest.raises(ValueError,match='not admitted'):m.launch(c,{})
    root=priority(c);c['finite_inference_dispatch_allowed']=True
    write(root/'status.json',dict(phase='executing',completed=[]))
    with pytest.raises(ValueError,match='priority'):m.launch(c,{})


def test_ambiguous_start_waits_exact_handle_and_preserves_failed_collection(tmp_path,monkeypatch):
    import tools.jev.collect_context_rl_selection as transport
    c=fixture(tmp_path);priority(c);Path(c['lease']).touch()
    c['finite_inference_dispatch_allowed']=True
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    calls=[];cid='a'*64
    def remote(argv,**kwargs):
        calls.append(argv)
        if argv[0]=='scp':return NS(returncode=0)
        cmd=argv[-1]
        if 'image inspect' in cmd:return NS(returncode=0,stdout=json.dumps([dict(Id=m.IMAGE)]))
        if 'npu-smi' in cmd:return NS(returncode=0,stdout='No running processes found in NPU 7')
        if cmd.startswith('docker create'):return NS(returncode=0,stdout=cid+'\n')
        if cmd.startswith('docker start'):
            handle=Path(c['operations_root'])/(c['tag']+'-launch.json')
            assert json.loads(handle.read_text())['container_id']==cid
            raise m.subprocess.TimeoutExpired(argv,55)
        if cmd.startswith('docker wait'):return NS(returncode=0,stdout='7\n')
        if cmd.startswith('docker inspect'):return NS(returncode=0,stdout=json.dumps([dict(Id=cid)]))
        return NS(returncode=0,stdout='')
    monkeypatch.setattr(m.subprocess,'run',remote)
    collected=[]
    def collect(record,output,host):
        collected.append((record['container_id'],host));Path(output).mkdir()
        return dict(state=dict(ExitCode=7))
    monkeypatch.setattr(transport,'collect',collect)
    def failed_audit(*args):raise ValueError('terminal failed')
    monkeypatch.setattr(m,'audit',failed_audit)
    with pytest.raises(ValueError,match='terminal failed'):m.launch(c,{})
    assert collected==[(cid,'npu')]
    assert sum(a[-1].startswith('docker create') for a in calls)==1
    assert sum(a[-1].startswith('docker wait '+cid) for a in calls)==1
    checks=[a[-1] for a in calls if a[0]=='ssh' and a[-1].startswith('python3 ')]
    assert len(checks)==2 and all(c.startswith('python3 -B -c ') for c in checks)
    n=len(calls)
    with pytest.raises(ValueError,match='existing attempt'):m.launch(c,{})
    assert len(calls)==n


def test_pinned_independent_auditor_uses_dual_contract(tmp_path):
    c=fixture(tmp_path);ns=m.auditor_namespace(c)
    from tools.reports.context_dual_measurement_candidate import audit_dual_measurements
    assert ns['audit_measurements'] is audit_dual_measurements
    assert ns['validate'](c['prepared'])[0]['tag']==c['tag']


@pytest.mark.parametrize('fault',[None,'privileged','device','mount','env','command','oom'])
def test_actual_runtime_scope_checked(tmp_path,fault):
    c=fixture(tmp_path);receipt,_=m.validate(c);argv=receipt['argv']
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    handle=dict(argv=argv,container_id='a'*64)
    obs=dict(Id=handle['container_id'],Image=m.IMAGE,
        State=dict(Status='exited',ExitCode=0,OOMKilled=False),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(m.IMAGE)+1:],Env=values('-e')),
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),
            Devices=[dict(PathOnHost=v.split(':')[0],PathInContainer=v.split(':')[1],CgroupPermissions='rwm')
                     for v in values('--device')]))
    if fault=='privileged':obs['HostConfig']['Privileged']=True
    if fault=='device':obs['HostConfig']['Devices'].pop()
    if fault=='mount':obs['HostConfig']['Binds'].pop()
    if fault=='env':obs['Config']['Env']=['TMPDIR=/shared']
    if fault=='command':obs['Config']['Cmd']=['python','other.py']
    if fault=='oom':obs['State']['OOMKilled']=True
    if fault is None:m.check_runtime(receipt,handle,obs)
    else:
        with pytest.raises(ValueError):m.check_runtime(receipt,handle,obs)
