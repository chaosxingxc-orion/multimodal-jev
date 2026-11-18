import json
from pathlib import Path

import pytest

from tools.reports import context_dual_training_operations as m


def fixture(tmp_path):
    package=tmp_path/'package';(package/'source').mkdir(parents=True)
    (package/'source/code.py').write_text('pass\n')
    (package/'run.py').write_text('pass\n')
    m.write(package/'RUNTIME-SOURCES.json',{'code.py':m.sha(package/'source/code.py')})
    ident={'source_manifest_sha256':m.sha(package/'RUNTIME-SOURCES.json')}
    config=dict(engineering_prefix=True,training_dispatch_allowed=False,start_update=0,stop_update=2,
        checkpoint_interval=1,global_batch=128,microbatch=8,physical_devices=[4,5,6,7],
        algorithm='dual_credit_rloo_clipped_v1',dual_arm='joint',source_root='/run/prefix/source',
        reference_cache='/run/prefix/reference-cache',dual_runtime_manifest='/run/prefix/RUNTIME-SOURCES.json',
        dual_runtime_identity=ident,pins={'/run/prefix/source/code.py':m.sha(package/'source/code.py')},
        source_policy='/inputs/initial/policy',encoded_audio_cache='/inputs/features/cache',audio_mirror_root='/inputs/audio')
    service=dict(pins=config['pins'],dual_runtime_identity=ident,policy=dict(max_batch_queries=4,
        max_batch_tokens=16384,physical_devices=[7],process_device_scope=dict(
            kind='same_container_runtime_visibility_v1',mounted_physical=[4,5,6,7],visible_ordinal='3')))
    m.write(package/'config.json',config);m.write(package/'service-config.json',service)
    m.write(package/'PREPARED.json',dict(status='integrated_prepared_only',remote_root='/run/prefix',
        image=m.IMAGE,dual_runtime_identity=ident,
        files={str(p.relative_to(package)):m.sha(p) for p in package.rglob('*') if p.is_file()}))
    lease=tmp_path/'lease';lease.touch()
    return dict(package=str(package),manifest_sha256=m.sha(package/'PREPARED.json'),
        host='npu',tag='dual-prefix-test',shared_root='/shared',driver_root='/driver',lease=str(lease),
        operations_root=str(tmp_path/'operations'),finite_training_dispatch_allowed=False)


def test_exact_mounts_and_no_donor_or_base_copy(tmp_path):
    c=fixture(tmp_path);receipt,config,_=m.validate(c);argv=m.command(c,receipt,config)
    assert '--volumes-from' not in argv and '--privileged' not in argv
    assert '/shared:/shared:ro' in argv
    assert '/inputs/initial:/inputs/initial:ro' in argv
    assert '/run/prefix/source:/run/prefix/source:ro' in argv
    assert '/inputs/features/cache:/inputs/features/cache' in argv
    devices=[argv[i+1] for i,v in enumerate(argv) if v=='--device']
    assert devices[:4]==[f'/dev/davinci{i}:/dev/davinci{i}' for i in range(4,8)]


@pytest.mark.parametrize('fault',['source','manifest','host','tag'])
def test_modified_or_invalid_package_rejected(tmp_path,fault):
    c=fixture(tmp_path)
    if fault=='source':(Path(c['package'])/'source/code.py').write_text('changed')
    if fault=='manifest':c['manifest_sha256']='0'*64
    if fault=='host':c['host']='-bad'
    if fault=='tag':c['tag']='../bad'
    with pytest.raises(ValueError):m.validate(c)


def test_outside_mount_dependency_rejected(tmp_path):
    c=fixture(tmp_path);receipt,config,_=m.validate(c);config['pins']['/unexpected/data']='x'
    with pytest.raises(ValueError,match='outside explicit mounts'):m.command(c,receipt,config)


def test_disabled_dispatch_never_calls_remote(tmp_path,monkeypatch):
    c=fixture(tmp_path)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**kw:pytest.fail('remote call'))
    with pytest.raises(ValueError,match='not enabled'):m.launch(c,{})


def test_existing_training_lease_prevents_any_remote_call(tmp_path,monkeypatch):
    c=fixture(tmp_path);c['finite_training_dispatch_allowed']=True
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**kw:pytest.fail('remote call'))
    with Path(c['lease']).open('r') as lease:
        m.fcntl.flock(lease,m.fcntl.LOCK_EX|m.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):m.launch(c,{})


def test_existing_attempt_never_recreated(tmp_path,monkeypatch):
    c=fixture(tmp_path);c['finite_training_dispatch_allowed']=True
    root=Path(c['operations_root']);root.mkdir();(root/(c['tag']+'-attempt.json')).write_text('{}')
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**kw:pytest.fail('remote call'))
    with pytest.raises(ValueError,match='existing attempt'):m.launch(c,{})


def test_ambiguous_start_retains_exact_handle_and_collects(tmp_path,monkeypatch):
    import shlex
    from types import SimpleNamespace as NS
    from tools.reports import context_dual_probe_operations as probe
    c=fixture(tmp_path);c['finite_training_dispatch_allowed']=True
    package=Path(c['package']);receipt,config,service=m.validate(c)
    config['source_policy_id']='policy'
    storage=package/'source/src/multimodal_jev/local_runtime_storage.py'
    storage.parent.mkdir(parents=True);storage.write_text('# fixture\n')
    monkeypatch.setattr(m,'validate',lambda _: (receipt,config,service))
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    monkeypatch.setattr(m,'inference_admission',lambda _:dict(policy_id='policy',image=m.IMAGE))
    cid='a'*64;calls=[]
    def run(argv,**kwargs):
        if argv[0]=='scp':return NS(returncode=0,stdout='',stderr='')
        command=shlex.split(argv[-1]);calls.append(command)
        if command[:3]==['docker','image','inspect']:out=json.dumps([{'Id':m.IMAGE}])
        elif command[:2]==['npu-smi','info']:out='\n'.join(f'No running processes found in NPU {i}' for i in range(4,8))
        elif command[:2]==['docker','create']:out=cid
        elif command[:2]==['docker','start']:
            handle=Path(c['operations_root'])/(c['tag']+'-launch.json')
            assert json.loads(handle.read_text())['container_id']==cid
            raise m.subprocess.TimeoutExpired(argv,55)
        elif command[:2]==['docker','inspect']:out=json.dumps([{'Id':cid}])
        else:out=''
        return NS(returncode=0,stdout=out,stderr='')
    monkeypatch.setattr(m.subprocess,'run',run)
    monkeypatch.setattr(m,'wait_exact',lambda remote,actual:0 if actual==cid else pytest.fail('wrong handle'))
    monkeypatch.setattr(m,'collect',lambda record,*a,**kw:dict(container_id=record['container_id']))
    monkeypatch.setattr(probe,'check_runtime',lambda r,h,o:None)
    result=m.launch(c,{})
    assert result['container_id']==cid and result['exit_code']==0
    assert sum(v[:2]==['docker','create'] for v in calls)==1
    assert sum(v[:2]==['docker','start'] for v in calls)==1
    assert json.loads((Path(c['operations_root'])/(c['tag']+'-start-response.json')).read_text())==dict(observation_timeout=True)
