"""Concurrency and lifecycle boundary tests; no NPU/model substitutes as results."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

from multimodal_jev import local_checkpoint
from tools.reports import context_async_pipeline as module


@pytest.fixture(autouse=True)
def local_disk(monkeypatch):
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)


def test_ready_priority_exclusive_ownership_and_fail_closed(tmp_path):
    q=module.ReadyQueue(tmp_path/'events');entered=threading.Event();release=threading.Event();order=[]
    def job(priority,name,hold=False):
        with q.slot(priority,name):
            order.append(name)
            if hold:entered.set();assert release.wait(5)
    with ThreadPoolExecutor(3) as pool:
        first=pool.submit(job,10,'running',True);assert entered.wait(5)
        low=pool.submit(job,20,'training');high=pool.submit(job,0,'evaluation')
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            with q.condition:
                if len(q.pending)==2:break
            time.sleep(.001)
        else:pytest.fail('pending requests not registered')
        release.set();first.result();low.result();high.result()
    assert order==['running','evaluation','training']
    q.abort()
    with pytest.raises(RuntimeError):
        with q.slot(0,'late'):pytest.fail('dispatch after failure')


def test_cpu_collection_overlaps_next_npu_launch(tmp_path,monkeypatch):
    """Execute the generated lifecycle with fake remote calls, real lease/flock."""
    from tools.reports import context_replica_training_operations as shared
    from tools.reports import context_cached_storage_operations as storage
    from tools.reports import context_dual_probe_operations as runtime
    q=module.ReadyQueue(tmp_path/'events');lease=tmp_path/'lease';lease.touch()
    package=tmp_path/'package';(package/'source/src/multimodal_jev').mkdir(parents=True)
    (package/'source/src/multimodal_jev/local_runtime_storage.py').write_text('')
    blocked=threading.Event();release=threading.Event();second=threading.Event();validations=[]
    model={'training_devices':[4,5],'inference_services':[{'device':6},{'device':7}],
           'remote_root':'/data/fixture','layout':{'services':{}}}
    (package/'config.json').write_text('{}')
    def validate(c):
        # No NPU ownership during costly CPU admission, including its second pass.
        assert not q.busy
        validations.append(c['tag'])
        return model,{'pins':{}},{}
    def remote(argv,**kwargs):
        command=argv[-1]
        if argv[0]=='scp':return SimpleNamespace(returncode=0,stdout='',stderr='')
        if 'docker image inspect' in command:out='[{"Id":"'+shared.IMAGE+'"}]'
        elif 'npu-smi' in command:out='\n'.join(f'No running processes found in NPU {n}' for n in (4,5,6,7))
        elif command=='fake-create':out='a'*64
        elif 'docker inspect' in command:out='[{}]'
        else:out=''
        return SimpleNamespace(returncode=0,stdout=out,stderr='')
    def collect(record,path,**kwargs):
        assert not q.busy
        if 'first' in str(path):blocked.set();assert release.wait(5)
        else:second.set()
        return {}
    monkeypatch.setattr(shared.subprocess,'run',remote)
    monkeypatch.setattr(shared,'check_paths',lambda _:None)
    monkeypatch.setattr(shared,'wait_exact',lambda *args:0)
    monkeypatch.setattr(shared,'collect',collect)
    monkeypatch.setattr(runtime,'check_runtime',lambda *args:None)
    monkeypatch.setattr(storage,'command',lambda *args,**kwargs:['fake-create'])
    monkeypatch.setattr(storage,'storage_preflight',lambda *args,**kwargs:'pass')
    launch=module.launcher(SimpleNamespace(validate=validate),q,0,producer=True,authorization='selection_dispatch_allowed')
    def config(tag):return dict(package=str(package),operations_root=str(tmp_path/tag),lease=str(lease),
        tag=tag,manifest_sha256='fixture',host='fixture',selection_dispatch_allowed=True)
    with ThreadPoolExecutor(2) as pool:
        one=pool.submit(launch,config('first'),{});assert blocked.wait(5)
        two=pool.submit(launch,config('second'),{});assert second.wait(5)
        release.set();assert one.result()['exit_code']==two.result()['exit_code']==0
    assert validations==['first','first','second','second']
    with pytest.raises(ValueError):launch(dict(config('forbidden'),selection_dispatch_allowed=False),{})


def test_selection_point_failure_cancels_following_dispatch(tmp_path,monkeypatch):
    from tools.reports import prepare_context_replica_selection as preparation
    from tools.reports import collect_context_replica_infonce_curve as curve
    q=module.ReadyQueue(tmp_path/'events')
    c=dict(output=str(tmp_path/'out'),prepared_root=str(tmp_path/'prepared'),operations_root=str(tmp_path/'ops'),
        remote_root='/data/fixture',tag_prefix='fixture',deployment={},evaluation_workers=2)
    def prepare(spec,p):raise ValueError('changed source')
    monkeypatch.setattr(preparation,'prepare',prepare)
    monkeypatch.setattr(curve,'collect',lambda *args:pytest.fail('failed curve published'))
    with pytest.raises(ValueError,match='changed source'):
        module.selection_curve(c,{},[0,318],{},q)
    assert q.failed
    assert not any(module.read(p)['event']=='npu_acquired' for p in (tmp_path/'events').glob('*.json'))


def test_registered_curve_order_is_preserved_after_endpoint_first_execution(tmp_path,monkeypatch):
    from tools.reports import prepare_context_replica_selection as preparation
    from tools.reports import collect_context_replica_infonce_curve as curve
    q=module.ReadyQueue(tmp_path/'events');prepared=[];published=[]
    c=dict(output=str(tmp_path/'out'),prepared_root=str(tmp_path/'prepared'),operations_root=str(tmp_path/'ops'),
        remote_root='/data/fixture',tag_prefix='fixture',deployment={},evaluation_workers=2)
    def prepare(spec,p):prepared.append(spec['source_step']);return {'manifest_sha256':'fixture'}
    def run(point,ops,p,queue,priority,**kwargs):
        s=ops['selection_package']['source_step'];root=Path(ops['operations_root'])/'finalization';root.mkdir(parents=True)
        module.commit_json(root/'audit-config.json',{'fixture':s})
        return dict(status='complete_replica_infonce_selection_audited',source_step=s,runtime_provenance_verified=True)
    def collect(spec,p):published.extend(e['step'] for e in spec['evaluations']);return {'status':'fixture_complete'}
    monkeypatch.setattr(preparation,'prepare',prepare);monkeypatch.setattr(module,'operation',run)
    monkeypatch.setattr(curve,'collect',collect)
    module.selection_curve(c,{},[0,32,318],{},q)
    assert set(prepared[:2])=={318,0}
    assert published==[0,32,318]


def test_exact_process_rejects_reused_pid():
    import os
    proc=Path('/proc')/str(os.getpid())
    record=dict(pid=os.getpid(),process_start_ticks=proc.joinpath('stat').read_text().split(') ',1)[1].split()[19],
                argv=proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    assert module.exact_process(record)
    with pytest.raises(ValueError,match='identity'):
        module.exact_process(dict(record,process_start_ticks='0'))
