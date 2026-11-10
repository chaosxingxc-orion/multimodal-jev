import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import watch_audio2tool_jev_capacity as queue


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def pin_native(monkeypatch,root):
    original=queue.file_hash
    monkeypatch.setattr(queue,'file_hash',lambda p:queue.NATIVE_PLAN_SHA if p==root/'queue-plan.json' else
        queue.NATIVE_LAUNCH_SHA if p==root/'evaluation-launch.json' else original(p))


def test_pending_dependency_requires_actual_process_identity(tmp_path,monkeypatch):
    stat=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    save(tmp_path/'evaluation-launch.json',dict(pid=os.getpid(),process_start=stat[19]))
    pin_native(monkeypatch,tmp_path)
    monkeypatch.setattr(queue.native,'inspect',lambda *a:pytest.fail('pending dependency cannot dispatch'))
    assert queue.native_ready(tmp_path) is False
    save(tmp_path/'evaluation-launch.json',dict(pid=os.getpid(),process_start='replaced'))
    with pytest.raises(RuntimeError,match='missing/replaced/dead'):queue.native_ready(tmp_path)
    save(tmp_path/'evaluation-terminal.json',dict(exit_code=1))
    with pytest.raises(RuntimeError,match='dependency failed'):queue.native_ready(tmp_path)


@pytest.mark.parametrize('fault',[None,'acceptance','sampling','container'])
def test_success_requires_fresh_native_acceptance(tmp_path,monkeypatch,fault):
    # This is an envelope fixture. Actual native NPU execution is still pending.
    pin_native(monkeypatch,tmp_path)
    save(tmp_path/'evaluation-launch.json',{})
    save(tmp_path/'evaluation-terminal.json',dict(exit_code=0))
    save(tmp_path/'launch.json',dict(container_id='a'*64))
    receipt=dict(container_id='a'*64,result_sha256='r',preparation_sha256='p')
    save(tmp_path/'acceptance.json',{**receipt,'result_sha256':'wrong' if fault=='acceptance' else 'r'})
    monkeypatch.setattr(queue.native,'check_pins',lambda *a:({},dict(dependency_root='sampling')))
    monkeypatch.setattr(queue.native,'sampling_ready',lambda *a:fault!='sampling')
    observed={'Id':'b'*64 if fault=='container' else 'a'*64}
    monkeypatch.setattr(queue.native,'inspect',lambda *a:observed)
    def require(root,inspection,pin,cid,processor):
        assert json.loads(inspection.read_text())==observed and queue.file_hash(inspection)==pin
        if observed['Id']!=cid:raise ValueError('unexpected actual container')
        return receipt
    monkeypatch.setattr(queue.native,'require_capacity',require)
    if fault:
        with pytest.raises(ValueError):queue.native_ready(tmp_path)
    else:assert queue.native_ready(tmp_path) is True


@pytest.mark.parametrize('fault',[None,'create_timeout','start_timeout','created_identity'])
def test_dispatch_records_identity_before_start_and_never_replays(tmp_path,monkeypatch,fault):
    cid='a'*64;prep=dict(argv=['docker','create','fixture'],image='image',source_commit='commit');calls=[]
    def remote(argv,**kwargs):
        calls.append(argv)
        assert json.loads((tmp_path/'intent.json').read_text())==prep
        if argv[1]=='create':
            if fault=='create_timeout':raise subprocess.TimeoutExpired(argv,90)
        else:
            receipt=json.loads((tmp_path/'launch.json').read_text())
            assert receipt['container_id']==cid and receipt['intent_sha256']==queue.file_hash(tmp_path/'intent.json')
            assert (tmp_path/'start-intent.json').exists()
            if fault=='start_timeout':raise subprocess.TimeoutExpired(argv,90)
        return SimpleNamespace(stdout=cid+'\n',stderr='')
    monkeypatch.setattr(queue.native,'remote',remote)
    monkeypatch.setattr(queue.native,'inspect',lambda _:dict(Id='b'*64 if fault=='created_identity' else cid,
                                                            Image='image',State={'Status':'created'}))
    if fault:
        with pytest.raises((subprocess.TimeoutExpired,ValueError)):queue.dispatch(tmp_path,prep)
    else:assert queue.dispatch(tmp_path,prep)==(cid,queue.file_hash(tmp_path/'intent.json'))
    count=len(calls)
    with pytest.raises(FileExistsError):queue.dispatch(tmp_path,prep)
    assert len(calls)==count


def test_observation_retry_stays_on_same_id(tmp_path,monkeypatch):
    cid='a'*64;prep={'image':'image'};calls=[]
    plan=dict(dependency_seconds=350000,observation_seconds=3900,dependency_root='dependency',resource_lock=str(tmp_path/'device.lock'))
    monkeypatch.setattr(queue,'check_pins',lambda *a:(plan,prep))
    monkeypatch.setattr(queue,'native_ready',lambda *a:True)
    monkeypatch.setattr(queue,'free_devices',lambda *a:True)
    def dispatch(*a):calls.append('dispatch');return cid,'intent'
    monkeypatch.setattr(queue,'dispatch',dispatch)
    observations=[]
    def inspect(current):
        observations.append(current)
        if len(observations)==1:raise subprocess.TimeoutExpired('inspect',30)
        return dict(Id=cid,Image='image',State=dict(Status='exited',ExitCode=0))
    monkeypatch.setattr(queue.native,'inspect',inspect)
    monkeypatch.setattr(queue.native,'successful',lambda *a:None)
    monkeypatch.setattr(queue.time,'sleep',lambda *a:None)
    registry=tmp_path/'registry';registry.write_bytes(b'fixture');monkeypatch.setattr(queue,'REGISTRY',str(registry))
    def accept(root,pin,inspection,inspection_hash,current,*args):
        assert current==cid and inspection_hash==queue.file_hash(inspection)
        assert json.loads(inspection.read_text())['State']['Status']=='exited'
        return {'container_id':current}
    monkeypatch.setattr(queue,'require_capacity',accept)
    queue.execute(tmp_path,'plan')
    assert calls==['dispatch'] and observations==[cid,cid]
    assert json.loads((tmp_path/'acceptance.json').read_text())=={'container_id':cid}


def test_previous_intent_blocks_execution_before_dependency(tmp_path,monkeypatch):
    save(tmp_path/'intent.json',{})
    monkeypatch.setattr(queue,'check_pins',lambda *a:pytest.fail('previous attempt must stop immediately'))
    with pytest.raises(FileExistsError):queue.execute(tmp_path,'plan')


@pytest.mark.parametrize('output,free',[('',True),('Process id:123 Process name:python3',False),('unrecognized output',False)])
def test_driver_processes_outside_docker_block_dispatch(tmp_path,monkeypatch,output,free):
    monkeypatch.setattr(queue.native,'free_devices',lambda:True)
    calls=[]
    def remote(argv):calls.append(argv);return SimpleNamespace(stdout=output)
    monkeypatch.setattr(queue.native,'remote',remote)
    assert queue.free_devices(tmp_path) is free
    assert [c[-3] for c in calls]==['6','7']
    assert (tmp_path/'device-process-observation.json').exists()
