import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools/jev'))
import watch_mmar_capacity as queue


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('fault', [None, 'pid', 'zombie', 'argv', 'missing', 'failed'])
def test_live_predecessor_requires_process_identity(tmp_path, monkeypatch, fault):
    root = tmp_path/'audio2tool-decode-profile-v1'
    save(root/'evaluation-launch.json', dict(pid=123, process_start='456', argv=['python', 'queue.py']))
    proc = tmp_path/'proc';proc.mkdir()
    fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
    fields[19] = 'other' if fault == 'pid' else '456'
    if fault == 'zombie':
        fields[0] = 'Z'
    if fault != 'missing':
        (proc/'stat').write_text('123 (worker) '+' '.join(fields))
        (proc/'cmdline').write_bytes(b'other\0' if fault == 'argv' else b'python\0queue.py\0')
    if fault == 'failed':
        save(root/'evaluation-terminal.json', dict(exit_code=1))
    monkeypatch.setattr(queue, 'Path', lambda p: proc if str(p) == '/proc/123' else Path(p))
    monkeypatch.setattr(queue, 'file_hash', lambda p: queue.DECODE_PLAN if p.name == 'queue-plan.json' else queue.DECODE_LAUNCH)
    if fault:
        with pytest.raises(RuntimeError):
            queue.dependency_ready(tmp_path)
    else:
        assert queue.dependency_ready(tmp_path) is False


@pytest.mark.parametrize('fault', [None, 'container', 'result', 'capacity'])
def test_completed_decode_requires_fresh_container_and_results(tmp_path, monkeypatch, fault):
    root = tmp_path/'audio2tool-decode-profile-v1';cid='a'*64
    save(root/'evaluation-launch.json', {})
    save(root/'evaluation-terminal.json', dict(exit_code=0))
    save(root/'launch.json', dict(container_id=cid));save(root/'intent.json', {})
    save(root/'run/terminal.json', dict(exit_code=0))
    save(root/'run/result.json', dict(status='profile_complete', model_loaded=True, records=[{}]*16, weight_updates=0, targets_read=False))
    save(root/'execution-complete.json', dict(container_id=cid, result_sha256='wrong' if fault=='result' else 'result'))
    monkeypatch.setattr(queue, 'file_hash', lambda p: queue.DECODE_PLAN if p.name=='queue-plan.json' else queue.DECODE_LAUNCH if p.name=='evaluation-launch.json' else 'result')
    monkeypatch.setattr(queue.decode, 'check_pins', lambda *a: None)
    monkeypatch.setattr(queue.decode, 'dependency_ready', lambda *a: fault!='capacity')
    seen=[]
    def inspect(cid):
        seen.append(cid)
        return dict(Id='b'*64 if fault=='container' else cid)
    monkeypatch.setattr(queue.native, 'inspect', inspect)
    def check(item, plan, cid):
        if item['Id']!=cid:
            raise ValueError('wrong container')
    monkeypatch.setattr(queue, 'check_container', check)
    if fault:
        with pytest.raises(ValueError):
            queue.dependency_ready(tmp_path)
    else:
        assert queue.dependency_ready(tmp_path) and seen==[cid]


def test_capacity_observation_retries_same_id_and_admits_actual_inspection(tmp_path, monkeypatch):
    cid='a'*64;calls=[];seen=[]
    plan=dict(dependency_seconds=350000,observation_seconds=3900,resource_lock=str(tmp_path/'device.lock'))
    monkeypatch.setattr(queue, 'check_pins', lambda *a: (plan,dict(image='image')))
    monkeypatch.setattr(queue, 'dependency_ready', lambda *a: True)
    monkeypatch.setattr(queue.jev, 'free_devices', lambda *a: True)
    def dispatch(root,prep):
        calls.append(cid);save(root/'intent.json', {});save(root/'launch.json', dict(container_id=cid))
        return cid,'intent'
    monkeypatch.setattr(queue.jev, 'dispatch', dispatch)
    def inspect(current):
        seen.append(current)
        if len(seen)==1:
            raise subprocess.TimeoutExpired('inspect',30)
        return dict(Id=cid,Image='image',State=dict(Status='exited',ExitCode=0))
    monkeypatch.setattr(queue.native, 'inspect', inspect)
    monkeypatch.setattr(queue.time, 'sleep', lambda _: None)
    monkeypatch.setattr(queue, 'population', lambda *a: (['rows'],['prior'],{}))
    def admit(root,pin,path,digest,rows,prior):
        assert json.loads(path.read_text())['Id']==cid and queue.file_hash(path)==digest
        assert rows==['rows'] and prior==['prior']
        return dict(container_id=cid)
    monkeypatch.setattr(queue, 'admit_capacity', admit)
    queue.execute(tmp_path,'pin')
    assert calls==[cid] and seen==[cid,cid]
    assert json.loads((tmp_path/'acceptance.json').read_text())==dict(container_id=cid)
    with pytest.raises(FileExistsError):
        queue.execute(tmp_path,'pin')
    assert calls==[cid]
