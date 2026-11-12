import os
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import queue_intent_zero_delta_gate as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path):
    root=tmp_path/'prior';root.mkdir();(root/'run').mkdir()
    atomic_json(root/'config.json',{'plan_root':str(tmp_path/'plan')})
    atomic_json(root/'launch.json',dict(pid=os.getpid(),argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    return dict(predecessor_controller=str(root),image='fixed',pins={str(root/n):file_hash(root/n) for n in ('config.json','launch.json')})


def test_ten_minute_wait_then_failure_stops(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);waits=[]
    def sleep(seconds):
        waits.append(seconds);atomic_json(Path(cfg['predecessor_controller'])/'run/terminal.json',dict(exit_code=1))
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(ValueError,match='predecessor failed'):module.await_predecessor(cfg,tmp_path)
    assert waits==[600]


def test_missing_pid_not_live_wait(tmp_path):
    c=fixture(tmp_path);p=Path(c['predecessor_controller'])/'launch.json'
    atomic_json(p,dict(pid=999999999,argv=['missing']));c['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError,match='missing without terminal'):module.await_predecessor(c,tmp_path)


def test_success_file_requires_actual_full_replay_acceptance(tmp_path,monkeypatch):
    c=fixture(tmp_path);atomic_json(Path(c['predecessor_controller'])/'run/terminal.json',dict(exit_code=0));called=[]
    def check(training,suite,image):
        called.append((training,suite,image));raise ValueError('actual replay not accepted')
    monkeypatch.setattr(module,'accepted',check)
    with pytest.raises(ValueError,match='actual replay'):module.await_predecessor(c,tmp_path)
    assert len(called)==1 and not (tmp_path/'predecessor-accepted.json').exists()
