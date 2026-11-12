import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import queue_intent_residency_replay as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path):
    root=tmp_path/'prior';root.mkdir();(root/'run').mkdir()
    atomic_json(root/'config.json',{})
    atomic_json(root/'launch.json',dict(pid=os.getpid(),argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    return dict(predecessor_controller=str(root),pins={str(root/n):file_hash(root/n) for n in ('config.json','launch.json')})


def test_ten_minute_wait_then_failed_predecessor_stops(tmp_path,monkeypatch):
    cfg=fixture(tmp_path);waits=[]
    def sleep(seconds):
        waits.append(seconds);atomic_json(Path(cfg['predecessor_controller'])/'run/terminal.json',dict(exit_code=1))
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(ValueError,match='predecessor failed'):module.await_predecessor(cfg,tmp_path)
    assert waits==[600]


def test_missing_predecessor_is_not_live_wait(tmp_path):
    cfg=fixture(tmp_path);p=Path(cfg['predecessor_controller'])/'launch.json'
    atomic_json(p,dict(pid=999999999,argv=['missing']));cfg['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError,match='missing without terminal'):module.await_predecessor(cfg,tmp_path)


def test_ordinary_acceptance_does_not_imply_residency_success(tmp_path,monkeypatch):
    training=tmp_path/'train';suite=tmp_path/'suite'
    for root in (training,suite):
        root.mkdir();atomic_json(root/'launch.json',dict(container_id=str(root),image='fixed'))
    monkeypatch.setattr(module,'completed',lambda *a:True)
    monkeypatch.setattr(module,'verify_final',lambda *a:dict(passed=True))
    atomic_json(suite/'residency-terminal.json',dict(exit_code=1))
    with pytest.raises(ValueError,match='comparison failed'):module.accepted(training,suite,'fixed')


def test_actual_container_completion_required_before_receipt(tmp_path,monkeypatch):
    training=tmp_path/'train';training.mkdir();atomic_json(training/'launch.json',dict(container_id='live',image='fixed'))
    monkeypatch.setattr(module,'completed',lambda *a:False)
    monkeypatch.setattr(module,'verify_final',lambda *a:pytest.fail('must observe actual exit first'))
    with pytest.raises(ValueError,match='must exit zero'):module.accepted(training,tmp_path/'suite','fixed')
