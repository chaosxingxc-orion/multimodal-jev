from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import run_intent_matched_pilots as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path,terminal=None):
    root=tmp_path/'D';root.mkdir();(root/'run').mkdir()
    atomic_json(root/'launch.json',dict(pid=123,argv=['exact']))
    if terminal is not None:atomic_json(root/'run/terminal.json',dict(exit_code=terminal))
    return dict(predecessor_controller=str(root),pins={str(root/'launch.json'):file_hash(root/'launch.json')})


def test_failed_d_never_dispatches(tmp_path):
    cfg=fixture(tmp_path,1)
    with pytest.raises(ValueError,match='predecessor failed'):module.await_d12(cfg,tmp_path)


def test_missing_live_handle_is_not_wait(tmp_path):
    cfg=fixture(tmp_path)
    cfgpath=Path(cfg['predecessor_controller'])/'launch.json';atomic_json(cfgpath,dict(pid=999999999,argv=['exact']))
    cfg['pins'][str(cfgpath)]=file_hash(cfgpath)
    with pytest.raises(ValueError,match='missing without terminal'):module.await_d12(cfg,tmp_path)


def test_wait_poll_interval_and_exact_handle(tmp_path,monkeypatch):
    import os
    cfg=fixture(tmp_path);p=Path(cfg['predecessor_controller'])/'launch.json'
    atomic_json(p,dict(pid=os.getpid(),argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    cfg['pins'][str(p)]=file_hash(p);waits=[]
    def sleep(seconds):
        waits.append(seconds)
        atomic_json(Path(cfg['predecessor_controller'])/'run/terminal.json',dict(exit_code=1))
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(ValueError,match='predecessor failed'):module.await_d12(cfg,tmp_path)
    assert waits==[600]


def test_success_flag_cannot_hide_incomplete_budget(tmp_path):
    cfg=fixture(tmp_path,0);root=Path(cfg['predecessor_controller'])
    atomic_json(root/'run/progress.json',dict(completed_updates=11,total_updates=12,accepted_suite='/wrong'))
    with pytest.raises(ValueError,match='complete D12'):module.await_d12(cfg,tmp_path)
