import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import run_intent_lr_control as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path):
    root=tmp_path/'controller';root.mkdir();(root/'run').mkdir()
    atomic_json(root/'config.json',{})
    atomic_json(root/'launch.json',dict(pid=os.getpid(),argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    return dict(predecessor_controller=str(root),predecessor_suite='suite',pins={str(root/n):file_hash(root/n) for n in ('launch.json','config.json')})


def test_waits_ten_minutes_and_stops_on_failure(tmp_path,monkeypatch):
    c=fixture(tmp_path);waits=[]
    def sleep(seconds):
        waits.append(seconds);atomic_json(Path(c['predecessor_controller'])/'run/terminal.json',dict(exit_code=1))
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(ValueError,match='predecessor failed'):module.await_qp(c,tmp_path)
    assert waits==[600]


@pytest.mark.parametrize('bad',['arm','updates','suite'])
def test_success_terminal_requires_actual_complete_budget(tmp_path,bad):
    c=fixture(tmp_path);root=Path(c['predecessor_controller'])
    atomic_json(root/'run/terminal.json',dict(exit_code=0))
    atomic_json(root/'run/progress.json',dict(arm='Q' if bad=='arm' else 'P',completed_updates=11 if bad=='updates' else 12,
        total_updates=12,accepted_suite='wrong' if bad=='suite' else 'suite'))
    with pytest.raises(ValueError,match='complete Q/P'):module.await_qp(c,tmp_path)


def test_missing_process_does_not_trigger_replacement(tmp_path):
    c=fixture(tmp_path);p=Path(c['predecessor_controller'])/'launch.json'
    atomic_json(p,dict(pid=999999999,argv=['missing']));c['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError,match='missing without terminal'):module.await_qp(c,tmp_path)
