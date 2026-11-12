import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import run_intent_momentum_first as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path):
    root=tmp_path/'predecessor';root.mkdir();(root/'run').mkdir()
    atomic_json(root/'config.json',dict(image='fixed',protocol='protocol',protocol_sha256='hash'))
    atomic_json(root/'launch.json',dict(pid=os.getpid(),argv=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')))
    return dict(predecessor_controller=str(root),image='fixed',pins={str(root/n):file_hash(root/n) for n in ('launch.json','config.json')})


def test_wait_600_then_fail_without_dispatch(tmp_path,monkeypatch):
    c=fixture(tmp_path);waits=[]
    def sleep(seconds):
        waits.append(seconds);atomic_json(Path(c['predecessor_controller'])/'run/terminal.json',dict(exit_code=1))
    monkeypatch.setattr(module.time,'sleep',sleep)
    with pytest.raises(ValueError,match='predecessor failed'):module.await_binding(c,tmp_path)
    assert waits==[600]


def test_missing_handle_not_restarted(tmp_path):
    c=fixture(tmp_path);p=Path(c['predecessor_controller'])/'launch.json'
    atomic_json(p,dict(pid=999999999,argv=['missing']));c['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError,match='missing without terminal'):module.await_binding(c,tmp_path)


def test_success_reauthenticates_actual_gates(tmp_path,monkeypatch):
    c=fixture(tmp_path);root=Path(c['predecessor_controller']);calls=[]
    result=dict(protocol_sha256='hash',reports={'step0':{},'step4':{}})
    atomic_json(root/'run/terminal.json',dict(exit_code=0));atomic_json(root/'run/comparison.json',result)
    def score(path,digest):calls.append((path,digest));return result
    monkeypatch.setattr(module,'score_binding',score)
    module.await_binding(c,tmp_path)
    assert calls==[('protocol','hash')]
    atomic_json(root/'run/comparison.json',dict(result,protocol_sha256='different'))
    with pytest.raises(ValueError,match='comparison differs'):module.await_binding(c,tmp_path)


def test_backend_failure_blocks_even_success_terminal(tmp_path,monkeypatch):
    c=fixture(tmp_path);root=Path(c['predecessor_controller'])
    atomic_json(root/'run/terminal.json',dict(exit_code=0))
    def score(*args):raise ValueError('actual Docker failed')
    monkeypatch.setattr(module,'score_binding',score)
    with pytest.raises(ValueError,match='actual Docker'):module.await_binding(c,tmp_path)
