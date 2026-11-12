import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import queue_intent_continuation as module
from multimodal_jev.baseline import atomic_json,file_hash


def fixture(tmp_path,monkeypatch):
    queue=tmp_path/'queue';suite=tmp_path/'suite';out=tmp_path/'out'
    for path in (queue/'run',suite,out):path.mkdir(parents=True)
    argv=(Path('/proc')/str(os.getpid())/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    atomic_json(queue/'launch.json',dict(pid=os.getpid(),argv=argv))
    cfg=dict(previous_queue=str(queue),previous_suite=str(suite),image='fixed',
        pins={str(queue/'launch.json'):file_hash(queue/'launch.json')})
    accepted={'passed':True,'verified_sha256':'checked'}
    monkeypatch.setattr(module,'verify_final',lambda root:accepted)
    monkeypatch.setattr(module,'completed',lambda cid,image:cid=='exact' and image=='fixed')
    launch=dict(root=str(suite),image='fixed',physical_devices=[4],container_id='exact')
    for path,value in ((suite/'launch.json',launch),(queue/'run/acceptance-dispatched.json',launch),
                       (queue/'run/accepted.json',accepted)):
        atomic_json(path,value)
    return cfg,out,queue,suite


def test_waits_live_exact_controller_then_accepts(tmp_path,monkeypatch):
    cfg,out,queue,suite=fixture(tmp_path,monkeypatch);calls=[]
    def finish(seconds):
        calls.append(seconds);atomic_json(queue/'run/terminal.json',{'exit_code':0})
    monkeypatch.setattr(module.time,'sleep',finish)
    module.await_predecessor(cfg,out)
    assert calls==[600] and (out/'predecessor-accepted.json').exists()


def test_follows_exact_manual_suite_without_restarting_failed_queue(tmp_path,monkeypatch):
    cfg,out,queue,suite=fixture(tmp_path,monkeypatch);calls=[]
    cfg.update(previous_container='exact',first_poll_after=123.)
    cfg['pins'][str(suite/'launch.json')]=file_hash(suite/'launch.json')
    atomic_json(queue/'run/terminal.json',{'exit_code':1})
    monkeypatch.setattr(module,'wait_exact',lambda *a:calls.append(a))
    module.await_predecessor(cfg,out)
    assert calls[0][0]=='exact' and calls[0][-1]==123.
    assert (out/'predecessor-accepted.json').exists()
    cfg['previous_container']='replacement'
    with pytest.raises(ValueError,match='exact independently'):module.await_predecessor(cfg,out)


@pytest.mark.parametrize('failure',['queue','dispatch','container','pid'])
def test_refuses_failed_or_replaced_predecessor(tmp_path,monkeypatch,failure):
    cfg,out,queue,suite=fixture(tmp_path,monkeypatch)
    if failure=='pid':
        atomic_json(queue/'launch.json',dict(pid=os.getpid(),argv=['different']))
        cfg['pins'][str(queue/'launch.json')]=file_hash(queue/'launch.json')
    else:
        atomic_json(queue/'run/terminal.json',{'exit_code':1 if failure=='queue' else 0})
    if failure=='dispatch':atomic_json(queue/'run/acceptance-dispatched.json',{'container_id':'replacement'})
    if failure=='container':monkeypatch.setattr(module,'completed',lambda *a:False)
    with pytest.raises(ValueError):module.await_predecessor(cfg,out)
    assert not (out/'predecessor-accepted.json').exists()
