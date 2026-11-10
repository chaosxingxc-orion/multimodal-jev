import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('layer_queue',Path(__file__).parents[1]/'tools/jev/watch_omni_npu_layer.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def setup_dependency(tmp_path,monkeypatch):
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    write(tmp_path/'queue-intent.json',{'order':['parent','p1']})
    write(tmp_path/'evaluation-launch.json',{'pid':1})
    monkeypatch.setattr(m,'FSC_QUEUE',m.file_hash(tmp_path/'queue-intent.json'))
    monkeypatch.setattr(m,'FSC_LAUNCH',m.file_hash(tmp_path/'evaluation-launch.json'))
    items={}
    for key,cid in zip(('parent','p1'),m.FSC_IDS):
        write(tmp_path/key/'launch.json',{'container_id':cid})
        write(tmp_path/key/'intent.json',{'argv':['docker','run',m.IMAGE,'command',key]})
        for n in ['run/terminal.json','evaluation-terminal.json']:write(tmp_path/key/n,{'exit_code':0})
        items[cid]=dict(Id=cid,Image=m.IMAGE,State=dict(Status='exited',ExitCode=0,OOMKilled=False),Config={'Cmd':['command',key]})
    monkeypatch.setattr(m,'inspect',lambda cid:items[cid])
    return items,write


def test_wait_requires_live_original_supervisor(tmp_path,monkeypatch):
    setup_dependency(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'live_process',lambda x:True)
    assert m.dependency_ready(tmp_path) is False
    monkeypatch.setattr(m,'live_process',lambda x:False)
    with pytest.raises(RuntimeError,match='missing'):m.dependency_ready(tmp_path)


@pytest.mark.parametrize('kind',['ok','queue_failed','running','oom','command','application_failed','identity'])
def test_completed_queue_checks_exact_containers_not_quality(tmp_path,monkeypatch,kind):
    items,write=setup_dependency(tmp_path,monkeypatch)
    write(tmp_path/'evaluation-terminal.json',{'exit_code':1 if kind=='queue_failed' else 0})
    item=items[m.FSC_IDS[1]]
    if kind=='running':item['State']['Status']='running'
    if kind=='oom':item['State']['OOMKilled']=True
    if kind=='command':item['Config']['Cmd']=['other']
    if kind=='application_failed':write(tmp_path/'p1/run/terminal.json',{'exit_code':1})
    if kind=='identity':item['Id']='f'*64
    if kind=='ok':assert m.dependency_ready(tmp_path)
    else:
        with pytest.raises((ValueError,RuntimeError)):m.dependency_ready(tmp_path)


def test_ambiguous_create_retains_intent_and_never_starts(tmp_path,monkeypatch):
    calls=[]
    def remote(argv,**kw):calls.append(argv);return SimpleNamespace(stdout='ambiguous',stderr='')
    monkeypatch.setattr(m,'remote',remote)
    with pytest.raises(RuntimeError,match='ambiguous'):m.dispatch(tmp_path,{'argv':['docker','create']})
    assert (tmp_path/'intent.json').exists() and len(calls)==1
    with pytest.raises(FileExistsError):m.execute(tmp_path,'unused')
    assert len(calls)==1


@pytest.mark.parametrize('kind',['docker_busy','driver_busy','free'])
def test_both_npu45_checked_before_dispatch(tmp_path,monkeypatch,kind):
    def remote(argv,**kw):
        if argv[:2]==['docker','ps']:return SimpleNamespace(stdout='a'*64 if kind=='docker_busy' else '')
        assert argv[0]=='npu-smi' and argv[5] in ('4','5')
        return SimpleNamespace(stdout='PID 12' if kind=='driver_busy' and argv[5]=='5' else '')
    monkeypatch.setattr(m,'remote',remote)
    monkeypatch.setattr(m,'inspect',lambda cid:{'HostConfig':{'Privileged':False,'Devices':[{'PathOnHost':'/dev/davinci5'}]}})
    assert m.free_devices(tmp_path) is (kind=='free')
