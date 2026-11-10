from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import watch_omni_verl_acceptance as watcher


@pytest.mark.parametrize('scenario',['success','transient','failed','changed','timeout','existing','invalid'])
def test_observer_never_replays_inference(tmp_path,monkeypatch,scenario):
    plan=dict(run=str(tmp_path/'run'),container_id='a'*64,image='image',timeout_seconds=15000,launch_sha256='pin')
    monkeypatch.setattr(watcher,'validate',lambda *args:plan)
    state=dict(Id='a'*64,Image='image',State=dict(Status='exited',ExitCode=0))
    calls=[];checks=[]
    if scenario=='existing':(tmp_path/'verification-intent.json').write_text('{}')
    if scenario=='failed':state['State']['ExitCode']=1
    def inspect(cid):
        calls.append(cid)
        if scenario=='transient' and len(calls)==1:raise subprocess.TimeoutExpired('ssh',30)
        result=deepcopy(state)
        if scenario=='changed' and len(calls)>1:result['State']['ExitCode']=1
        if scenario=='timeout':result['State']['Status']='running'
        return result
    monkeypatch.setattr(watcher,'inspect',inspect)
    def accept(*args):
        checks.append(args)
        if scenario=='invalid':raise ValueError('invalid comparison')
        return {'accepted':True}
    monkeypatch.setattr(watcher,'accept',accept)
    monkeypatch.setattr(watcher.time,'sleep',lambda _:None)
    times=iter([0,15001])
    if scenario=='timeout':monkeypatch.setattr(watcher.time,'monotonic',lambda:next(times))
    if scenario in ('success','transient'):
        watcher.execute(tmp_path,'pin')
        assert json.loads((tmp_path/'result.json').read_text())['inference_replayed'] is False
        assert len(checks)==1 and len(calls)==(3 if scenario=='transient' else 2)
    else:
        with pytest.raises((ValueError,TimeoutError)):watcher.execute(tmp_path,'pin')
        assert not (tmp_path/'acceptance.json').exists()
        if scenario in ('failed','timeout','existing'):assert not checks
        if scenario=='existing':assert not calls
    assert all(cid=='a'*64 for cid in calls)
