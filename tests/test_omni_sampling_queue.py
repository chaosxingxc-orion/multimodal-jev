from pathlib import Path
from types import SimpleNamespace as NS
import json

import pytest

from test_sfc_jev import load_tool


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'tools/jev'))
    return load_tool('watch_omni_sampling')


@pytest.mark.parametrize('occupied',[False,True,'privileged'])
def test_device_check(monkeypatch,occupied):
    m=module(monkeypatch)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'cid\n')
    monkeypatch.setattr(m,'inspect',lambda cid:dict(HostConfig=dict(Privileged=occupied=='privileged',
        Devices=[{'PathOnHost':'/dev/davinci6' if occupied is True else '/dev/davinci0'}])))
    if occupied:
        with pytest.raises(RuntimeError):m.require_free_devices()
    else:m.require_free_devices()


def test_dispatch_saves_intent_before_call_and_never_replays(tmp_path,monkeypatch):
    m=module(monkeypatch);calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        assert (tmp_path/'sampling-intent.json').exists()
        return NS(returncode=0,stdout='a'*64+'\n',stderr='')
    monkeypatch.setattr(m.subprocess,'run',run)
    monkeypatch.setattr(m,'wait',lambda *a:dict(Status='exited',ExitCode=0))
    assert m.dispatch(tmp_path,'sampling',['docker','run'], 'image','commit',60,[6,7])=='a'*64
    assert json.loads((tmp_path/'launch.json').read_text())['physical_devices']==[6,7]
    before=len(calls)
    with pytest.raises(FileExistsError):m.dispatch(tmp_path,'sampling',[], 'image','commit',60,[6,7])
    assert len(calls)==before


@pytest.mark.parametrize('mode',['ambiguous','timeout','failed_container'])
def test_failed_dispatch_never_replayed(tmp_path,monkeypatch,mode):
    m=module(monkeypatch)
    def run(*a,**k):
        if mode=='timeout':raise m.subprocess.TimeoutExpired('ssh',90)
        return NS(returncode=0,stdout='unknown' if mode=='ambiguous' else 'a'*64,stderr='')
    monkeypatch.setattr(m.subprocess,'run',run)
    monkeypatch.setattr(m,'wait',lambda *a:dict(Status='exited',ExitCode=1))
    with pytest.raises((RuntimeError,m.subprocess.TimeoutExpired)):
        m.dispatch(tmp_path,'sampling',[],'image','commit',60,[6,7])
    assert (tmp_path/'sampling-intent.json').exists()
    with pytest.raises(FileExistsError):m.dispatch(tmp_path,'sampling',[],'image','commit',60,[6,7])
