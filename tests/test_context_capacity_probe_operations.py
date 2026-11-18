"""New capacity admission retains legacy finite probe's command/pin checks."""
import json
from pathlib import Path

import pytest

from tools.reports.context_capacity_probe_operations import namespace
from test_context_dual_probe_operations import fixture,seal,write


@pytest.mark.parametrize('batch,tokens',[(32,131072),(48,196608)])
def test_sealed_capacity_configuration_and_tamper_rejection(tmp_path,batch,tokens):
    c=fixture(tmp_path);c.update(batch_queries=batch,batch_tokens=tokens)
    p=Path(c['prepared'])/'package/config.json';config=json.loads(p.read_text())
    config['policy'].update(max_batch_queries=batch,max_batch_tokens=tokens)
    write(p,config);seal(c)
    ns=namespace(c);receipt,actual=ns['validate'](c)
    assert actual['policy']['max_batch_queries']==batch
    assert receipt['training_dispatch_allowed'] is False
    # Even a self-consistent resealed file cannot silently change declared capacity.
    config['policy']['max_batch_queries']=4;write(p,config);seal(c)
    with pytest.raises(ValueError):ns['validate'](c)


def test_capacity_release_cannot_skip_current_device_lease(tmp_path,monkeypatch):
    c=fixture(tmp_path);c.update(batch_queries=32,batch_tokens=131072,
        finite_inference_dispatch_allowed=True,priority_mode='idle_resource_window')
    p=Path(c['prepared'])/'package/config.json';config=json.loads(p.read_text())
    config['policy'].update(max_batch_queries=32,max_batch_tokens=131072)
    write(p,config);seal(c);Path(c['lease']).touch();ns=namespace(c)
    ns['check_paths']=lambda _:None
    monkeypatch.setattr(ns['subprocess'],'run',lambda *a,**kw:pytest.fail('unexpected remote access'))
    with Path(c['lease']).open('r') as lock:
        ns['fcntl'].flock(lock,ns['fcntl'].LOCK_EX|ns['fcntl'].LOCK_NB)
        with pytest.raises(BlockingIOError):ns['launch'](c,{})


def test_launch_creates_missing_parent_but_never_reuses_attempt_root(tmp_path,monkeypatch):
    import subprocess
    c=fixture(tmp_path);c.update(batch_queries=32,batch_tokens=131072,
        finite_inference_dispatch_allowed=True,priority_mode='idle_resource_window')
    p=Path(c['prepared'])/'package/config.json';config=json.loads(p.read_text())
    config['policy'].update(max_batch_queries=32,max_batch_tokens=131072)
    write(p,config);seal(c);Path(c['lease']).touch();ns=namespace(c)
    ns['check_paths']=lambda _:None;ns['check_device_observation']=lambda _:None
    commands=[]
    def remote(argv,**kw):
        import shlex
        command=shlex.split(argv[-1]);commands.append(command)
        if command[:3]==['docker','image','inspect']:
            return subprocess.CompletedProcess(argv,0,json.dumps([{'Id':ns['IMAGE']}]),'')
        if command==['mkdir',c['remote_root']]:
            return subprocess.CompletedProcess(argv,1,'','attempt root exists')
        return subprocess.CompletedProcess(argv,0,'','')
    monkeypatch.setattr(ns['subprocess'],'run',remote)
    ns['archive_package']=lambda prepared,path:path.write_bytes(b'archive')
    with pytest.raises(RuntimeError,match='attempt root exists'):ns['launch'](c,{})
    parent=['mkdir','-p',str(Path(c['remote_root']).parent)]
    assert commands[-2:]==[parent,['mkdir',c['remote_root']]]
    assert not any(command[:2]==['docker','create'] for command in commands)
