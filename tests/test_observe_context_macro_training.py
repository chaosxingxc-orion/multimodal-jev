import json
import os
import pytest
from tools.reports import observe_context_macro_training as m


def fixture(tmp_path,monkeypatch):
    p=tmp_path/'attempt.json';p.write_text('{}');os.utime(p,(1000,1000))
    monkeypatch.setattr(m,'validate',lambda c:dict(files_sha256='fixed'))
    return dict(operations_root=str(tmp_path),remote_root='/run',host='host')


def test_interval_and_failed_observation_consume_slot(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);calls=[]
    def remote(args):calls.append(args);raise TimeoutError('observation timeout')
    assert m.observe(c,now=1899,remote=remote)['status']=='not_due_no_remote_call'
    assert not calls
    assert m.observe(c,now=1900,remote=remote)['status']=='observation_failed_preserve_handles'
    assert len(calls)==1
    assert m.observe(c,now=2799,remote=remote)['status']=='not_due_no_remote_call'
    assert len(calls)==1
    m.observe(c,now=2800,remote=remote)
    assert len(calls)==2


def test_success_collects_only_declared_handles_and_small_progress(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch)
    (tmp_path/'client-launch.json').write_text('{"argv":["actual"],"container_id":"abc"}')
    monkeypatch.setattr(m,'command',lambda c,r:['actual'])
    inspected=[];monkeypatch.setattr(m,'inspect_runtime',lambda obs,h:inspected.append(h))
    def remote(args):
        if args[:2]==['docker','inspect']:return '[{"Id":"abc","State":{"Running":true}}]'
        return '{"ranks":[{"rank":0,"completed_updates":7},{"rank":1,"completed_updates":7}],"committed_checkpoints":7}'
    r=m.observe(c,now=1900,remote=remote)
    assert r['status']=='observed_exact_attempt' and len(inspected)==1
    assert r['progress']['committed_checkpoints']==7
    assert r['containers']['service-0']['status']=='not_created'
