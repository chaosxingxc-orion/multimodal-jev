import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.reports import run_multidevice_evaluation as module
from test_prepare_context_replica_selection import write


@pytest.mark.parametrize('fault',[None,'prepare','predecessor','run'])
def test_prepare_validate_wait_run_order_and_fail_closed(tmp_path,monkeypatch,fault):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'validate',lambda _:None)
    events=[]
    class Engine:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*args):events.append('close')
    monkeypatch.setattr(module,'IsolatedSourceAuditEngine',Engine)
    def prepare(*args):
        events.append('prepare')
        if fault=='prepare':raise ValueError('prepare failed')
        return dict(manifest_sha256='digest')
    def audit(*args):events.append('audit');return {}
    def validate(*args):events.append('validate');return {},{},{}
    def run(*args):
        events.append('run')
        if fault=='run':raise ValueError('run failed')
        return dict(status='complete')
    monkeypatch.setattr(module,'scoped',lambda e:(SimpleNamespace(prepare=prepare,audit=audit),SimpleNamespace(validate=validate,run=run)))
    monkeypatch.setattr(module.operations,'command',lambda *a:[])
    monkeypatch.setattr(module,'read_bound',lambda _:b'{}')
    monkeypatch.setattr(module,'wait_exact',lambda *a:events.append('wait'))
    terminal=tmp_path/'prior.json';write(terminal,dict(exit_code=int(fault=='predecessor'),status='accepted'))
    c=dict(output=str(tmp_path/'out'),package={'output':str(tmp_path/'package')},deployment={},cpu_workers=4,io_workers=4,
        predecessor=dict(driver={},terminal=str(terminal),required_status='accepted'))
    if fault:
        with pytest.raises(ValueError):module.run(c,{})
    else:module.run(c,{})
    expected=['prepare'] if fault=='prepare' else ['prepare','audit','validate','wait']+([] if fault=='predecessor' else ['run'])
    assert events==expected+['close']
    assert json.loads((Path(c['output'])/'terminal.json').read_text())['exit_code']==int(fault is not None)


def test_scoped_lifecycle_does_not_mutate_modules(monkeypatch):
    before=module.operations.launch
    engine=SimpleNamespace(preparation=lambda:SimpleNamespace(audit=lambda *a:None))
    _,api=module.scoped(engine)
    assert module.operations.launch is before
    with pytest.raises(ValueError,match='authorization'):api.run({'operations_root':'/tmp/unused','selection_dispatch_allowed':False},{})
