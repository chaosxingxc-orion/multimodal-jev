import json
from pathlib import Path
import pytest
from tools.reports import objective_acceptance_successor as module


@pytest.mark.parametrize('fault',[None,'predecessor','wait','changed','sequence'])
def test_wait_before_dispatch_and_fail_closed(tmp_path,monkeypatch,fault):
    import multimodal_jev.local_checkpoint as checkpoint
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda _:None)
    c=dict(output=str(tmp_path/'successor'),predecessor_terminal=str(tmp_path/'prior.json'))
    calls=[];checks=[]
    def validate(c):
        checks.append('check')
        if fault=='changed' and len(checks)==2:raise ValueError('changed')
        return {'pid':123},{'sequence':'fixture'},{}
    def wait(driver,terminal):
        calls.append('wait');assert driver['pid']==123
        if fault=='wait':raise ValueError('missing exact process')
        Path(terminal).write_text(json.dumps(dict(exit_code=int(fault=='predecessor'),status='async_simple_pipeline_completed')))
    def run(c,p):
        calls.append('dispatch');assert len(checks)==2
        if fault=='sequence':raise ValueError('training failed')
        return dict(exit_code=0,status='objective_prefix_and_recovery_audited')
    monkeypatch.setattr(module,'validate',validate);monkeypatch.setattr(module,'wait_exact',wait)
    monkeypatch.setattr(module.sequence,'run',run)
    if fault:
        with pytest.raises(ValueError):module.run(c,{})
    else:assert module.run(c,{})['exit_code']==0
    assert calls==(['wait','dispatch'] if fault in (None,'sequence') else ['wait'])
    t=json.loads((Path(c['output'])/'terminal.json').read_text())
    assert t['exit_code']==int(fault is not None)
    if fault:assert t['restart'] is False


def test_validate_binds_existing_queue_and_disjoint_paths(tmp_path,monkeypatch):
    import hashlib
    def bound(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    prior=tmp_path/'prior'
    cfg=dict(prefix=dict(lease='same',host='host',operations_root=str(tmp_path/'ops'),package=str(tmp_path/'pkg')),
        recovery=dict(operations_root=str(tmp_path/'recovery-ops'),package=str(tmp_path/'recovery-pkg')),
        output=str(tmp_path/'sequence'))
    c=dict(output=str(tmp_path/'successor'),pins=[],predecessor_terminal=str(prior/'terminal.json'),
        predecessor_config=bound('prior/config.json',dict(output=str(prior),deployment={'lease':'same','host':'host'})),
        predecessor_driver=bound('prior-driver/started.json',dict(argv=['python','-m','tools.reports.context_stage1_priority_pipeline'])),
        sequence_template=bound('template.json',{}),sequence_local=bound('local.json',{}))
    monkeypatch.setattr(module,'load_config',lambda *a:(cfg,{}))
    monkeypatch.setattr(module.sequence,'validate',lambda _:None)
    assert module.validate(c)[1] is cfg
    cfg['prefix']['lease']='different'
    with pytest.raises(ValueError,match='lease'):module.validate(c)
