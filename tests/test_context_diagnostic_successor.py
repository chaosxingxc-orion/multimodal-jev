import json
from types import SimpleNamespace

import pytest

from tools.reports import run_context_diagnostic_successor as module
from multimodal_jev import local_checkpoint


@pytest.mark.parametrize('terminal',[{'exit_code':1},
    {'exit_code':0,'status':'collected_only'},
    {'exit_code':0,'status':'formal_infonce_audited_curve_bound','points':3,'evaluation_complete':False}])
def test_does_not_dispatch_on_exit0_or_incomplete_predecessor(tmp_path,monkeypatch,terminal):
    driver=tmp_path/'driver.json';driver.write_text('{}')
    state=tmp_path/'terminal.json';state.write_text(json.dumps(terminal))
    c=dict(output=str(tmp_path/'queue'),predecessor_driver=str(driver),predecessor_terminal=str(state))
    calls=[]
    monkeypatch.setattr(module,'validate',lambda c:(SimpleNamespace(launch=lambda *a:calls.append('launch')),{},{}))
    monkeypatch.setattr(module,'wait_exact',lambda *a:calls.append('wait'))
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda p:None)
    with pytest.raises(ValueError,match='successful full'):module.run(c,{})
    assert calls==['wait']
    assert json.loads((tmp_path/'queue/terminal.json').read_text())['restart'] is False


@pytest.mark.parametrize('kind',['language_stage','attention_stage'])
def test_successful_predecessor_dispatches_once_then_requires_audit(tmp_path,monkeypatch,kind):
    driver=tmp_path/'driver.json';driver.write_text('{}')
    state=tmp_path/'terminal.json';state.write_text(json.dumps(dict(exit_code=0,
        status='formal_infonce_audited_curve_bound',points=11,evaluation_complete=False)))
    c=dict(output=str(tmp_path/'queue'),predecessor_driver=str(driver),predecessor_terminal=str(state),diagnostic=kind)
    calls=[]
    def launch(*a):calls.append('launch');return dict(status=module.RESULTS[kind])
    monkeypatch.setattr(module,'validate',lambda c:(SimpleNamespace(launch=launch),{},{}))
    monkeypatch.setattr(module,'wait_exact',lambda *a:calls.append('wait'))
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda p:None)
    module.run(c,{})
    with pytest.raises(FileExistsError):module.run(c,{})
    assert calls==['wait','launch']
