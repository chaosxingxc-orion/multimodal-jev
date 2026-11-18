"""Candidate recovery preserves fixed Stage1 and audited selection history."""
import copy,json
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev import batched_stage2_resume as module


def admission(tmp_path, monkeypatch):
    import multimodal_jev.batched_stage2_initialization as initial
    import multimodal_jev.batched_continuous_resume as resume
    path = tmp_path / 'run/evaluation/selection-0000/result.json'
    path.parent.mkdir(parents=True); path.write_text('{}')
    digest = file_hash(path)
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(dict(selection=[dict(passed=True, updates=0, policy_id='base:stage1',result=str(path),
                                                      result_sha256=digest)])))
    recovered = dict(reference_policy_id='base:stage1',completed_updates=3,files={str(path):digest})
    calls=[]
    def initialized(config,*,verify_parity): calls.append('initial'); return dict(reference_policy_id='base:stage1',files={})
    def resumed(config):
        calls.append('resume'); return copy.deepcopy(recovered)
    monkeypatch.setattr(initial,'authenticate',initialized)
    monkeypatch.setattr(resume,'authenticate',resumed)
    config=dict(continuous_resume=dict(training_root=str(tmp_path),audit=str(audit)))
    return config,recovered,path,audit,calls


def test_release_initial_reference_and_full_recovery_all_required(tmp_path,monkeypatch):
    config,_,path,_,calls=admission(tmp_path,monkeypatch)
    result=module.authenticate(config,verify_parity=None)
    assert calls==['initial','resume']
    assert result['selection_history']==[dict(updates=0,policy_id='base:stage1',
        result=str(path),result_sha256=file_hash(path))]


@pytest.mark.parametrize('fault',['reference','unbound_selection','changed_selection','audit_hash','unaccepted'])
def test_join_refuses_wrong_reference_or_unverified_selection_history(tmp_path,monkeypatch,fault):
    config,recovered,path,audit,_=admission(tmp_path,monkeypatch)
    if fault=='reference': recovered['reference_policy_id']='base:rl'
    elif fault=='unbound_selection': recovered['files'].clear()
    elif fault=='changed_selection':path.write_text('changed')
    else:
        value=json.loads(audit.read_text())
        value['selection'][0]['result_sha256' if fault=='audit_hash' else 'passed']='wrong'
        audit.write_text(json.dumps(value))
    with pytest.raises(ValueError):module.authenticate(config,verify_parity=None)
