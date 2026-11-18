"""Collected raw audit bytes retain scope and complete ordered update evidence."""
import json
from pathlib import Path
import runpy
import sys
import pytest
from multimodal_jev.resident_service_admission import IMAGE


@pytest.mark.parametrize('fault',[None,'live','incomplete','policy','missing_selection','wrong_requests','promoted','mode','initial'])
def test_collector_refuses_incomplete_audit_and_preserves_exact_bytes(tmp_path,monkeypatch,fault):
    import subprocess
    tag='stage2-test-audit';target=Path(__file__).parents[1]/'tools/jev/collect_simple_stage2_audit.py'
    launch=dict(container_id='audit',image=IMAGE,remote_root='/data/audit',completed_updates=3)
    observed=dict(container_id='audit',image=IMAGE,state=dict(Status='running' if fault=='live' else 'exited',ExitCode=0))
    for suffix,value in [('launch',launch),('observed',observed)]:
        (tmp_path/(tag+'-'+suffix+'.json')).write_text(json.dumps(value))
    report=dict(passed=True,stage='simple_stage2_rl',ranks=4,updates=3,new_updates=3,
        audit_mode='stopped_prefix',complete_training=False,exact_master_export=True,
        exact_peft_reload=True,exact_pre_update_recovery=True,
        policies={str(i):'policy'+str(i) for i in range(4)},
        checks=[dict(passed=True,updates=i,requests=128,policy_id='policy'+str(i)) for i in range(1,4)],
        selection=[dict(passed=True,updates=0)],initial_state_checks=[dict(passed=True)]*4)
    if fault=='incomplete':report['checks'].pop()
    elif fault=='policy':report['checks'][1]['policy_id']='wrong'
    elif fault=='missing_selection':report['selection']=[]
    elif fault=='wrong_requests':report['checks'][1]['requests']=64
    elif fault=='promoted':report['complete_training']=True
    elif fault=='mode':report['audit_mode']='unknown'
    elif fault=='initial':report['initial_state_checks']=[]
    raw=json.dumps(report,indent=2)+'\n';calls=[]
    def remote(*args,**kwargs):calls.append(args);return json.dumps(dict(raw=raw,verified_files=9))
    monkeypatch.setattr(subprocess,'check_output',remote)
    monkeypatch.setattr(sys,'argv',[str(target),'--audit-tag',tag,'--state-dir',str(tmp_path)])
    if fault:
        with pytest.raises(ValueError):runpy.run_path(str(target),run_name='__main__')
        assert not (tmp_path/(tag+'-verified.json')).exists()
        if fault=='live':assert not calls
    else:
        runpy.run_path(str(target),run_name='__main__')
        assert (tmp_path/(tag+'-verified.json')).read_text()==raw
