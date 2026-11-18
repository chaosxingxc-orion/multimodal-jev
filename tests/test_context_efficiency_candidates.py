import ast
import copy
import json
from pathlib import Path
import pytest
from test_context_dual_measurement_candidate import dual_measured
from test_context_batched_service_candidate import engine_case
from test_context_macro_training_operations import fixture as command_fixture
from tools.reports.measure_context_acoustic_candidate import compare
from tools.reports import context_bucket_training_operations as bucket
from tools.reports import context_macro_training_operations as shared
from tools.reports import run_context_efficiency_successor as sequence


def reports():
    a,b=dual_measured('batched'),dual_measured('batched')
    for mode,r in [('baseline',a),('acoustic',b)]:
        r['mode']=mode
        for p in r['passes']:
            p['acoustic_counts']=dict(conv=[[2,1,8,100]],encoder=[[0,1,2]])
    return dict(baseline=a,acoustic=b)


def test_acoustic_comparison_uses_actual_logits_timers_and_sampling():
    _,rows,_=engine_case();r=compare(reports(),rows)
    assert r['speedup']==1 and r['exact_probabilities_and_logits'] and r['exact_sampling']
    assert r['training_dispatch_allowed'] is False


@pytest.mark.parametrize('fault',['length','memory','sampling','policy','order','missing'])
def test_acoustic_comparison_rejects_corrupt_evidence(fault):
    _,rows,_=engine_case();r=reports();a=r['acoustic'];p=a['passes'][0]
    if fault=='length':p['telemetry']['counts']['language']=[[1]]
    if fault=='memory':p['telemetry']['peak_allocated_bytes']=30
    if fault=='sampling':p['sampling'][0]['seed']+=1
    if fault=='policy':a['policy_id']='c'*64
    if fault=='order':p['order']=[1,0]
    if fault=='missing':a['passes'].pop()
    with pytest.raises(ValueError):compare(r,rows)


@pytest.mark.parametrize('role',['client','preflight','service-0','service-1'])
def test_bucket_keeps_runtime_boundary_and_changes_only_entry(tmp_path,role):
    c,_=command_fixture(tmp_path)
    a=shared.command(c,role);b=bucket.command(c,role)
    boundary=a.index(shared.IMAGE)+1
    assert a[:boundary]==b[:boundary]
    if role.startswith('service'):assert a==b
    elif role=='preflight':ast.parse(b[-1]);assert 'context_bucket_trainer_candidate import generate' in b[-1]
    else:assert 'context_bucket_trainer_candidate.py' in ' '.join(b)


def test_sequence_preserves_order_and_does_not_promote_acoustic(tmp_path,monkeypatch):
    writes=lambda p,d:Path(p).write_text(json.dumps(d))
    a=tmp_path/'acoustic.json';b=tmp_path/'bucket.json';driver=tmp_path/'driver.json';terminal=tmp_path/'terminal.json'
    writes(a,{'arm':'acoustic'});writes(b,{'parent_operations':{'arm':'baseline'},'arm':'bucket'})
    writes(driver,{'pid':1});writes(terminal,dict(exit_code=0,status='three_target_width_parity_audited'))
    events=[]
    monkeypatch.setattr(sequence,'pins',lambda c:None);monkeypatch.setattr(sequence,'check_paths',lambda c:None)
    monkeypatch.setattr(sequence.acoustic,'validate',lambda c:None);monkeypatch.setattr(sequence.bucket,'validate',lambda c:None)
    monkeypatch.setattr(sequence,'wait_driver',lambda *a:events.append('wait'))
    monkeypatch.setattr(sequence,'audit_collected',lambda c:events.append('baseline_audit') or {})
    monkeypatch.setattr(sequence.acoustic,'namespace',lambda c:{'launch':lambda *args:events.append('acoustic') or {'exact_probabilities_and_logits':False}})
    monkeypatch.setattr(sequence.bucket,'run',lambda *a:events.append('bucket'))
    monkeypatch.setattr(sequence,'collect_and_audit',lambda *a,**k:events.append('bucket_audit') or {})
    monkeypatch.setattr(sequence,'compare_training',lambda *a:events.append('compare') or {})
    c=dict(output=str(tmp_path/'result'),acoustic_operations=str(a),bucket_operations=str(b),predecessor_driver=str(driver),predecessor_terminal=str(terminal))
    sequence.run(c,{})
    assert events==['wait','baseline_audit','acoustic','bucket','bucket_audit','compare']
    assert json.loads((tmp_path/'result/terminal.json').read_text())['automatic_promotion'] is False
