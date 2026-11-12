from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import report_intent_update_metrics as module


def test_independent_acceptance_failure_precedes_metrics(monkeypatch):
    def fail(*args):raise ValueError('backend mismatch')
    monkeypatch.setattr(module,'compare',fail)
    monkeypatch.setattr(module,'read',lambda *a:pytest.fail('read training before acceptance'))
    monkeypatch.setattr(module,'summarize',lambda *a:pytest.fail('reported before acceptance'))
    with pytest.raises(ValueError,match='backend mismatch'):module.run('/training','/suite')


def test_classification_uses_requested_template_and_preserves_policy():
    checked={}
    for key,adapter in [('full_before','old'),('full_after','new')]:
        checked[key]=dict(base_id='base',adapter_id=adapter,reports={
            wire:{'metrics':{k:value for k in module.CLASSIFICATION_METRICS}}
            for wire,value in [('D',.1),('Q',.2)]})
    result=module.classification(checked,'Q')
    assert result['before']['adapter_id']=='old' and result['after']['adapter_id']=='new'
    assert result['after']['metrics']['final_accuracy']==.2
