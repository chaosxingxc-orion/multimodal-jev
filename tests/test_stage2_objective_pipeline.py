import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.reports import run_stage2_objective_pipeline as pipeline
from tools.reports import stage2_objective_run_operations as operations
from tools.reports import prepare_stage2_objective_run as preparation


def test_private_graph_preserves_validators_and_parallel_source():
    engine=SimpleNamespace(reconstruct=lambda c: {'fixture': c})
    prepare,ops=pipeline.implementation(engine)
    assert prepare.prepare.__code__ is preparation.prepare.__code__
    assert ops.admission.__code__ is operations.admission.__code__
    source=prepare.expected.__globals__['prefix'].expected.__globals__ if hasattr(prepare.expected.__globals__['prefix'],'expected') else None
    assert ops.admission.__globals__['preparation'] is prepare
    assert ops.audit.__globals__['preparation'] is prepare
    assert ops.validate.__globals__['admission'] is ops.admission
    assert ops.run.__globals__['launch'] is ops.launch


@pytest.mark.parametrize('dispatch,interval',[(False,900),(True,60)])
def test_rejects_unauthorized_or_frequent_monitoring(dispatch,interval):
    with pytest.raises(ValueError,match='dispatch'):
        pipeline.validate(dict(stage2_training_dispatch_allowed=dispatch,monitor_interval_seconds=interval))


def test_failed_acceptance_never_supplies_training_evidence(tmp_path,monkeypatch):
    terminal=tmp_path/'terminal.json';terminal.write_text(json.dumps(dict(exit_code=1,status='failed')))
    monkeypatch.setattr(pipeline,'validate',lambda c: {'same':True})
    monkeypatch.setattr(pipeline,'read_bound',lambda c:'{}')
    monkeypatch.setattr(pipeline,'wait_exact',lambda *a:None)
    with pytest.raises(ValueError,match='predecessor failed'):
        pipeline.wait_acceptance(dict(predecessor=dict(driver={},terminal=str(terminal))))


def test_incomplete_collection_never_audited_or_success(tmp_path,monkeypatch):
    monkeypatch.setattr(operations,'launch',lambda *a:dict(status='incomplete'))
    monkeypatch.setattr(operations,'audit',lambda *a:pytest.fail('must not audit incomplete collection'))
    with pytest.raises(ValueError,match='complete Stage2 collection'):
        operations.run(dict(operations_root=str(tmp_path)),{})
    assert json.loads((tmp_path/'finalization/terminal.json').read_text())['exit_code']==1


def test_incomplete_updates_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(operations,'launch',lambda *a:dict(status='collected_independent_cached_training_audit_required'))
    monkeypatch.setattr(operations,'audit',lambda *a:dict(status='stage2_objective_formal_audited',updates=[1],real_query_exposures=81026))
    with pytest.raises(ValueError,match='complete independent'):
        operations.run(dict(operations_root=str(tmp_path),tag='fixture',reference_audit={},training_package=dict(kind='formal')), {})
    assert json.loads((tmp_path/'finalization/terminal.json').read_text())['exit_code']==1
