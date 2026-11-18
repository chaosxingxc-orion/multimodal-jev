"""Release wiring, immutable packages and authorization; no model results."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.reports.context_stage2_release_components import graph,command,stage_admission
from test_objective_training_candidate import setup,write


def views():
    return graph(SimpleNamespace(reconstruct=lambda *a:None,replay=lambda *a:None,verify=lambda:None),SimpleNamespace())


def test_private_graph_does_not_mutate_shared_functions():
    from tools.reports import prepare_objective_training_candidate as prefix
    from tools.reports import context_objective_training_operations as ops
    from tools.reports import context_cached_training_operations as cached
    old=(prefix.expected,ops.command,cached.command,ops.run)
    v=views()
    assert (prefix.expected,ops.command,cached.command,ops.run)==old
    assert v['tools.reports.context_objective_training_operations'].command is command
    from tools.reports import audit_context_train_parity as parity
    assert v['tools.reports.audit_context_train_parity'].command is parity.command
    assert v['tools.reports.context_cached_training_operations'].command is cached.command
    assert v['tools.reports.prepare_objective_training_candidate'].expected is v['tools.reports.context_stage2_package_components'].expected


def test_prefix_package_pins_helper_and_preserves_forward(tmp_path,monkeypatch):
    c,parent=setup(tmp_path,monkeypatch)
    p=views()['tools.reports.prepare_objective_training_candidate']
    c['manifest_sha256']=p.prepare(c,{})['manifest_sha256'];p.audit(c,{})
    root=Path(c['output']);cfg=json.loads((root/'config.json').read_text())
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(parent/'RUNTIME-SOURCES.json').read_bytes()
    for name in ('run.py','runtime-startup.py','device-visibility.py'):
        assert cfg['pins'][c['remote_root']+'/'+name]==p.sha((root/name).read_bytes())
    assert 'device-visibility.py' in json.loads((root/'PREPARED.json').read_text())['layout']['configs']


def test_recovery_composition_binds_actual_objective_contract():
    v=views();r=v['tools.reports.prepare_objective_recovery'];api=r.implementation()
    assert api.configured.__globals__['expected_contract'] is r.expected_contract
    assert 'context_stage2_runtime_runner' not in api.replay_runner({}) # file content, not an imported worktree module
    assert "preflight_command" in api.replay_runner({})


@pytest.mark.parametrize('kind',['tail','formal'])
def test_downstream_package_same_runtime_and_fixed_reference(tmp_path,monkeypatch,kind):
    from test_stage2_objective_run import fixture
    c=fixture(tmp_path,monkeypatch);c['kind']=kind
    v=views();p=v['tools.reports.prepare_stage2_objective_run']
    c['manifest_sha256']=p.prepare(c,{})['manifest_sha256'];p.audit(c,{})
    root=Path(c['output']);old=Path(c['prefix_package']['output'])
    assert (root/'run.py').read_bytes()==Path(__file__).parents[1].joinpath('tools/reports/context_stage2_runtime_runner.py').read_bytes()
    assert json.loads((root/'config.json').read_text())['reference_dataset']==json.loads((old/'config.json').read_text())['reference_dataset']


def test_formal_gate_does_not_require_efficiency_but_does_require_tail():
    admission=stage_admission({'tools.reports.stage2_objective_run_operations':SimpleNamespace()})
    with pytest.raises(ValueError,match='no standalone'):
        admission(dict(training_package={'kind':'efficiency'},acceptance={}))
    with pytest.raises(ValueError,match='no standalone'):
        admission(dict(training_package={'kind':'formal'},acceptance={'prefix':{},'recovery':{}}))


def test_unapproved_lifecycle_has_no_side_effects():
    v=views()
    with pytest.raises(ValueError,match='explicit stage dispatch'):
        v['tools.reports.context_objective_training_operations'].launch({}, {})
