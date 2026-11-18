import copy
import json

import pytest

from multimodal_jev.baseline import file_hash
from tools.reports import prepare_context_acoustic_formal as adapter
from tools.reports import prepare_context_macro_formal as shared


def inputs(tmp_path, monkeypatch):
    local = tmp_path/'local.json'
    local.write_text(json.dumps(dict(schema='jev-local-macros-v1', macros={})))
    paths = {}
    for name in adapter.efficiency.ARMS:
        path = tmp_path/(name+'.json')
        path.write_text(json.dumps(dict(manifest_sha256=name, remote_root='/remote/'+name)))
        paths[name] = str(path)
    cfg = dict(operations=paths, pins={p:file_hash(p) for p in paths.values()})
    template = tmp_path/'report.json'; template.write_text(json.dumps(cfg))
    report = dict(status='complete_efficiency_evidence_reconstructed',
                  manifests={n:n for n in paths}, fastest_observed_arm='acoustic_kv16')
    calls = []
    def reconstruct(c):
        assert c == cfg
        calls.append('reconstructed')
        return copy.deepcopy(report)
    monkeypatch.setattr(adapter.efficiency, 'report', reconstruct)
    c = dict(selected_arm='acoustic', selection_reason='Retain measured 1 GiB runtime.',
             recommendation_config=str(template), recommendation_local=str(local))
    return c, cfg, report, calls


@pytest.mark.parametrize('selected', adapter.efficiency.ARMS)
def test_explicit_selection_retains_all_arms_for_overlap_checks(tmp_path, monkeypatch, selected):
    c, cfg, report, calls = inputs(tmp_path, monkeypatch); c['selected_arm'] = selected
    out = adapter.choice(c)
    assert calls == ['reconstructed']
    assert out[3] == selected and out[5] == selected
    assert out[4]['manifest_sha256'] == selected
    assert out[2]['configured_selection'] == selected
    assert out[2]['selection_reason'] == c['selection_reason']
    assert set(out[0]['arms']) == set(adapter.efficiency.ARMS)
    for name, spec in out[0]['arms'].items():
        assert shared.load_config(spec['config'], spec['local'])[0]['manifest_sha256'] == name


@pytest.mark.parametrize('fault', ['pending', 'failed', 'manifest', 'changed', 'unsupported', 'reason'])
def test_incomplete_or_changed_evidence_refused_before_builder(tmp_path, monkeypatch, fault):
    c, cfg, report, calls = inputs(tmp_path, monkeypatch)
    if fault in ('pending', 'failed'): report['status'] = fault
    elif fault == 'manifest': report['manifests']['acoustic'] = 'different'
    elif fault == 'changed':
        from pathlib import Path
        Path(cfg['operations']['acoustic']).write_text('{}')
    elif fault == 'unsupported': c['selected_arm'] = 'bucket'
    else: c['selection_reason'] = ' '
    monkeypatch.setattr(adapter, 'bind', lambda *a, **k:pytest.fail('no builder before evidence'))
    with pytest.raises(ValueError): adapter.prepare(c, {})


@pytest.mark.parametrize('selected', adapter.efficiency.ARMS)
def test_shared_builder_uses_matching_service_validator_without_mutation(tmp_path, monkeypatch, selected):
    c, cfg, report, calls = inputs(tmp_path, monkeypatch); c['selected_arm'] = selected
    original_validator = shared.validate
    # Real FunctionType binding must isolate shared module globals.
    def fake_builder(c, provenance):
        selected_choice = efficiency_choice(c)
        return validate, selected_choice, provenance
    monkeypatch.setattr(shared, 'prepare', fake_builder)
    validator, selected_choice, provenance = adapter.prepare(c, {'test':True})
    expected = adapter.baseline.validate if selected == 'service32' else adapter.acoustic.validate
    assert validator is expected and selected_choice[3] == selected
    assert provenance == {'test':True}
    assert shared.validate is original_validator
    assert calls == ['reconstructed']
