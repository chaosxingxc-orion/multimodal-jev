import copy
import json
from pathlib import Path

import pytest

from tools.reports import run_context_efficiency_study as study


def evidence(tmp_path):
    terminal = tmp_path/'terminal.json'
    comparison = tmp_path/'comparison.json'
    terminal.write_text(json.dumps(dict(exit_code=0, status='service_batch32_100_steps_compared')))
    report = dict(schema='jev-service-batch-efficiency-comparison-v1',
        matched_actual_queries=True, matched_reference_misses=True,
        candidate=dict(optimizer_steps=100, all_updates=dict(real_query_exposures=12800)),
        source_audits=[{}, dict(status='finite_dp2_efficiency_audited', files_sha256='parent')])
    comparison.write_text(json.dumps(report))
    c = dict(predecessor_terminal=str(terminal), comparison=str(comparison),
        predecessor_driver='driver', acoustic_operations='acoustic', bucket_operations='bucket')
    c['pins'] = {v:'hash' for v in c.values()}
    return c, report


@pytest.mark.parametrize('fault', [None, 'failed', 'partial', 'different_parent', 'misses', 'pins', 'audit'])
def test_requires_complete_source_baseline_and_independent_reaudit(tmp_path, monkeypatch, fault):
    c, r = evidence(tmp_path)
    if fault == 'failed': Path(c['predecessor_terminal']).write_text(json.dumps(dict(exit_code=1)))
    if fault == 'partial': r['candidate']['optimizer_steps'] = 99
    if fault == 'different_parent': r['source_audits'][1]['files_sha256'] = 'other'
    if fault == 'misses': r['matched_reference_misses'] = False
    if fault == 'pins': del c['pins'][c['comparison']]
    Path(c['comparison']).write_text(json.dumps(r))
    monkeypatch.setattr(study.shared, 'pins', lambda c: None)
    calls = []
    monkeypatch.setattr(study.shared, 'audit_collected', lambda c: calls.append(c) or dict(
        status='failed' if fault == 'audit' else 'finite_dp2_efficiency_audited', files_sha256='parent'))
    if fault:
        with pytest.raises(ValueError): study.admit(c, dict(manifest_sha256='parent'))
        assert len(calls) == int(fault == 'audit')
    else:
        assert study.admit(c, dict(manifest_sha256='parent'))['files_sha256'] == 'parent'
        assert len(calls) == 1


def test_source_only_sequence_never_promotes_or_claims_target_parity(tmp_path, monkeypatch):
    c, _ = evidence(tmp_path)
    c['output'] = str(tmp_path/'result')
    objects = {'acoustic': {}, 'bucket': {'parent_operations': {}}, 'driver': {}}
    monkeypatch.setattr(study.shared, 'read', lambda p: copy.deepcopy(objects[p]))
    monkeypatch.setattr(study.shared, 'pins', lambda c: None)
    monkeypatch.setattr(study, 'check_paths', lambda c: None)
    monkeypatch.setattr(study.shared.acoustic, 'validate', lambda c: None)
    monkeypatch.setattr(study.shared.bucket, 'validate', lambda c: None)
    events = []
    monkeypatch.setattr(study, 'wait_driver', lambda *a: events.append('wait'))
    monkeypatch.setattr(study, 'admit', lambda *a: events.append('audit_baseline') or {})
    monkeypatch.setattr(study.shared.acoustic, 'namespace', lambda c: {'launch': lambda *a:
        events.append('acoustic') or {'exact_probabilities_and_logits': False}})
    monkeypatch.setattr(study.shared.bucket, 'run', lambda *a: events.append('bucket'))
    monkeypatch.setattr(study.shared, 'collect_and_audit', lambda *a, **k: events.append('audit_bucket') or {})
    monkeypatch.setattr(study.shared, 'compare_training', lambda *a: events.append('compare') or {})
    study.run(c, {})
    assert events == ['wait', 'audit_baseline', 'acoustic', 'bucket', 'audit_bucket', 'compare']
    terminal = json.loads((Path(c['output'])/'terminal.json').read_text())
    assert terminal['exit_code'] == 0
    assert not any(terminal[k] for k in ['formal_dispatched', 'automatic_promotion',
                                       'target_parity_verified', 'tail_admission_verified'])
