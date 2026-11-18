import json
from pathlib import Path
import pytest
from tools.reports import run_context_kv_training_study as s
from tools.reports import watch_context_kv_training as w
from tools.reports import observe_context_macro_training as original


@pytest.mark.parametrize('failure', [None, 'predecessor', 'admission', 'baseline', 'training', 'audit', 'compare'])
def test_successor_is_single_attempt_and_preserves_failure(tmp_path, monkeypatch, failure):
    def write(name, obj):
        path = tmp_path/(name+'.json'); path.write_text(json.dumps(obj)); return str(path)
    events = []
    ops = dict(parent_operations={'baseline': True})
    probe = dict(operations_root=str(tmp_path), tag='probe')
    c = dict(output=str(tmp_path/'out'), predecessor_driver=write('driver', {'pid': 1}),
             predecessor_terminal=write('terminal', dict(exit_code=int(failure=='predecessor'),
                                                        status='acoustic_training100_compared')))
    monkeypatch.setattr(s, 'validate', lambda c: (ops, probe))
    monkeypatch.setattr(s, 'pins', lambda c: None)
    monkeypatch.setattr(s, 'check_paths', lambda c: None)
    monkeypatch.setattr(s, 'wait_driver', lambda *a: events.append('wait'))
    def stage(name):
        events.append(name)
        if name == failure: raise ValueError('failed '+name)
        return dict(exact_probabilities_and_logits=True, exact_sampling=True)
    monkeypatch.setattr(s.probe_ops, 'audit', lambda *a: stage('admission'))
    monkeypatch.setattr(s, 'audit_collected', lambda *a, **kw: stage('baseline'))
    def train(c, provenance):
        assert c['finite_training_dispatch_allowed'] is True
        return stage('training')
    monkeypatch.setattr(s.training, 'run', train)
    monkeypatch.setattr(s, 'collect_and_audit', lambda *a, **kw: stage('audit'))
    monkeypatch.setattr(s, 'compare', lambda *a: stage('compare'))
    if failure:
        with pytest.raises(ValueError): s.run(c, {})
    else: s.run(c, {})
    expected = ['wait', 'admission', 'baseline', 'training', 'audit', 'compare']
    if failure == 'predecessor': expected = ['wait']
    elif failure: expected = expected[:expected.index(failure)+1]
    assert events == expected
    terminal = json.loads((tmp_path/'out/terminal.json').read_text())
    assert terminal['exit_code'] == int(failure is not None)
    if failure: assert terminal['restart'] is False and terminal['preserve_handles'] is True
    else: assert terminal['automatic_promotion'] is False


@pytest.mark.parametrize('fault', [None, 'order', 'cache', 'image', 'incomplete'])
def test_real_comparator_binding_requires_full_matched100(tmp_path, monkeypatch, fault):
    baseline = dict(operations_root=str(tmp_path/'base'), manifest_sha256='base')
    candidate = dict(operations_root=str(tmp_path/'new'), manifest_sha256='new', parent_operations=baseline)
    for ops in (baseline, candidate):
        root = Path(ops['operations_root'])/'collected/run'; root.mkdir(parents=True)
        for i in range(100):
            rows = [i, i+100]
            if fault=='order' and ops is candidate and i==99: rows.reverse()
            (root/f'batch-{i:04d}.json').write_text(json.dumps(dict(records=rows)))
    monkeypatch.setattr(s.training, 'validate', lambda c: None)
    def summary(root, **kw):
        assert kw == dict(start_update=0, stop_update=100, warmup_updates=2)
        new = 'new' in str(root)
        return dict(all_updates=dict(complete_seconds=80 if new else 100),
                    measured=dict(complete_seconds=60 if new else 90),
                    steps=[dict(reference_misses=127 if fault=='cache' and new else 128)]*100)
    monkeypatch.setitem(s._compare.__globals__, 'summarize', summary)
    audits = [dict(status='finite_dp2_efficiency_audited', files_sha256=n, image='same') for n in ('base','new')]
    if fault=='image': audits[1]['image']='other'
    if fault=='incomplete': audits[1]['status']='partial'
    if fault:
        with pytest.raises(ValueError): s.compare(baseline,candidate,audits)
    else:
        result = s.compare(baseline,candidate,audits)
        assert result['all_updates_speedup']==1.25
        assert result['schema']=='jev-kv-training-efficiency-v1'


def test_observer_binding_does_not_mutate_original():
    assert w.observe.__globals__['validate'] is s.training.validate
    assert w.observe.__globals__['command'] is s.training.command
    assert original.observe.__globals__['validate'] is not s.training.validate
    assert w.run.__globals__['observe'] is w.observe
    assert w.run.__globals__['INTERVAL']==900
