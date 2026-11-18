import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_context_reference_audit import run as fixture
from tools.reports import context_shared_reference_audit as m


def test_concurrent_reuse_exact_proof_and_isolated_results(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    root, journal = tmp_path/'reference', tmp_path/'journal'
    expected = m.original_audit(root, journal, **args)
    engine = m.ReferenceAuditEngine(io_workers=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(engine.audit, root, journal, **args) for _ in range(2)]
        results = [f.result() for f in futures]
    assert results == [expected, expected]
    assert engine.store.executions == 1 and engine.store.hits == 1
    results[0]['recordings'] = -1
    assert engine.audit(root, journal, **args) == expected


@pytest.mark.parametrize('fault', ['same_stat', 'missing', 'extra_macro', 'policy', 'binding', 'symlink', 'logits'])
def test_cache_never_hides_changed_evidence(tmp_path, monkeypatch, fault):
    args = fixture(tmp_path, monkeypatch)
    root, journal = tmp_path/'reference', tmp_path/'journal'
    engine = m.ReferenceAuditEngine(io_workers=2)
    engine.audit(root, journal, **args)
    path = journal/'batch-00000000/rollout-0000.json'
    if fault == 'same_stat':
        before = path.stat()
        raw = path.read_text()
        value = json.loads(raw)['policy_id']
        path.write_text(raw.replace(value, 'f'*len(value)))
        assert path.stat().st_size == before.st_size
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif fault == 'missing':
        path.unlink()
    elif fault == 'extra_macro':
        (journal/'batch-99999999').mkdir()
    elif fault == 'policy':
        args = copy.deepcopy(args)
        args['identity']['policy_id'] = 'f'*64
    elif fault == 'binding':
        args = copy.deepcopy(args)
        args['input_bindings']['s0']['positions'][0] += 1
    elif fault == 'symlink':
        dest = tmp_path/'copied.json'
        dest.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(dest)
    elif fault == 'logits':
        row = json.loads(path.read_text())
        row['records'][0]['prob'][0] = .5
        path.write_text(json.dumps(row))
    with pytest.raises((ValueError, FileNotFoundError)):
        engine.audit(root, journal, **args)


def test_changed_code_invalidates_reuse(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    engine = m.ReferenceAuditEngine(io_workers=2)
    engine.audit(tmp_path/'reference', tmp_path/'journal', **args)
    original = m.file_hash
    monkeypatch.setattr(m, 'file_hash', lambda p: 'f'*64 if str(p) == m.__file__ else original(p))
    engine.audit(tmp_path/'reference', tmp_path/'journal', **args)
    assert engine.store.executions == 2


def test_mid_audit_mutation_never_publishes_success(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    engine = m.ReferenceAuditEngine(io_workers=2)
    original = m.original_audit
    def mutate(root, journal, **kwargs):
        proof = original(root, journal, **kwargs)
        (journal/'batch-99999999').mkdir()
        return proof
    monkeypatch.setattr(m, 'original_audit', mutate)
    with pytest.raises(ValueError, match='changed during audit'):
        engine.audit(tmp_path/'reference', tmp_path/'journal', **args)
    assert engine.store.executions == 0


def test_scoped_adapter_preserves_upstream_checks(monkeypatch):
    from tools.reports import prepare_context_reference_job as original
    before = dict(original.__dict__)
    engine = m.ReferenceAuditEngine()
    adapted = engine.results_auditor()
    assert adapted.__code__ is original.audit_results.__code__
    for key, value in original.audit_results.__globals__.items():
        if key != 'replay_reference_outputs':
            assert adapted.__globals__[key] is value
    assert adapted.__globals__['replay_reference_outputs'].__self__ is engine
    assert all(original.__dict__[key] is value for key, value in before.items())
