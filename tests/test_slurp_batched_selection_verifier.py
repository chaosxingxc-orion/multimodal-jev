"""Candidate terminal selection must reject partial or misbound evidence."""
import importlib.util
import json
import math
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.span_batch_training_config import PROFILE

ROOT = Path(__file__).parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

old = module('original_parity_tests', ROOT/'tests/test_slurp_simple_parity_verifier.py')
m = module('candidate_parity', ROOT/'tools/jev/verify_slurp_batched_parity.py')


def fixture(root, monkeypatch, full=True):
    gate, put = old.fixture(root, monkeypatch, count=9226 if full else 256)
    train = root/'train'
    release = root/'release'
    put(release/'MANIFEST.json', {'release': 'fixture'})
    digest = file_hash(release/'MANIFEST.json')
    put(train/'config.json', dict(task_mode='direct_final', candidate_profile=PROFILE,
        total_updates=634, base_id='b', runtime_profile='fixed',
        release=str(release), release_sha256=digest))
    launch = json.loads((train/'launch.json').read_text())
    launch['config_sha256'] = file_hash(train/'config.json'); put(train/'launch.json', launch)
    p = gate/'state-verified.json'; state = json.loads(p.read_text())
    state.update(original_adapter_id='a', updates=634 if full else 4, epoch=2 if full else 0, next_batch=0 if full else 4)
    put(p, state)
    p = gate/'inputs.json'; data = json.loads(p.read_text())
    data['population_mode'] = 'full_selection' if full else 'prefix'; put(p, data)
    p = gate/'config.json'; c = json.loads(p.read_text())
    c.update(candidate_component='full', rollout_candidate_component='full_native_lora', runtime_profile='fixed')
    c['pins'][str(gate/'inputs.json')] = file_hash(gate/'inputs.json'); put(p, c)
    launch = json.loads((gate/'launch.json').read_text()); launch['config_sha256'] = file_hash(p)
    put(gate/'launch.json', launch)
    audit = dict(backend_root=str(gate), backend_launch=str(gate/'launch.json'),
        backend_observed=str(gate/'observed-final.json'), training_launch=str(train/'launch.json'),
        training_observed=str(gate/'training-observed.json'), checkpoint_evidence=str(gate/'state-verified.json'))
    audit['pins'] = {audit[k]:file_hash(audit[k]) for k in ('checkpoint_evidence','training_observed')}
    path = root/'audit.json'; put(path, audit)
    monkeypatch.setattr(m, 'checkpoint_identity', lambda path: ('a', {}))
    import multimodal_jev.slurp_official_training as released
    import multimodal_jev.simple_selection_inputs as selection
    def load(path, sha, role, *, purpose):
        assert (path, sha, role, purpose) == (str(release), digest, 'selection', 'evaluate')
        return list(range(9226)), {}, {}
    def verify_inputs(artifact, rows, task, sha):
        assert len(rows) == 9226 and sha == digest
        assert artifact['population_mode'] == 'full_selection'
    monkeypatch.setattr(released, 'load', load)
    monkeypatch.setattr(selection, 'verify', verify_inputs)
    return path, gate, put


@pytest.mark.parametrize('full', [False, True])
def test_complete_probability_recomputation(tmp_path, monkeypatch, full):
    path, _, _ = fixture(tmp_path, monkeypatch, full)
    report = m.verify(path)
    assert report['requests'] == (9226 if full else 256)
    assert report['updates'] == (634 if full else 4)
    assert report['maximum_logprob_error'] == 0
    assert report['files'][str(path)] == file_hash(path)


@pytest.mark.parametrize('fault', ['partial', 'forged_pass', 'foreign_request', 'nonterminal',
    'wrong_cursor', 'wrong_profile', 'wrong_candidate', 'wrong_adapter', 'live', 'unpinned_state'])
def test_reject_invalid_full_selection(tmp_path, monkeypatch, fault):
    path, gate, put = fixture(tmp_path, monkeypatch)
    audit = json.loads(path.read_text())
    if fault in ('partial', 'forged_pass', 'foreign_request'):
        p = gate/'run/vllm/result.json'; data = json.loads(p.read_text())
        if fault == 'partial': data['records'].pop()
        elif fault == 'foreign_request': data['records'][0]['request_id'] = 'other'
        else: data['records'][-1]['candidate_logprobs'] = [[math.log(.5)] + [math.log(.5/59)]*59]
    elif fault in ('nonterminal', 'wrong_cursor', 'wrong_adapter'):
        p = gate/'state-verified.json'; data = json.loads(p.read_text())
        if fault == 'nonterminal': data['updates'] = 633
        elif fault == 'wrong_cursor': data['next_batch'] = 1
        else: data['original_adapter_id'] = 'other'
    elif fault == 'live':
        p = gate/'observed-final.json'; data = json.loads(p.read_text()); data['state']['Status'] = 'running'
    elif fault == 'unpinned_state':
        audit['pins'].pop(audit['checkpoint_evidence']); put(path, audit)
        with pytest.raises(ValueError): m.verify(path)
        return
    else:
        p = gate/'config.json'; data = json.loads(p.read_text())
        data['runtime_profile' if fault == 'wrong_profile' else 'rollout_candidate_component'] = 'other'
    put(p, data)
    if str(p) in audit['pins']: audit['pins'][str(p)] = file_hash(p)
    if p == gate/'config.json':
        launch = json.loads((gate/'launch.json').read_text()); launch['config_sha256'] = file_hash(p)
        put(gate/'launch.json', launch)
    put(path, audit)
    with pytest.raises(ValueError): m.verify(path)


def finalizer(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools/jev'))
    return module('candidate_finalizer', ROOT/'tools/jev/finalize_slurp_batched_selection.py')


def test_finalizer_routes_external_execution_receipts_and_preserves_metric_differences(tmp_path, monkeypatch):
    path, gate, _ = fixture(tmp_path, monkeypatch)
    f = finalizer(monkeypatch)
    monkeypatch.setattr(f, 'verify', m.verify)
    captured = []
    def score(config):
        captured.append(config)
        for p, digest in config['pins'].items(): assert file_hash(p) == digest
        accuracy = .2 if '/hf/' in config['backend_results'] else .21
        return dict(policy_id='b:a', accuracy=accuracy, macro_F1_fixed60=.1,
                    macro_recall_supported=.2, invalid_rate=0)
    monkeypatch.setattr(f, 'score_run', score)
    out = tmp_path/'metrics'; out.mkdir(); result = f.run(path, out)
    assert len(captured) == 2
    audit = json.loads(path.read_text())
    assert all(c['launch'] == audit['backend_launch'] and c['observed'] == audit['backend_observed'] for c in captured)
    assert result['vllm_minus_hf']['accuracy'] == pytest.approx(.01)
    assert json.loads((out/'result.json').read_text())['recordings'] == 9226


def test_finalizer_rejects_prefix_before_scoring(tmp_path, monkeypatch):
    path, _, _ = fixture(tmp_path, monkeypatch, full=False)
    f = finalizer(monkeypatch)
    monkeypatch.setattr(f, 'verify', m.verify)
    monkeypatch.setattr(f, 'score_run', lambda _: pytest.fail('prefix must not be scored'))
    out = tmp_path/'metrics'
    with pytest.raises(ValueError, match='terminal634'): f.run(path, out)
    assert not out.exists()
