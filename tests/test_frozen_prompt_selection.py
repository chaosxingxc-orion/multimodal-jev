import importlib.util
import json
from pathlib import Path
import sys
import hashlib
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical

tools = Path(__file__).parents[1] / 'tools/jev'
spec = importlib.util.spec_from_file_location('compare_frozen_control', tools / 'compare_frozen_control.py')
control = importlib.util.module_from_spec(spec); spec.loader.exec_module(control)
sys.modules['compare_frozen_control'] = control
spec = importlib.util.spec_from_file_location('selection', tools / 'select_frozen_prompt.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def setup(tmp_path, monkeypatch, candidate_wins):
    monkeypatch.setattr(module, 'compare', lambda *a: {'passed': True})
    monkeypatch.setattr(module, 'load_evidence', lambda *a: {})
    requests = []; targets = []
    for i in range(256):
        row = {'id': str(i), 'source_group': 'g' + str(i), 'split': 'train'}
        requests.append(row)
        targets.append({'id': str(i), 'target': 'a', 'metadata': {},
                        'request_sha256': hashlib.sha256(canonical(row)).hexdigest()})
    manifest = tmp_path / 'train.requests.jsonl'; manifest.write_bytes(b''.join(map(canonical, requests)))
    gold = tmp_path / 'train.targets.jsonl'; gold.write_bytes(b''.join(map(canonical, targets)))
    split = tmp_path / 'split.json'; split.write_text(json.dumps({'learning': ['learning'],
        'selection': [r['source_group'] for r in requests]}))
    cfg = {'route': 'asr-text', 'parameters_frozen': True, 'requests': str(manifest),
           'split': 'train', 'source_groups': [r['source_group'] for r in requests],
           'dtype': 'bfloat16', 'device': 'npu:0', 'seed': 17, 'sampling': {'temperature': 0, 'n': 1},
           'max_context_tokens': 4096, 'max_new_tokens': 128, 'asr_max_new_tokens': 128,
           'asr_evidence': {'path': 'fixture', 'sha256': 'fixture'}}
    paths = []
    for name, wins in [('control', 54), ('p1', candidate_wins)]:
        path = tmp_path / name; path.mkdir(); paths.append(path)
        prompt = path / 'prompt.txt'; prompt.write_text('original instruction' if name == 'control' else 'new')
        identity = {'config': {**cfg, 'instruction': str(prompt)}, 'models': {},
                    'requests_sha256': file_hash(manifest), 'instruction_sha256': file_hash(prompt)}
        (path / 'identity.json').write_text(json.dumps(identity))
        (path / 'terminal.json').write_text('{"exit_code":0}')
        predictions = [{'example_id': str(i), 'replicate': 0, 'request_sha256': targets[i]['request_sha256'],
                        'raw_text': '{"value":"a"}' if i < wins else '{"value":"b"}',
                        'input_tokens': 10, 'token_ids': [1, 2]} for i in range(256)]
        (path / 'predictions.jsonl').write_bytes(b''.join(map(canonical, predictions)))
        (path / 'result.json').write_text(json.dumps({'status': 'passed', 'weight_updates': 0,
             'model_files_unchanged': True, 'elapsed_seconds': 1,
             'predictions_sha256': file_hash(path / 'predictions.jsonl')}))
    review = tmp_path / 'review.json'
    review.write_text(json.dumps({'evaluation_eligible': ['p1'], 'candidates': [
        {'id': 'p1', 'decision': 'eligible_for_selection_evaluation',
         'instruction_sha256': file_hash(paths[1] / 'prompt.txt')}]}))
    return (paths[0], paths[0], {'p1': paths[1]}, review, gold, split)


@pytest.mark.parametrize('wins,expected', [(53, 'parent'), (54, 'parent'), (55, 'p1')])
def test_original_wins_ties_even_when_candidate_shorter(tmp_path, monkeypatch, wins, expected):
    args = setup(tmp_path, monkeypatch, wins)
    assert module.select(*args)['selected_id'] == expected


def test_selection_rejects_control_failure_and_missing_candidate(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch, 60)
    monkeypatch.setattr(module, 'compare', lambda *a: {'passed': False})
    with pytest.raises(ValueError, match='control'): module.select(*args)
    monkeypatch.setattr(module, 'compare', lambda *a: {'passed': True})
    args[2].clear()
    with pytest.raises(ValueError, match='eligible'): module.select(*args)


@pytest.mark.parametrize('change', ['test', 'sampling', 'prompt', 'prediction'])
def test_selection_rejects_changed_conditions(tmp_path, monkeypatch, change):
    args = setup(tmp_path, monkeypatch, 60)
    run = args[2]['p1']
    identity_path = run / 'identity.json'; identity = json.loads(identity_path.read_text())
    if change == 'test': identity['config']['split'] = 'test'
    if change == 'sampling': identity['config']['sampling']['n'] = 4
    if change == 'prompt': (run / 'prompt.txt').write_text('edited after review')
    if change == 'prediction': (run / 'predictions.jsonl').write_text('')
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError): module.select(*args)
