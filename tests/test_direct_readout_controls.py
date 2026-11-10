import importlib.util
import json
from pathlib import Path

import pytest
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.direct_readout import decision

spec = importlib.util.spec_from_file_location('controls',
    Path(__file__).parents[1] / 'tools/jev/compare_direct_readout_controls.py')
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


def fixture_run(root, condition):
    root.mkdir()
    request_file = root.parent / 'requests.jsonl'
    request_file.write_text('fixture request\n')
    prompt_file = root.parent / 'prompt.txt'
    prompt_file.write_text('Listen and select.')
    options = [{'label': '0', 'value': False}, {'label': '1', 'value': True}]
    selected = decision(options, [0., 1.])
    row = {'example_id': 'group:noul.True', 'replicate': 0,
           'audio_condition': condition, 'alternatives': options, **selected,
           'raw_text': json.dumps({'value': selected['value']}),
           'request_sha256': 'fixture', 'prompt_sha256': 'fixture',
           'source_group': 'group', 'split': 'train',
           'label_token_ids': [48, 49], 'input_tokens': 12}
    (root / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
    atomic_json(root / 'selection.json', {'ids': [row['example_id']]})
    atomic_json(root / 'terminal.json', {'exit_code': 0})
    atomic_json(root / 'result.json', {'status': 'passed', 'audio_condition': condition,
        'model_files_unchanged': True, 'weight_updates': 0, 'requests': 1, 'predictions': 1,
        'predictions_sha256': file_hash(root / 'predictions.jsonl')})
    atomic_json(root / 'identity.json', {'requests_sha256': file_hash(request_file),
        'instruction_sha256': file_hash(prompt_file), 'models': {'audio_model': 'fixture'},
        'runner_sha256': 'fixture', 'source_sha256': {'fixture': 'fixture'},
        'config': {'requests': str(request_file), 'instruction': str(prompt_file),
                   'audio_condition': condition, 'seed': 17}})


def test_identical_control_predictions_and_changed_configuration(tmp_path):
    actual, zeros = tmp_path / 'actual', tmp_path / 'zeros'
    fixture_run(actual, 'actual')
    fixture_run(zeros, 'zeros')
    report = controls.compare(actual, zeros)
    assert report['by_view']['noul.True']['changed_values'] == 0
    assert report['by_view']['noul.True']['actual_indices'] == {'1': 1}
    identity = json.loads((zeros / 'identity.json').read_text())
    identity['config']['seed'] = 18
    atomic_json(zeros / 'identity.json', identity)
    with pytest.raises(AssertionError):
        controls.compare(actual, zeros)


def test_self_consistency_rejects_altered_decision_even_with_updated_hash(tmp_path):
    root = tmp_path / 'actual'
    fixture_run(root, 'actual')
    row = json.loads((root / 'predictions.jsonl').read_text())
    row['selected_index'] = 0
    (root / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
    result = json.loads((root / 'result.json').read_text())
    result['predictions_sha256'] = file_hash(root / 'predictions.jsonl')
    atomic_json(root / 'result.json', result)
    with pytest.raises(AssertionError):
        controls.load_run(root, 'actual')
