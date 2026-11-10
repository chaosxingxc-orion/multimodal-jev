import importlib.util
import json
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical

spec = importlib.util.spec_from_file_location('control', Path(__file__).parents[1] / 'tools/jev/compare_frozen_control.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def runs(tmp_path):
    paths = [tmp_path / 'original', tmp_path / 'reused']
    config = dict(route='asr-text', parameters_frozen=True, requests='requests', split='train',
                  source_groups=['g'], device='npu:0', dtype='bfloat16', seed=17,
                  max_context_tokens=4096, max_new_tokens=128, asr_max_new_tokens=128,
                  sampling={'n': 1, 'temperature': 0})
    identity = dict(config=config, models={}, requests_sha256='req', instruction_sha256='prompt')
    row = dict(example_id='request', replicate=0, request_sha256='req', prompt_sha256='prompt',
               input_tokens=123, token_ids=[1, 2], raw_text='{"value":"a"}', format_valid=True,
               schema_valid=True, length_limited=False)
    for path in paths:
        path.mkdir()
        (path / 'identity.json').write_text(json.dumps(identity))
        (path / 'selection.json').write_text('{"ids":["request"]}')
        (path / 'terminal.json').write_text('{"exit_code":0}')
        (path / 'predictions.jsonl').write_bytes(canonical(row))
        (path / 'result.json').write_text(json.dumps({'status': 'passed', 'predictions_sha256': file_hash(path / 'predictions.jsonl')}))
    return paths


def test_control_checks_exact_tokens_even_with_identical_answer(tmp_path):
    paths = runs(tmp_path)
    assert module.compare(*paths)['passed']
    path = paths[1] / 'predictions.jsonl'; row = json.loads(path.read_text())
    row['token_ids'] = [1, 3]; path.write_bytes(canonical(row))
    (paths[1] / 'result.json').write_text(json.dumps({'status': 'passed', 'predictions_sha256': file_hash(path)}))
    report = module.compare(*paths)
    assert not report['passed'] and report['mismatch_counts']['token_ids'] == 1


def test_control_refuses_changed_prompt_identity(tmp_path):
    paths = runs(tmp_path); path = paths[1] / 'identity.json'; data = json.loads(path.read_text())
    data['instruction_sha256'] = 'different'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='identity'): module.compare(*paths)
