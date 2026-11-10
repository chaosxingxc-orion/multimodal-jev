import hashlib
import json
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.frozen_evidence import export_evidence, load_evidence


def fixture(tmp_path):
    run = tmp_path / 'run'; run.mkdir(); (run / 'transcriptions').mkdir()
    rows = [{'id': 'request-1', 'split': 'train', 'source_group': 'g1',
             'audio_sha256': 'audio-hash', 'input': {'audio': '/audio', 'schema': {'question': 'turn?'}}}]
    requests = tmp_path / 'requests.jsonl'; requests.write_bytes(canonical(rows[0]))
    config = {'route': 'asr-text', 'parameters_frozen': True, 'split': 'train',
              'requests': str(requests), 'source_groups': ['g1'], 'sampling': {'n': 1},
              'dtype': 'bfloat16', 'device': 'npu:0', 'asr_max_new_tokens': 128, 'max_context_tokens': 4096}
    models = {'audio_model': {'policy_id': 'unchanged'}}
    source = Path(__file__).parents[1] / 'src/multimodal_jev'
    identity = {'config': config, 'models': models, 'requests_sha256': file_hash(requests),
                'source_sha256': {k: file_hash(source / k) for k in ('asr.py', 'audio_policy.py')}}
    (run / 'identity.json').write_text(json.dumps(identity))
    (run / 'selection.json').write_text(json.dumps({'ids': ['request-1']}))
    prediction = {'example_id': 'request-1', 'replicate': 0,
                  'request_sha256': hashlib.sha256(canonical(rows[0])).hexdigest()}
    (run / 'predictions.jsonl').write_bytes(canonical(prediction))
    result = {'status': 'passed', 'weight_updates': 0, 'model_files_unchanged': True,
              'requests': 1, 'predictions_sha256': file_hash(run / 'predictions.jsonl')}
    (run / 'result.json').write_text(json.dumps(result))
    (run / 'terminal.json').write_text('{"exit_code":0}')
    transcript = run / 'transcriptions/audio-hash.json'
    transcript.write_text(json.dumps({'audio_sha256': 'audio-hash', 'transcription': 'turn on'}))
    destination = tmp_path / 'evidence.json'
    export_evidence(run, destination)
    pin = {'path': str(destination), 'sha256': file_hash(destination)}
    return run, rows, config, models, pin, transcript


def test_evidence_reuse_exact_and_transcript_tampering_rejected(tmp_path):
    run, rows, cfg, models, pin, transcript = fixture(tmp_path)
    assert load_evidence(pin, rows, cfg, models)['audio-hash']['transcription'] == 'turn on'
    transcript.write_text(transcript.read_text().replace('turn on', 'turn off'))
    with pytest.raises(ValueError, match='transcript changed'): load_evidence(pin, rows, cfg, models)


@pytest.mark.parametrize('change', ['model', 'split', 'dtype', 'schema', 'coverage', 'manifest'])
def test_paired_evidence_rejects_different_condition(tmp_path, change):
    run, rows, cfg, models, pin, transcript = fixture(tmp_path)
    if change == 'model': models['audio_model']['policy_id'] = 'different'
    if change == 'split': cfg['split'] = 'test'
    if change == 'dtype': cfg['dtype'] = 'float16'
    if change == 'schema': rows[0]['input']['schema']['question'] = 'different'
    if change == 'coverage': rows = []
    if change == 'manifest': Path(pin['path']).write_text('{}')
    with pytest.raises(ValueError): load_evidence(pin, rows, cfg, models)


def test_export_requires_success_and_complete_predictions(tmp_path):
    run, rows, cfg, models, pin, transcript = fixture(tmp_path)
    (run / 'terminal.json').write_text('{"exit_code":1}')
    with pytest.raises(ValueError, match='completed'): export_evidence(run, tmp_path / 'fail.json')
    (run / 'terminal.json').write_text('{"exit_code":0}')
    (run / 'predictions.jsonl').write_text('')
    result = json.loads((run / 'result.json').read_text())
    result['predictions_sha256'] = file_hash(run / 'predictions.jsonl')
    (run / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='coverage'): export_evidence(run, tmp_path / 'fail.json')
