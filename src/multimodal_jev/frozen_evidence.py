"""Pin completed ASR evidence for paired frozen-prompt comparisons; no gold IO."""
import hashlib
import json
from pathlib import Path

from .baseline import file_hash
from .consumption import canonical

ASR_SETTINGS = ('dtype', 'device', 'asr_max_new_tokens', 'max_context_tokens')


def export_evidence(run, destination):
    run, destination = Path(run), Path(destination)
    identity = json.loads((run / 'identity.json').read_text())
    config = identity['config']
    result = json.loads((run / 'result.json').read_text())
    terminal = json.loads((run / 'terminal.json').read_text())
    if (terminal['exit_code'] != 0 or result['status'] != 'passed'
            or config['route'] != 'asr-text' or config['parameters_frozen'] is not True
            or result['weight_updates'] != 0 or not result['model_files_unchanged']):
        raise ValueError('completed frozen ASR-text source required')
    if file_hash(run / 'predictions.jsonl') != result['predictions_sha256']:
        raise ValueError('source predictions changed')
    if file_hash(config['requests']) != identity['requests_sha256']:
        raise ValueError('source requests changed')
    selected = json.loads((run / 'selection.json').read_text())['ids']
    rows = [json.loads(line) for line in Path(config['requests']).read_text().splitlines()]
    if config.get('source_groups') is not None:
        rows = [r for r in rows if r['source_group'] in set(config['source_groups'])]
    if selected != [r['id'] for r in rows]:
        raise ValueError('source selection differs from configured population')
    if len(selected) != len(set(selected)) or len(rows) != len(selected) or not rows:
        raise ValueError('source selection coverage changed')
    if result['requests'] != len(rows) or any(r['split'] != config['split'] for r in rows):
        raise ValueError('source population changed')
    expected = {(r['id'], i): hashlib.sha256(canonical(r)).hexdigest()
                for r in rows for i in range(config['sampling']['n'])}
    seen = {}
    for prediction in map(json.loads, (run / 'predictions.jsonl').read_text().splitlines()):
        key = (prediction['example_id'], prediction['replicate'])
        if key in seen: raise ValueError('duplicate source predictions')
        seen[key] = prediction['request_sha256']
    if seen != expected: raise ValueError('source prediction coverage changed')
    records = {}
    for audio_hash in sorted({r['audio_sha256'] for r in rows}):
        path = run / 'transcriptions' / (audio_hash + '.json')
        record = json.loads(path.read_text())
        if record['audio_sha256'] != audio_hash or type(record['transcription']) is not str:
            raise ValueError('source transcription identity changed')
        records[audio_hash] = {'path': str(path.resolve()), 'sha256': file_hash(path)}
    manifest = {
        'schema': 'jev-frozen-asr-evidence-v1', 'source_run': str(run.resolve()),
        'source_identity_sha256': file_hash(run / 'identity.json'),
        'source_result_sha256': file_hash(run / 'result.json'),
        'split': config['split'], 'audio_policy_id': identity['models']['audio_model']['policy_id'],
        'settings': {k: config[k] for k in ASR_SETTINGS},
        'preprocessing_sha256': {k: identity['source_sha256'][k] for k in ('asr.py', 'audio_policy.py')},
        'requests': {r['id']: hashlib.sha256(canonical(r)).hexdigest() for r in rows},
        'records': records,
    }
    with destination.open('xb') as handle:
        handle.write(canonical(manifest))
    return manifest


def load_evidence(pin, rows, config, models):
    if set(pin) != {'path', 'sha256'} or file_hash(pin['path']) != pin['sha256']:
        raise ValueError('ASR evidence manifest pin changed')
    manifest = json.loads(Path(pin['path']).read_text())
    if manifest['schema'] != 'jev-frozen-asr-evidence-v1' or config['route'] != 'asr-text':
        raise ValueError('ASR-text evidence required')
    if (manifest['split'] != config['split']
            or manifest['audio_policy_id'] != models['audio_model']['policy_id']
            or manifest['settings'] != {k: config[k] for k in ASR_SETTINGS}):
        raise ValueError('ASR evidence model/settings/split changed')
    if manifest['preprocessing_sha256'] != {k: file_hash(Path(__file__).with_name(k))
                                          for k in ('asr.py', 'audio_policy.py')}:
        raise ValueError('ASR preprocessing code changed')
    expected = {r['id']: hashlib.sha256(canonical(r)).hexdigest() for r in rows}
    if manifest['requests'] != expected or len(expected) != len(rows):
        raise ValueError('ASR evidence request coverage changed')
    if set(manifest['records']) != {r['audio_sha256'] for r in rows}:
        raise ValueError('ASR evidence audio coverage changed')
    records = {}
    for key, record_pin in manifest['records'].items():
        if file_hash(record_pin['path']) != record_pin['sha256']:
            raise ValueError('ASR transcript changed')
        record = json.loads(Path(record_pin['path']).read_text())
        if record['audio_sha256'] != key or type(record['transcription']) is not str:
            raise ValueError('ASR transcript identity changed')
        records[key] = record
    return records
