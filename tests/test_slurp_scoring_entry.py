"""Scoring-entry custody tests; official metric numerics are tested separately."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_score import prepare_predictions

spec = importlib.util.spec_from_file_location('slurp_scoring_entry',
    Path(__file__).resolve().parents[1]/'tools/jev/score_slurp_native.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def fixture(tmp_path, monkeypatch):
    release = tmp_path/'release'; release.mkdir()
    run = tmp_path/'run'; run.mkdir()
    repository = tmp_path/'upstream'; repository.mkdir()
    atomic_json(repository/'FETCH.json', {})
    taxonomy = {'scenario_action_pairs': [['email', 'send']], 'entity_types': []}
    (release/'taxonomy.json').write_bytes(canonical(taxonomy))
    request = dict(id='r1', split='train', source_group='g1', audio_sha256='audio',
        input=dict(audio='/audio', schema_sha256=file_hash(release/'taxonomy.json')))
    digest = hashlib.sha256(canonical(request)).hexdigest()
    target = dict(scenario='email', action='send', entities=[])
    gold = dict(id='r1', target=target, metadata={'file': 'r1.flac'}, request_sha256=digest)
    prediction = dict(example_id='r1', replicate=0, request_sha256=digest,
        raw_text=json.dumps(target), audio_condition='actual')
    for name, row in [('train.requests.jsonl', request), ('train.targets.jsonl', gold)]:
        (release/name).write_bytes(canonical(row))
    atomic_json(release/'MANIFEST.json', dict(release_id='release',
        identity={'taxonomy_sha256': file_hash(release/'taxonomy.json')},
        splits={'train': dict(recordings=1, requests_sha256=file_hash(release/'train.requests.jsonl'),
            targets_sha256=file_hash(release/'train.targets.jsonl'))}))
    cfg = dict(task_format='slurp-native', sampling={'temperature': 0, 'n': 1},
        route='audio-direct', parameters_frozen=True, split='train', purpose='baseline',
        requests=str(release/'train.requests.jsonl'))
    identity = dict(config=cfg, requests_sha256=file_hash(release/'train.requests.jsonl'),
        taxonomy_sha256=file_hash(release/'taxonomy.json'), models={'audio_model': {'policy_id': 'model'}},
        instruction_sha256='instruction', runner_sha256='runner')
    result = dict(status='passed', weight_updates=0, route='audio-direct',
        model_files_unchanged=True, audio_condition='actual', requests=1, predictions=1)
    atomic_json(run/'selection.json', {'ids': ['r1']})
    atomic_json(run/'terminal.json', {'exit_code': 0})
    def save():
        (run/'predictions.jsonl').write_bytes(canonical(prediction))
        result['predictions_sha256'] = file_hash(run/'predictions.jsonl')
        atomic_json(run/'identity.json', identity); atomic_json(run/'result.json', result)
    save()
    monkeypatch.setattr(entry, 'load_official_metrics', lambda _: None)
    def score(requests, targets, predictions, taxonomy, _):
        normalized, report = prepare_predictions(requests, targets, predictions, taxonomy)
        return report, normalized
    monkeypatch.setattr(entry, 'score_native', score)
    args = (run, release/'train.targets.jsonl', repository, tmp_path/'score')
    return args, identity, result, prediction, save


def test_records_route_model_and_prompt_provenance(tmp_path, monkeypatch):
    args, identity, _, _, _ = fixture(tmp_path, monkeypatch)
    report = entry.score_run(*args)
    assert report['audio_task_result'] is True and report['audio_condition'] == 'actual'
    assert report['models'] == identity['models']
    assert report['instruction_sha256'] == 'instruction'
    assert report['identity_sha256'] == file_hash(args[0]/'identity.json')
    assert report['examples'] == 1


def test_zero_audio_is_explicit_diagnostic(tmp_path, monkeypatch):
    args, identity, result, prediction, save = fixture(tmp_path, monkeypatch)
    identity['config']['audio_condition'] = result['audio_condition'] = prediction['audio_condition'] = 'zeros'
    save(); report = entry.score_run(*args)
    assert report['audio_task_result'] is False and report['audio_condition'] == 'zeros'


@pytest.mark.parametrize('bad', ['config_route', 'result_route', 'unfrozen', 'adapter',
    'weights_changed', 'condition', 'prediction_condition', 'requests', 'predictions', 'boolean_count',
    'test_zero'])
def test_rejects_inconsistent_run_before_metric_scoring(tmp_path, monkeypatch, bad):
    args, identity, result, prediction, save = fixture(tmp_path, monkeypatch)
    cfg = identity['config']
    if bad == 'config_route': cfg['route'] = 'asr-text'
    if bad == 'result_route': result['route'] = 'asr-text'
    if bad == 'unfrozen': cfg['parameters_frozen'] = False
    if bad == 'adapter': cfg['adapter'] = '/adapter'
    if bad == 'weights_changed': result['model_files_unchanged'] = False
    if bad == 'condition': result['audio_condition'] = 'zeros'
    if bad == 'prediction_condition': prediction['audio_condition'] = 'zeros'
    if bad == 'requests': result['requests'] = 2
    if bad == 'predictions': result['predictions'] = 2
    if bad == 'boolean_count': result['requests'] = True
    if bad == 'test_zero':
        cfg['split'] = 'test'
        cfg['audio_condition'] = result['audio_condition'] = prediction['audio_condition'] = 'zeros'
    save()
    monkeypatch.setattr(entry, 'score_native', lambda *a: pytest.fail('invalid run reached scorer'))
    with pytest.raises(ValueError): entry.score_run(*args)
    assert not args[-1].exists()
