import csv
import json
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_training import content_hash
from tools.jev.prepare_fsc_context_transfer import compile_native
from tools.reports import score_context_transfer_macro as module


def write(path, value):
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=file_hash(path))


def fsc_fixture(tmp_path):
    task = json.loads(Path('configs/tasks/fsc-span-transfer-v1.json').read_text())
    (tmp_path/'data').mkdir()
    hashes = {}; records = []; assets = []
    for split in ('train', 'valid', 'test'):
        rows = []
        for i in range(3793 if split == 'test' else 1):
            speaker = split+str(i%10)
            relative = f'wavs/speakers/{speaker}/{i}.wav'
            audio = tmp_path/relative
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(f'{split}-{i}'.encode())
            rows.append(dict(path=relative, speakerId=speaker, **task['commands'][i%31]['values']))
            if split == 'test':
                records.append(dict(id='fsc:'+relative, audio=str(audio), audio_sha256=file_hash(audio), speaker_id=speaker))
                request = compile_native(str(audio), task)
                assets.append(dict(id=records[-1]['id'], audio_sha256=records[-1]['audio_sha256'],
                                   request_sha256=content_hash(request.model_input())))
        path = tmp_path/'data'/f'{split}_data.csv'
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        hashes[split] = file_hash(path)
    population = dict(schema='fsc-official-span-population-v1', role='frozen_transfer_evaluation_inputs',
        split='test', dataset_root=str(tmp_path), source_csv_sha256=hashes, records=records, recordings=3793, speakers=10)
    scoring = dict(population=write(tmp_path/'population.json', population), task=write(tmp_path/'task.json', task))
    config = dict(inputs=dict(input_manifest=write(tmp_path/'MANIFEST.json', dict(
        population_sha256=scoring['population']['sha256'], task_sha256=scoring['task']['sha256']))))
    admitted = dict(population=assets, source_rows=records, candidate_ids=[c['id'] for c in task['commands']])
    return config, scoring, admitted, task


def test_fsc_full_csv_gold_matches_exact_audio_task_and_speakers(tmp_path):
    config, scoring, admitted, task = fsc_fixture(tmp_path)
    population, truth = module.fsc_truth(config, scoring, admitted)
    assert len(population) == len(truth) == 3793
    assert truth[population[-1]['id']] == task['commands'][3792%31]['id']
    path = Path(population[-1]['audio']); path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='audio identity'):
        module.fsc_truth(config, scoring, admitted)


def test_fsc_changed_speaker_rejected_before_csv_labels(tmp_path, monkeypatch):
    config, scoring, admitted, _ = fsc_fixture(tmp_path)
    admitted['source_rows'] = [dict(r) for r in admitted['source_rows']]
    admitted['source_rows'][0]['speaker_id'] = 'wrong'
    monkeypatch.setattr(module, 'scoring_targets', lambda *args: pytest.fail('labels read before binding'))
    with pytest.raises(ValueError, match='speaker'):
        module.fsc_truth(config, scoring, admitted)


def minds_fixture(tmp_path, *, fault=None):
    candidates = [f'c{i}' for i in range(14)]
    inputs = [dict(id=str(i), audio_sha256=f'target{i}') for i in range(8168)]
    records = [dict(id=str(i), encoded_audio_sha256=f'source{i}', samples=80+i,
        upstream_split='train', intent=candidates[i%14], language='en-US') for i in range(8168)]
    ledger = [dict(id=str(i), source_audio_sha256=f'source{i}', target_audio_sha256=f'target{i}',
        source_samples=80+i, target_samples=2*(80+i)) for i in range(8168)]
    if fault == 'split': records[-1]['upstream_split'] = 'test'
    if fault == 'source_audio': ledger[-1]['source_audio_sha256'] = 'other'
    if fault == 'target_audio': ledger[-1]['target_audio_sha256'] = 'other'
    if fault == 'duration': ledger[-1]['target_samples'] -= 1
    if fault == 'duplicate': records[-1] = records[0]
    if fault == 'order': ledger.reverse()
    inventory = write(tmp_path/'inventory.json', dict(records=records, intent_ids=candidates))
    protocol = write(tmp_path/'protocol.json', dict(population={'inventory': inventory},
        candidates=[{'id': c} for c in candidates]))
    artifacts = dict(inventory=inventory, protocol=protocol)
    for name, value in [('MANIFEST.json', {}), ('definitions.json', {})]:
        artifacts[name.replace('.', '_')] = write(tmp_path/name, value)
    for name, rows in [('inputs.jsonl', inputs), ('conversion.jsonl', ledger)]:
        path = tmp_path/name
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        artifacts[name.replace('.', '_')] = dict(path=str(path), sha256=file_hash(path))
    contract = write(tmp_path/'contract.json', dict(schema='minds-native-descriptor-scoring-v1',
        threshold=.5, target_tuning=False, artifacts=artifacts))
    config = dict(inputs=dict(inputs=str(tmp_path), protocol=protocol))
    admitted = dict(source_rows=inputs, candidate_ids=candidates)
    return config, {'contract': contract}, admitted


def test_minds_full_resampled_population_bound_to_native_inventory(tmp_path):
    config, scoring, admitted = minds_fixture(tmp_path)
    rows, truth = module.minds_truth(config, scoring, admitted)
    assert len(rows) == len(truth) == 8168
    assert truth['8167'] == dict(intent=f'c{8167%14}', language='en-US')


@pytest.mark.parametrize('fault', ['split', 'source_audio', 'target_audio', 'duration', 'duplicate', 'order'])
def test_minds_rehashed_inconsistent_truth_or_ledger_rejected(tmp_path, fault):
    args = minds_fixture(tmp_path, fault=fault)
    with pytest.raises(ValueError): module.minds_truth(*args)
