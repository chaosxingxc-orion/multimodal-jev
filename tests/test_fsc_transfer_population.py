import csv
import copy
from pathlib import Path
import pytest
from test_fsc_span_transfer import config
from multimodal_jev.baseline import file_hash
from multimodal_jev.fsc_transfer_population import build_population, scoring_targets


def dataset(tmp_path):
    (tmp_path/'data').mkdir(); hashes = {}
    for split in ('train', 'valid', 'test'):
        rel = f'wavs/speakers/{split}/audio.wav'; audio = tmp_path/rel
        audio.parent.mkdir(parents=True); audio.write_bytes(split.encode())
        row = dict(path=rel, speakerId=split, transcription='NEVER EXPORTED',
                   **config()['commands'][0]['values'])
        path = tmp_path/'data'/f'{split}_data.csv'
        with path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        hashes[split] = file_hash(path)
    return hashes


def test_roster_excludes_targets_and_scoring_recovers_exact_official_labels(tmp_path):
    hashes = dataset(tmp_path); result = build_population(tmp_path, 'test', hashes)
    assert result['recordings'] == 1
    assert set(result['records'][0]) == {'id', 'audio', 'audio_sha256', 'speaker_id'}
    assert 'NEVER EXPORTED' not in str(result)
    assert scoring_targets(result, config())[0]['gold'] == config()['commands'][0]['values']


@pytest.mark.parametrize('bad', ['csv', 'speaker', 'path', 'audio', 'count', 'missing', 'role'])
def test_scoring_rejects_population_changes(tmp_path, bad):
    hashes = dataset(tmp_path); result = build_population(tmp_path, 'test', hashes)
    if bad == 'csv': (tmp_path/'data/test_data.csv').write_text('changed')
    if bad == 'speaker': result['records'][0]['speaker_id'] = 'other'
    if bad == 'path': result['records'][0]['audio'] = '/other'
    if bad == 'audio': Path(result['records'][0]['audio']).write_bytes(b'changed')
    if bad == 'count': result['recordings'] += 1
    if bad == 'missing': result['records'] = []
    if bad == 'role': result['role'] = 'engineering_train_inputs'
    with pytest.raises(ValueError): scoring_targets(result, config())


@pytest.mark.parametrize('bad', ['overlap', 'escape', 'duplicate', 'speaker_path'])
def test_rejects_invalid_official_metadata(tmp_path, bad):
    hashes = dataset(tmp_path); path = tmp_path/'data/test_data.csv'
    rows = list(csv.DictReader(path.open())); row = rows[0]
    if bad == 'overlap': row.update(path='wavs/speakers/train/other.wav', speakerId='train')
    if bad == 'escape': row['path'] = '../escape.wav'
    if bad == 'duplicate': rows.append(copy.deepcopy(row))
    if bad == 'speaker_path': row['speakerId'] = 'wrong'
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
    hashes['test'] = file_hash(path)
    with pytest.raises(ValueError): build_population(tmp_path, 'test', hashes)
