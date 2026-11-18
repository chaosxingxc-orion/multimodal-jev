import json

import pytest

from tools.reports import committed_signal_report as report


def fixture(tmp_path):
    collection = tmp_path / 'collected'
    files = {}
    for rank in range(4):
        name = f'run/rank-{rank}/update-0512.json'
        path = collection / name
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(update=512, rollout={'update': 511},
            preparation_seconds=1, actor_seconds=2, total_seconds=3,
            input_service={}, metrics={})))
        files[name] = report.sha(path)
    proof = tmp_path / 'proof.json'
    proof.write_text(json.dumps(dict(passed=True, step=512, collection=str(collection),
        files=files, container_id='fixture', image='fixture')))
    config = dict(schema='committed-signal-report-v1', step=512, proof=str(proof),
        proof_sha256=report.sha(proof), collection=str(collection), release='fixture',
        schedule='fixture', receipts='fixture', cache_proof='fixture',
        output_directory=str(tmp_path / 'output'))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    return path, config


def fake_compute(observation, config):
    common = dict(actor_update=512, behavior_update=511,
                  sources={str(observation): report.sha(observation)})
    return (dict(common, summaries={}, paired_exact_changes={}),
            dict(common, summary={}))


def test_completed_run_is_idempotent_and_tampered_output_rejected(tmp_path, monkeypatch):
    path, config = fixture(tmp_path)
    monkeypatch.setattr(report, 'check_paths', lambda *a: [])
    monkeypatch.setattr(report, 'compute', fake_compute)
    assert report.run(path)['status'] == 'complete'
    monkeypatch.setattr(report, 'compute', lambda *a: pytest.fail('duplicate compute'))
    assert report.run(path)['status'] == 'already_complete'
    output = tmp_path / 'output'
    (output / 'summary.json').write_text('{}')
    with pytest.raises(ValueError, match='Source changed'):
        report.run(path)


def test_changed_source_rejected_and_failure_not_automatically_retried(tmp_path, monkeypatch):
    path, config = fixture(tmp_path)
    monkeypatch.setattr(report, 'check_paths', lambda *a: [])
    monkeypatch.setattr(report, 'compute', lambda *a: pytest.fail('untrusted input'))
    rank = tmp_path / 'collected/run/rank-3/update-0512.json'
    rank.write_text('{}')
    with pytest.raises(ValueError, match='Committed rank record changed'):
        report.run(path)
    assert not (tmp_path / 'output/COMPLETE.json').exists()
    assert report.read(tmp_path / 'output/status.json')['phase'] == 'failed_preserve_evidence'
    with pytest.raises(ValueError, match='Incomplete run preserved'):
        report.run(path)


def test_source_change_during_analysis_prevents_publication(tmp_path, monkeypatch):
    path, config = fixture(tmp_path)
    monkeypatch.setattr(report, 'check_paths', lambda *a: [])
    def changing_compute(observation, spec):
        result = fake_compute(observation, spec)
        path.write_text('{}')
        return result
    monkeypatch.setattr(report, 'compute', changing_compute)
    with pytest.raises(ValueError, match='Source changed'):
        report.run(path)
    assert not (tmp_path / 'output/COMPLETE.json').exists()


def test_remote_storage_rejected_before_writes(tmp_path, monkeypatch):
    path, config = fixture(tmp_path)
    def reject(*args):
        raise ValueError('hot writes require local storage')
    monkeypatch.setattr(report, 'check_paths', reject)
    with pytest.raises(ValueError, match='local storage'):
        report.run(path)
    assert not (tmp_path / 'output').exists()


def test_existing_owner_blocks_duplicate_run(tmp_path, monkeypatch):
    import fcntl
    path, config = fixture(tmp_path)
    monkeypatch.setattr(report, 'check_paths', lambda *a: [])
    output = tmp_path / 'output'
    output.mkdir()
    with (output / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            report.run(path)
    assert not (output / 'status.json').exists()


@pytest.mark.parametrize('mutation', ['step', 'proof', 'collection'])
def test_proof_contract_rejected(tmp_path, mutation):
    _, config = fixture(tmp_path)
    if mutation == 'step':
        config['step'] = 576
    elif mutation == 'proof':
        config['proof_sha256'] = '0' * 64
    else:
        config['collection'] = str(tmp_path)
    with pytest.raises(ValueError):
        report.load_observation(config)
