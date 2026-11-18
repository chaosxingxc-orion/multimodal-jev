import json
from pathlib import Path

import pytest

from tools.jev.prepare_context_checkpoint_package import prepare, sha
from tools.jev.prepare_context_checkpoint_runtime import transform


@pytest.fixture
def package(tmp_path):
    source = tmp_path / 'source'
    trainer = source / 'source/tools/jev/train_context_online_candidate.py'
    trainer.parent.mkdir(parents=True)
    trainer.write_text(Path('tools/jev/train_context_online_candidate.py').read_text())
    remote = '/data/xingchao/multimodal-jev/simple-final-v1/original'
    pins = {remote + '/source/tools/jev/train_context_online_candidate.py': sha(trainer)}
    config = dict(start_update=0, stop_update=2, engineering_prefix=True,
                  training_dispatch_allowed=False, checkpoint_interval=1, algorithm='grpo_v1',
                  pins=pins, source_policy='/data/reference', source_policy_id='fixed',
                  reference_cache=remote + '/reference-cache', model='/mnt/cephfs/model')
    for name in ('config.json', 'service-config.json'):
        (source / name).write_text(json.dumps(config))
    (source / 'run.py').write_text('unchanged runner')
    files = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    (source / 'PREPARED.json').write_text(json.dumps(dict(remote_root=remote, files=files)))
    candidate = tmp_path / 'candidate.py';candidate.write_text(transform(trainer.read_text()))
    return dict(source=source, source_manifest_sha256=sha(source / 'PREPARED.json'),
                candidate=candidate, candidate_sha256=sha(candidate), output=tmp_path / 'out',
                remote_root='/data/xingchao/multimodal-jev/simple-final-v1/interval-candidate')


def test_package_changes_only_interval_trainer_and_local_paths(package):
    source = package['source']
    before = {str(p): sha(p) for p in source.rglob('*') if p.is_file()}
    result = prepare(**package)
    out = package['output']
    assert result['expected_recovery_updates'] == [2]
    assert result['expected_policy_exports'] == [1, 2]
    assert all(sha(Path(p)) == h for p, h in before.items())
    old = json.loads((source / 'config.json').read_text())
    new = json.loads((out / 'config.json').read_text())
    assert new['checkpoint_interval'] == 64
    for key in ('start_update', 'stop_update', 'engineering_prefix', 'training_dispatch_allowed',
                'source_policy', 'source_policy_id', 'algorithm', 'model'):
        assert old[key] == new[key]
    assert (out / 'run.py').read_bytes() == (source / 'run.py').read_bytes()
    assert all(sha(out / p) == h for p, h in result['files'].items())
    assert (out / 'source/tools/jev/train_context_online_candidate.py').read_bytes() == package['candidate'].read_bytes()
    assert json.loads((out / 'service-config.json').read_text())['pins'] == new['pins']


@pytest.mark.parametrize('mutation', ['manifest', 'source', 'extra', 'candidate', 'forged_candidate',
                                     'same_remote', 'escape_remote', 'source_output'])
def test_reject_invalid_package_before_writing(package, mutation):
    if mutation == 'manifest':package['source_manifest_sha256'] = 'wrong'
    elif mutation == 'source':(package['source'] / 'run.py').write_text('changed')
    elif mutation == 'extra':(package['source'] / 'extra').write_text('extra')
    elif mutation in ('candidate', 'forged_candidate'):
        package['candidate'].write_text(package['candidate'].read_text() + '\n# unrelated change\n')
        if mutation == 'forged_candidate':package['candidate_sha256'] = sha(package['candidate'])
    elif mutation == 'same_remote':package['remote_root'] = '/data/xingchao/multimodal-jev/simple-final-v1/original'
    elif mutation == 'escape_remote':package['remote_root'] = '/data/xingchao/multimodal-jev/simple-final-v1/../other'
    elif mutation == 'source_output':package['output'] = package['source'] / 'nested'
    with pytest.raises(ValueError):prepare(**package)
    assert not package['output'].exists()


@pytest.mark.parametrize('change', ['missing_trainer_pin', 'formal_dispatch', 'changed_interval'])
def test_rehashed_source_cannot_bypass_candidate_contract(package, change):
    path = package['source'] / 'config.json'
    config = json.loads(path.read_text())
    if change == 'missing_trainer_pin':config['pins'] = {}
    elif change == 'formal_dispatch':config['training_dispatch_allowed'] = True
    elif change == 'changed_interval':config['checkpoint_interval'] = 64
    path.write_text(json.dumps(config))
    mp = package['source'] / 'PREPARED.json'
    manifest = json.loads(mp.read_text());manifest['files']['config.json'] = sha(path)
    mp.write_text(json.dumps(manifest));package['source_manifest_sha256'] = sha(mp)
    with pytest.raises(ValueError):prepare(**package)
    assert not package['output'].exists()
