from collections import Counter
import json
from pathlib import Path

import pytest
import torch

from multimodal_jev import local_checkpoint
from multimodal_jev.baseline import file_hash
from multimodal_jev.tpo_anchor_layout import part_name, partition
from tools.reports import joint_tpo_window, prepare_full_joint_tpo, prepare_tpo_windows
from tools.reports.train_joint_tpo import load_anchor_parts


@pytest.mark.parametrize('updates', [2, 212])
def test_each_of_eight_workers_has_distinct_work(updates):
    assignments = partition(6, updates, 8)
    positions = [[(rank, step) for rank, start, stop in group for step in range(start, stop)]
                 for group in assignments]
    assert len(positions) == 8 and min(map(len, positions)) > 0
    assert max(map(len, positions)) - min(map(len, positions)) <= 1
    assert Counter(pair for group in positions for pair in group) == Counter(
        (rank, step) for rank in range(6) for step in range(updates))


def test_training_reassembles_verified_parts(tmp_path):
    root = tmp_path / 'anchor'
    cards = [1, 3, 5, 7, 0, 2, 4, 6]
    plan = dict(training_devices=list(range(6)), inference_devices=[6, 7],
                offline_reference_devices=cards, source_policy='initial', target_sha='population',
                global_batch=6, updates=4)
    joint = 'joint'
    receipts = []
    for replica, assignment in enumerate(partition(6, 4, 8)):
        shard = root / f'reference-{replica}'
        shard.mkdir(parents=True)
        parts = {}
        for rank, first, last in assignment:
            name = part_name(rank, first, last)
            torch.save([[torch.tensor((rank, step))] for step in range(first, last)], shard / name)
            parts[name] = file_hash(shard / name)
        receipts.append(dict(card=cards[replica], source_policy='initial', target_sha='population',
                             joint_sha=joint, updates=4, parts=parts))
    seal = dict(passed=True, offline_reference_devices=cards, binding={'joint_sha': joint},
                updates=4, replicas=receipts)
    manifest = root / 'receipt.json'
    manifest.write_text(json.dumps(seal))
    rank = 2  # This rank's four steps span more than one worker.
    assert [row[0].tolist() for row in load_anchor_parts(root, plan, joint, rank, 4)] == [
        [rank, step] for step in range(4)]
    name = next(iter(receipts[0]['parts']))
    sha = receipts[0]['parts'].pop(name)
    manifest.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match='incomplete fixed-reference shard'):
        load_anchor_parts(root, plan, joint, rank, 4)
    receipts[0]['parts'][name] = sha
    manifest.write_text(json.dumps(seal))
    (root / 'reference-0' / name).write_bytes(b'changed')
    with pytest.raises(ValueError, match='content identity changed'):
        load_anchor_parts(root, plan, joint, 0, 4)


def test_frozen_roles_dispatch_eight_isolated_reference_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda paths: None)
    repo = Path(__file__).resolve().parents[1]
    package = tmp_path / 'package'
    parent = dict(physical_devices=list(range(6)), view_execution={
        'container': dict(cpus=24, memory_gib=128, shm_gib=16)})
    plan = dict(training_devices=list(range(6)), inference_devices=[6, 7], updates=212,
                source_policy='initial', target_sha='targets')
    parent_path = tmp_path / 'parent.json'
    parent_path.write_text(json.dumps(parent))

    def fake_publish(c, p, entry, repo):
        for name in ('parent.json', 'initial/policy/MANIFEST.json', 'targets/MANIFEST.json'):
            path = entry / name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(parent_path.read_bytes())
        return p, dict(plan, parent_sha=file_hash(parent_path))

    monkeypatch.setattr(prepare_full_joint_tpo, 'publish', fake_publish)
    monkeypatch.setattr(prepare_tpo_windows, 'freeze', lambda *args, **kwargs: {'manifest_sha256': 'source'})
    prepare_tpo_windows.prepare(dict(
        package=str(package), repository=str(repo), parent_config=str(parent_path),
        full_joint=dict(runtime_sha256=file_hash(parent_path), **plan), offline_reference_devices=list(range(8)),
        source_patterns=['src/multimodal_jev/*.py'], deployment={}, io_workers=2))
    entry = package / 'entry'
    operations = json.loads((package / 'operations-local.json').read_bytes())['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    groups = operations['chains'][0]
    assert [group[0]['name'] for group in groups] == ['finite-anchor', 'finite', 'anchor', 'tpo']
    assert all(group[0]['devices'] == list(range(8)) for group in groups)
    frozen = json.loads((entry / 'joint.json').read_bytes())
    assert (frozen['training_devices'], frozen['inference_devices']) == (list(range(6)), [6, 7])

    base_tmp = tmp_path / 'runtime'
    base_tmp.mkdir()
    monkeypatch.setenv('TMPDIR', str(base_tmp))
    observed = {}

    def worker(argv, *, env, check):
        replica = int(argv[argv.index('--replica') + 1])
        out = Path(argv[argv.index('--output') + 1])
        out.mkdir()
        observed[replica] = (env['ASCEND_RT_VISIBLE_DEVICES'], env['TMPDIR'])
        parts = {}
        for part in partition(6, 2, 8)[replica]:
            name = part_name(*part)
            (out / name).write_bytes(b'test part')
            parts[name] = file_hash(out / name)
        (out / 'receipt.json').write_text(json.dumps(dict(
            card=replica, updates=2, source_policy='initial', target_sha='targets',
            joint_sha=file_hash(entry / 'joint.json'), parts=parts)))

    monkeypatch.setattr(joint_tpo_window.subprocess, 'run', worker)
    options = json.loads((entry / 'finite-anchor.json').read_bytes())['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    options['training'].update(parent=str(entry / 'parent.json'), plan=str(entry / 'joint.json'),
                               output=str(tmp_path / 'finite-anchor'))
    receipt = joint_tpo_window.run(options)
    assert observed == {i: (str(i), str(base_tmp / f'anchor_{i}')) for i in range(8)}
    assert receipt['passed'] and len(receipt['replicas']) == 8
