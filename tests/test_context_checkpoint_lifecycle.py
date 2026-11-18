import hashlib
import json
from pathlib import Path
import runpy

import pytest

from tools.jev.audit_context_checkpoint_lifecycle import audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value))


def change(path, **values):
    data = json.loads(path.read_text());data.update(values);write(path, data)


@pytest.fixture
def run(tmp_path):
    # Execute the actual transformed publication block; add simulated engine
    # receipts. This checks disk lifecycle, not NPU tensor authentication.
    execute = runpy.run_path('tests/test_context_checkpoint_runtime.py')['execute_segment']
    root = tmp_path / 'run'
    execute(root, 62, 66)
    for step in range(63, 67):
        folder = root / f'policy-{step:04d}'
        payload = dict(schema='context-policy-execution-v1', contract={'description': '语音'},
                       files={'weights.pt': sha(folder / 'weights.pt')})
        pid = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode()).hexdigest()
        write(folder / 'MANIFEST.json', dict(payload, policy_id=pid))
        change(root / f'policy-export-{step:04d}.json', policy_id=pid,
               policy_manifest_sha256=sha(folder / 'MANIFEST.json'))
        commit = root / f'checkpoint-{step:04d}-committed.json'
        if commit.exists():
            change(commit, joint_policy={'policy_id': pid},
                   joint_policy_manifest_sha256=sha(folder / 'MANIFEST.json'))
        write(root / f'activation-{step:04d}.json', dict(error=None, policy_id=pid,
              receipt=dict(policy_id=pid, head_marker_exact=True, native_loaded_tensors=112)))
        for rank in range(4):
            write(root / f'rank-{rank}/update-{step:04d}.json',
                  dict(update=step, checkpoint=step in (64, 66), next_policy_id=pid))
    return root


def verify(root):
    return audit(root, files={str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()},
                 start_update=62, stop_update=66, checkpoint_interval=64)


def test_interval_and_terminal_with_every_step_publication(run):
    result = verify(run)
    assert result['recovery_steps'] == [64, 66]
    assert result['published_updates'] == 4
    assert result['training_dispatch_allowed'] is False


@pytest.mark.parametrize('mutation', [
    'missing_export', 'stale_activation', 'failed_activation', 'missing_shard',
    'unplanned_checkpoint', 'wrong_rank_flag', 'wrong_manifest_hash',
    'changed_policy_asset', 'bad_commit_shard', 'extra_policy_asset', 'bad_policy_identity',
])
def test_reject_incomplete_or_mixed_publication(run, mutation):
    if mutation == 'missing_export':
        (run / 'policy-export-0063.json').unlink()
    elif mutation == 'stale_activation':
        change(run / 'activation-0065.json', policy_id='stale')
    elif mutation == 'failed_activation':
        change(run / 'activation-0065.json', error='failed')
    elif mutation == 'missing_shard':
        (run / 'rank-3/checkpoint-0064.pt').unlink()
    elif mutation == 'unplanned_checkpoint':
        (run / 'rank-0/checkpoint-0063.pt').write_bytes(b'unexpected')
    elif mutation == 'wrong_rank_flag':
        change(run / 'rank-2/update-0065.json', checkpoint=True)
    elif mutation == 'wrong_manifest_hash':
        change(run / 'policy-export-0065.json', policy_manifest_sha256='wrong')
    elif mutation == 'changed_policy_asset':
        (run / 'policy-0065/weights.pt').write_bytes(b'changed')
    elif mutation == 'bad_commit_shard':
        path = run / 'checkpoint-0064-committed.json'
        data = json.loads(path.read_text());data['rank_shards']['rank-0/checkpoint-0064.pt'] = 'wrong'
        write(path, data)
    elif mutation == 'extra_policy_asset':
        (run / 'policy-0063/extra').write_bytes(b'unlisted')
    elif mutation == 'bad_policy_identity':
        path = run / 'policy-0063/MANIFEST.json'
        change(path, contract={'changed': True})
        change(run / 'policy-export-0063.json', policy_manifest_sha256=sha(path))
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(run)


def test_transport_hash_not_trusted_after_change(run):
    files = {str(p.relative_to(run)): sha(p) for p in run.rglob('*') if p.is_file()}
    change(run / 'activation-0063.json', error='changed after collection')
    with pytest.raises(ValueError, match='changed artifact'):
        audit(run, files=files, start_update=62, stop_update=66, checkpoint_interval=64)


def test_symlink_rejected(run, tmp_path):
    path = run / 'policy-0063/weights.pt'
    target = tmp_path / 'external'
    path.rename(target);path.symlink_to(target)
    with pytest.raises(ValueError, match='symlink'):
        verify(run)
