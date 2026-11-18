"""Audit collected interval checkpoints and every-update policy publication.

Read-only structural/hash audit, complementary to tensor, optimizer, sampling
and terminal-process audits. Does not authorize training or delete artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root, *, files, start_update, stop_update, checkpoint_interval):
    root = Path(root)
    if (any(type(x) is not int for x in (start_update, stop_update, checkpoint_interval))
            or not 0 <= start_update < stop_update <= 634 or checkpoint_interval < 1):
        raise ValueError('valid bounded update segment and checkpoint interval required')

    def bound(name):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('relative collected artifact required')
        path = root / relative
        if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
            raise ValueError('symlink artifacts forbidden')
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('artifact escapes collection')
        if files.get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('unbound or changed artifact: ' + name)
        return path

    def read(name):
        return json.loads(bound(name).read_text())

    expected = [step for step in range(start_update + 1, stop_update + 1)
                if step % checkpoint_interval == 0 or step == stop_update]
    # Inspect disk as well as the receipt: an unlisted checkpoint is a failure.
    for pattern, names in [
        ('checkpoint-*-committed.json', {f'checkpoint-{s:04d}-committed.json' for s in expected}),
        ('execution-*.pt', {f'execution-{s:04d}.pt' for s in expected}),
        *[(f'rank-{rank}/checkpoint-*.pt', {f'rank-{rank}/checkpoint-{s:04d}.pt' for s in expected})
          for rank in range(4)],
    ]:
        actual = {str(p.relative_to(root)) for p in root.glob(pattern)}
        if actual != names:
            raise ValueError('recovery checkpoint population differs: ' + pattern)

    policies = []
    for step in range(start_update + 1, stop_update + 1):
        receipt = read(f'policy-export-{step:04d}.json')
        saving = step in expected
        if receipt['update'] != step or receipt['recovery_checkpoint'] is not saving:
            raise ValueError('export update or recovery flag differs')
        directory = f'policy-{step:04d}'
        manifest_name = directory + '/MANIFEST.json'
        manifest = read(manifest_name)
        if (receipt['policy_manifest_sha256'] != files[manifest_name]
                or receipt['policy_id'] != manifest['policy_id']):
            raise ValueError('export policy binding differs')
        payload = {k: v for k, v in manifest.items() if k != 'policy_id'}
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                              allow_nan=False).encode()).hexdigest()
        if manifest['policy_id'] != identity:
            raise ValueError('policy identity differs from manifest content')
        actual_assets = {str(p.relative_to(root / directory)) for p in (root / directory).rglob('*')
                         if p.is_file() and p.name != 'MANIFEST.json'}
        if actual_assets != set(manifest['files']):
            raise ValueError('policy asset population differs')
        for name, digest in manifest['files'].items():
            artifact = directory + '/' + name
            bound(artifact)
            if files[artifact] != digest:
                raise ValueError('policy asset differs from manifest')
        activation = read(f'activation-{step:04d}.json')
        ack = activation['receipt']
        if (activation['error'] is not None or activation['policy_id'] != manifest['policy_id']
                or not isinstance(ack, dict) or ack.get('policy_id') != manifest['policy_id']
                or ack.get('head_marker_exact') is not True or ack.get('native_loaded_tensors') != 112):
            raise ValueError('successful matching engine activation required')
        for rank in range(4):
            update = read(f'rank-{rank}/update-{step:04d}.json')
            if (update['update'] != step or update['checkpoint'] is not saving
                    or update['next_policy_id'] != manifest['policy_id']):
                raise ValueError('rank publication record differs')
        if saving:
            commit = read(f'checkpoint-{step:04d}-committed.json')
            shards = {f'rank-{rank}/checkpoint-{step:04d}.pt' for rank in range(4)}
            if (commit['updates'] != step or set(commit['rank_shards']) != shards
                    or commit['execution_file'] != f'execution-{step:04d}.pt'
                    or commit['joint_policy_directory'] != directory
                    or commit['joint_policy']['policy_id'] != manifest['policy_id']
                    or commit['joint_policy_manifest_sha256'] != files[manifest_name]):
                raise ValueError('recovery commit differs from published policy')
            for name, digest in {**commit['rank_shards'],
                                 commit['execution_file']: commit['execution_sha256']}.items():
                bound(name)
                if files[name] != digest:
                    raise ValueError('recovery asset differs from commit')
        policies.append(manifest['policy_id'])
    return dict(passed=True, start_update=start_update, stop_update=stop_update,
                published_updates=len(policies), recovery_steps=expected,
                policy_ids=policies, training_dispatch_allowed=False,
                scope='Collected publication/checkpoint bindings only; tensor, Adam, sampling, '
                      'terminal-state and NPU acceptance remain independent')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--files', type=Path, required=True, help='Collected run-relative SHA256 mapping')
    p.add_argument('--start-update', type=int, required=True)
    p.add_argument('--stop-update', type=int, required=True)
    p.add_argument('--checkpoint-interval', type=int, required=True)
    a = vars(p.parse_args())
    a['files'] = json.loads(a['files'].read_text())
    print(json.dumps(audit(**a), indent=2))
