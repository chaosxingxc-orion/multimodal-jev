"""Runtime binding to a published formal RL package; does not publish one.

The publisher must independently revalidate NPU evidence. This runtime check
protects package/config/source identity, not the truth of research claims.
"""
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(c):
    fixed = dict(stage='task', engineering_prefix=False, training_dispatch_allowed=True,
                 start_update=0, stop_update=634, checkpoint_interval=64,
                 global_batch=128, microbatch=8, physical_devices=[4,5,6,7],
                 runtime_profile='qwen3-asr-context-bf16-packed-row-v1')
    if any(type(c.get(k)) is not type(v) or c[k] != v for k,v in fixed.items()):
        raise ValueError('formal full634 frozen configuration required')
    if c['algorithm'] not in ('grpo_v1', 'rloo_v1') or c.get('resume_root'):
        raise ValueError('registered algorithm and fresh task optimizer required')
    path = Path(c['formal_release'])
    if path.is_symlink() or sha(path) != c['formal_release_sha256'] or c['pins'].get(str(path)) != sha(path):
        raise ValueError('pinned formal release required')
    release = json.loads(path.read_text());root = path.parent
    payload = {k:v for k,v in release.items() if k != 'release_id'}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (release.get('schema') != 'jev-context-formal-rl-release-v1'
            or release.get('release_id') != identity or release.get('status') != 'released'
            or release.get('image') != 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'):
        raise ValueError('published fixed-image release required')
    for name, digest in release['files'].items():
        rel = Path(name);asset = root / rel
        if (rel.is_absolute() or '..' in rel.parts or asset.is_symlink()
                or not asset.resolve().is_relative_to(root.resolve()) or sha(asset) != digest):
            raise ValueError('formal release asset differs: ' + name)
    for name in ('continuous', 'fresh_replay', 'checkpoint_frequency'):
        if release['acceptance'][name] not in release['files']:
            raise ValueError('bound acceptance evidence required')
    if (release['source']['binding_file'] not in release['files']
            or json.loads((root / release['source']['binding_file']).read_text()) != release['source']['binding']):
        raise ValueError('full Stage1 binding evidence required')
    if release['config_contract'] != {k:v for k,v in c.items() if k not in ('pins','formal_release','formal_release_sha256')}:
        raise ValueError('executed config differs from formal release')
    source = release['source']
    if (source['source_step'] != 634 or source['reset_optimizer_for_task_stage'] is not True
            or source['policy_id'] != c['source_policy_id']
            or source['reference_policy_id'] != c['source_policy_id']
            or source['contract'] != c['source_contract']
            or c['initial_adapter'] != c['reference_adapter']):
        raise ValueError('common full Stage1 source and reference required')
    binding = source['binding']
    if (binding['source_step'] != 634 or binding['source_policy_id'] != source['policy_id']
            or binding['reference_policy_id'] != source['reference_policy_id']
            or binding['source_contract'] != source['contract']
            or binding['reset_optimizer_for_task_stage'] is not True):
        raise ValueError('Stage1 binding and source differ')
    for folder in ('source_policy', 'initial_adapter'):
        directory = Path(c[folder])
        if directory.is_symlink() or not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError('self-contained source assets required')
        for asset in directory.rglob('*'):
            if asset.is_symlink():raise ValueError('source symlink forbidden')
            if asset.is_file() and release['files'].get(str(asset.relative_to(root))) != sha(asset):
                raise ValueError('unbound Stage1 source asset')
    return dict(release_id=identity, scope='Runtime package identity; independent publication audits remain required')
