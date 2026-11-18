"""Prepare a two-update interval-checkpoint variant of the pinned shared package.

No remote calls or launch. Keeps the same source policy and training numerics
for a matched comparison against the original every-update checkpoint run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from tools.jev.prepare_context_checkpoint_runtime import transform


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(*, source, source_manifest_sha256, candidate, candidate_sha256,
            output, remote_root):
    source, candidate, output = map(Path, (source, candidate, output))
    if sha(source / 'PREPARED.json') != source_manifest_sha256:
        raise ValueError('source package manifest changed')
    manifest = json.loads((source / 'PREPARED.json').read_text())
    original_root = manifest['remote_root']
    parent = Path('/data/xingchao/multimodal-jev/simple-final-v1')
    destination = Path(remote_root)
    if (destination.parent != parent or '..' in destination.parts
            or str(destination) != remote_root or remote_root == original_root):
        raise ValueError('distinct direct child of hot-storage root required')
    if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError('source and output packages must be independent')
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('source package symlinks forbidden')
    actual = {str(p.relative_to(source)): sha(p) for p in source.rglob('*')
              if p.is_file() and p != source / 'PREPARED.json'}
    if actual != manifest['files']:
        raise ValueError('source package population or hashes changed')
    trainer = 'source/tools/jev/train_context_online_candidate.py'
    if sha(candidate) != candidate_sha256 or candidate.read_text() != transform((source / trainer).read_text()):
        raise ValueError('candidate differs from exact checkpoint-only transformation')
    config = json.loads((source / 'config.json').read_text())
    service = json.loads((source / 'service-config.json').read_text())
    if (config['start_update'] != 0 or config['stop_update'] != 2
            or config['engineering_prefix'] is not True or config['training_dispatch_allowed'] is not False
            or config['checkpoint_interval'] != 1 or config['algorithm'] != 'grpo_v1'):
        raise ValueError('original two-update every-step GRPO candidate required')
    if output.exists():
        raise FileExistsError(output)

    def relocate(value):
        if isinstance(value, str):
            return value.replace(original_root, remote_root)
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, dict):
            return {relocate(k): relocate(v) for k, v in value.items()}
        return value

    config, service = relocate(config), relocate(service)
    config['checkpoint_interval'] = 64
    changed_remote = remote_root + '/' + trainer
    if changed_remote not in config['pins'] or changed_remote not in service['pins']:
        raise ValueError('trainer must have an existing source pin')
    config['pins'][changed_remote] = candidate_sha256
    service['pins'][changed_remote] = candidate_sha256
    # Validate source pins and destination before creating output.
    shutil.copytree(source, output)
    shutil.copyfile(candidate, output / trainer)
    for name, value in [('config.json', config), ('service-config.json', service)]:
        (output / name).write_text(json.dumps(value, indent=2) + '\n')
    result = dict(status='prepared_only_no_dispatch', remote_root=remote_root,
                  source_package=str(source), source_manifest_sha256=source_manifest_sha256,
                  trainer_sha256=candidate_sha256, checkpoint_interval=64,
                  expected_recovery_updates=[2], expected_policy_exports=[1, 2],
                  comparison='same start, seed, two updates, algorithm and policy; '
                             'compare step2 execution/master/Adam with original interval1 run',
                  required_before_launch=['original_shared_continuous_and_fresh_replay_passed',
                                          'versioned_launcher_and_independent_collector',
                                          'original_shared_device_lease'],
                  required_before_release=['actual_NPU_interval_candidate_success',
                                           'step2_exact_comparison_against_original_interval1',
                                           'interval_lifecycle_and_existing_sampling_tensor_optimizer_audits'],
                  files={str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                         if p.is_file() and p != output / 'PREPARED.json'})
    (output / 'PREPARED.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'source-manifest-sha256', 'candidate', 'candidate-sha256', 'output', 'remote-root'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))
