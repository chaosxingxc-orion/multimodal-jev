#!/usr/bin/env python3
"""Version the bounded RL candidate to separate rollout exports from recovery.

Does not change the original trainer, remove artifacts, authorize formal RL or
launch a workload. The engineering-only guard stays intact until acceptance.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(source):
    before = source
    old = '                save_now=True\n'
    new = "                save_now=((index+1)%c['checkpoint_interval']==0 or index+1==stop_update)\n"
    if source.count(old) != 1:
        raise ValueError('unexpected checkpoint decision source')
    source = source.replace(old, new)
    start = source.index('                    with FSDP.summon_full_params', source.index(new))
    end = source.index('                    dist.barrier()\n', start)
    export = source[start:end]
    # Every update exports and activates a policy; optimizer checkpoint saving
    # alone follows the interval. Each collective runs on every rank.
    dedented = '\n'.join(line[4:] if line else line for line in export.split('\n'))
    old_execution = "                        commit_file(root/f'execution-{index+1:04d}.pt',lambda f:torch.save(full,f))\n"
    if dedented.count(old_execution) != 1:
        raise ValueError('unexpected execution export source')
    dedented = dedented.replace(old_execution,
        "                        if save_now:\n    " + old_execution)
    # Existing checkpoint commit follows the export barrier. Restore its
    # conditional indentation without changing any commit fields.
    source = source[:start] + dedented + '                dist.barrier()\n                if save_now:\n' + source[end+len('                    dist.barrier()\n'):]
    # Keep an authenticated export receipt for every policy, including steps
    # without an Adam checkpoint. It remains available for independent audits.
    marker = "                activation=[None]\n"
    receipt = """                if rank==0:
                    commit_json(root/f'policy-export-{index+1:04d}.json',dict(
                        update=index+1,policy_id=joint['policy_id'],
                        policy_manifest_sha256=file_hash(root/f'policy-{index+1:04d}/MANIFEST.json'),
                        recovery_checkpoint=save_now))
                dist.barrier()
"""
    if source.count(marker) != 1:raise ValueError('unexpected activation boundary')
    source = source.replace(marker, receipt + marker)
    # Native actor updates, losses, sampling and engine activation are unchanged.
    ast.parse(source)
    unchanged_guard = "if c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False:"
    if unchanged_guard not in before or unchanged_guard not in source:
        raise ValueError('bounded engineering admission must remain')
    return source


def prepare(*, source, source_sha256, output):
    source, output = Path(source), Path(output)
    if sha(source) != source_sha256:
        raise ValueError('original accepted candidate source changed')
    value = transform(source.read_text())
    output.mkdir(exist_ok=False)
    target = output / 'train_context_online_candidate.py'
    target.write_text(value)
    result = dict(status='bounded_checkpoint_candidate_no_dispatch', source=str(source),
        source_sha256=source_sha256, files={target.name: sha(target)},
        changes=['full_optimizer_checkpoint_on_interval_or_segment_end',
                 'joint_policy_export_and_activation_every_update',
                 'per_step_policy_export_receipt'],
        retained=['engineering_only_two_update_guard', 'fixed_reference', 'native_actor_loss_and_update',
                  'all_existing_artifacts_no_deletion'],
        pending=['actual_continuous_and_replay_acceptance', 'checkpoint_variant_NPU_acceptance',
                 'formal_runtime_release', 'owned_transient_policy_retention'])
    (output / 'PREPARED.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(**vars(a)), indent=2))
