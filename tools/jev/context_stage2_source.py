"""Bind Stage2 initialization/reference to collected full Stage1 and selection.

No dispatch and no choice of checkpoint based on target data. Both algorithms
use instruction step634; its evaluated export must contain the same tensors.
"""
import json
from pathlib import Path

from tools.jev.context_rl_evaluation_plan import bound, sha


def bind(*, experiment, training_root, training_proof, selection_root, selection_proof):
    roots = [Path(training_root), Path(selection_root)]
    proofs = [training_proof, selection_proof]
    for proof in proofs:
        state = proof['state']
        if state['Status'] != 'exited' or state['ExitCode'] != 0 or state['OOMKilled']:
            raise ValueError('successfully collected terminal source required')
    for root, proof in zip(roots, proofs, strict=True):
        for name in proof['files']:
            bound(root, proof, name)
    def read(i, name):return json.loads(bound(roots[i], proofs[i], name).read_text())
    c, e = read(0, 'config.json'), read(1, 'config.json')
    step = experiment['instruction_stage']['updates']
    if (step != 634 or c['stage'] != 'instruction' or c['engineering_prefix'] is not False
            or c['training_dispatch_allowed'] is not True or c['stop_update'] != step
            or c['release_sha256'] != experiment['release_sha256']):
        raise ValueError('full clean Stage1 step634 required')
    result = read(0, 'run/result.json')
    if (result['passed'] is not True or result['stop_update'] != step
            or result['learning_population'] != experiment['learning_recordings']
            or result['schedule_updates'] != step or result['engineering_prefix'] is not False
            or training_proof.get('admitted_joint_policy_checkpoints') != 10
            or training_proof.get('checkpoint_steps') != [*range(64, 634, 64), 634]):
        raise ValueError('independent full schedule and checkpoint collection required')
    commit = read(0, f'run/checkpoint-{step:04d}-committed.json')
    directory = f'run/policy-{step:04d}'
    manifest_name = directory + '/MANIFEST.json'
    manifest = read(0, manifest_name)
    contract = manifest['contract']
    if (commit['updates'] != step or commit['contract']['stage'] != 'instruction'
            or commit['joint_policy_directory'] != f'policy-{step:04d}'
            or commit['joint_policy_manifest_sha256'] != training_proof['files'][manifest_name]
            or commit['joint_policy']['policy_id'] != manifest['policy_id']
            or contract != c['source_contract'] or contract['base_id'] != c['base_id']
            or contract['token']['registration_status'] != 'admitted'
            or contract['request_version'] != 'jev-context-exclusive_event-v1'):
        raise ValueError('full Stage1 commit and admitted policy binding differ')
    for asset, digest in manifest['files'].items():
        if sha(bound(roots[0], training_proof, directory + '/' + asset)) != digest:
            raise ValueError('Stage1 policy asset differs')
    shards = {f'rank-{rank}/checkpoint-{step:04d}.pt' for rank in range(4)}
    if set(commit['rank_shards']) != shards:
        raise ValueError('four Stage1 recovery shards required')
    for name, digest in {**commit['rank_shards'], commit['execution_file']: commit['execution_sha256']}.items():
        if sha(bound(roots[0], training_proof, 'run/' + name)) != digest:
            raise ValueError('Stage1 recovery/execution asset differs')
    if (e['checkpoint_step'] != step or e['source_policy_id'] != manifest['policy_id']
            or e['source_contract'] != contract or e['release_sha256'] != experiment['release_sha256']
            or selection_proof['step'] != step or selection_proof['source_policy_id'] != manifest['policy_id']
            or selection_proof.get('source_tensors_exact') is not True
            or selection_proof.get('parity_queries') != 256 or selection_proof.get('parity_max_error') != 0):
        raise ValueError('complete Stage1 endpoint selection/parity required')
    joint = read(1, 'run/joint.json')
    evaluated = read(1, 'run/policy/MANIFEST.json')
    if (joint['policy_id'] != evaluated['policy_id'] or joint['contract'] != contract
            or selection_proof['inference_policy_id'] != joint['policy_id']):
        raise ValueError('selection export identity differs')
    from tools.jev.run_context_rl_selection import validate_parity
    validate_parity(e, joint, read(1, 'run/hf.json'), read(1, 'run/vllm.json')['records'])
    for asset, digest in evaluated['files'].items():
        if sha(bound(roots[1], selection_proof, 'run/policy/' + asset)) != digest:
            raise ValueError('selection policy asset differs')
    # Rebind complete scoring records; trust neither a free-floating aggregate
    # nor its source_tensors_exact flag for actual weight authentication below.
    predictions = read(1, 'selection/predictions.json')['records']
    receipts = read(1, 'selection/input-receipts.json')
    if (len(predictions) != experiment['selection_recordings'] or len(receipts) != len(predictions)
            or len({p['id'] for p in predictions}) != len(predictions)):
        raise ValueError('complete unique selection population required')
    for p, receipt in zip(predictions, receipts, strict=True):
        if p['id'] != receipt['id'] or p['policy_id'] != joint['policy_id']:
            raise ValueError('selection population/policy mismatch')
    metrics = selection_proof['result']
    if (metrics['role'] != 'selection' or metrics['policy_id'] != joint['policy_id']
            or metrics['release_sha256'] != experiment['release_sha256']
            or metrics['predictions_sha256'] != selection_proof['files']['selection/predictions.json']):
        raise ValueError('selection metrics not bound to full endpoint predictions')
    for folder in ('run/policy', 'run/adapter'):
        for path in (roots[1] / folder).rglob('*'):
            if path.is_file():bound(roots[1], selection_proof, str(path.relative_to(roots[1])))
    authenticate(roots[0] / directory, roots[1] / 'run/policy', roots[1] / 'run/adapter',
                 manifest['policy_id'], joint['policy_id'], contract)
    return dict(status='bound_source_only_no_stage2_admission', source_step=step,
                source_policy=str(roots[0] / directory), source_policy_id=manifest['policy_id'],
                source_contract=contract, reference_policy_id=manifest['policy_id'],
                evaluated_policy_id=joint['policy_id'],
                selection_predictions_sha256=metrics['predictions_sha256'],
                stage1_commit_sha256=training_proof['files'][f'run/checkpoint-{step:04d}-committed.json'],
                algorithms=['grpo_v1', 'rloo_v1'], reset_optimizer_for_task_stage=True,
                training_dispatch_allowed=False,
                scope='Collected full Stage1 endpoint and evaluated tensors bound; '
                      'continuous runtime and formal release remain separate')


def authenticate(source, evaluated, adapter, source_id, evaluated_id, contract):
    import torch
    from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
    a = read_execution_bundle(source, expected_policy_id=source_id,
                              expected_contract=contract, expected_base_id=contract['base_id'])
    b = read_execution_bundle(evaluated, expected_policy_id=evaluated_id,
                              expected_contract=contract, expected_base_id=contract['base_id'])
    if a['tensors'].keys() != b['tensors'].keys() or any(
            not torch.equal(v, b['tensors'][k]) for k, v in a['tensors'].items()):
        raise ValueError('selection tensors differ from full Stage1 endpoint')
    af = json.loads((source / 'MANIFEST.json').read_text())['files']
    bf = json.loads((evaluated / 'MANIFEST.json').read_text())['files']
    if {k:v for k,v in af.items() if k.startswith('processor/')} != {
            k:v for k,v in bf.items() if k.startswith('processor/')}:
        raise ValueError('selection processor differs from full Stage1 endpoint')
    verify_native_adapter(adapter, b)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    for name in ('experiment', 'training_proof', 'selection_proof'):
        spec[name] = json.loads(Path(spec[name]).read_text())
    result = bind(**spec)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
