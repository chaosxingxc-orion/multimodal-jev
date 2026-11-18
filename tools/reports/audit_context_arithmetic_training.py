"""Read-only full-schedule RL audit; a transport collector must run first.

Consumes exact terminal artifacts. Reuses schedule, sample, policy, checkpoint
and optimizer checks. Does not dispatch training or infer efficacy from reward.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from tools.jev.context_rl_evaluation_plan import bound
from tools.reports.context_probability_policy_v2 import PROFILE
from tools.jev.context_rl_group_audit import audit as audit_groups
from tools.jev.audit_context_checkpoint_lifecycle import audit as audit_lifecycle


def update_targets(batch, receipt, planned, population, task, *, update, policy_id, reference_id):
    from multimodal_jev.context_candidate_task import compile_candidates, bind_candidate_targets
    if (batch['policy_id'] != policy_id or receipt['behavior_policy_id'] != policy_id
            or receipt['fixed_reference_policy_id'] != reference_id or receipt['update'] != update
            or batch['rank_indices'] != planned['rank_indices']
            or len(batch['records']) != len(planned['slots']) or len(batch['records']) != 128
            or len(receipt['queries']) != 128):
        raise ValueError('complete scheduled batch/current behavior/fixed reference required')
    candidates = [dict(id=c['id'], definition=c['description']) for c in task['schema']['candidates']]
    targets = {}
    for i, (slot, row) in enumerate(zip(planned['slots'], batch['records'], strict=True)):
        source = population[slot['id']]
        uid = f'{source["id"]}::epoch{planned["epoch"]}-batch{planned["index"]}-slot{i}'
        request = compile_candidates(source['audio'], candidates, task['thinking']['requirements'], semantics='exclusive_event')
        target = bind_candidate_targets(request, semantics='exclusive_event', exclusive_event=source['intent'])
        if row['id'] != slot['id'] or row['request_id'] != uid or row['weight'] != slot['weight'] or row['target'] != target:
            raise ValueError('learning target, slot identity or weight differs')
        targets[uid] = dict(labels=target['labels'], real_request=not slot['padding'], weight=slot['weight'])
    return targets


def verify_checkpoint(run, step, contract, bundle):
    from multimodal_jev.context_training_checkpoint import validate
    full = torch.load(run / f'execution-{step:04d}.pt', map_location='cpu', weights_only=True)
    if (len(full) != 128 or set(bundle['tensors']) != set(full) | {'decision_token_row'}
            or any(v.dtype != torch.bfloat16 or not torch.equal(v, bundle['tensors'][k]) for k,v in full.items())):
        raise ValueError('assembled checkpoint differs from published execution')
    for rank in range(4):
        state = torch.load(run / f'rank-{rank}/checkpoint-{step:04d}.pt', map_location='cpu', weights_only=True)
        named = [(n, torch.nn.Parameter(v.clone())) for n,v in state['parameters'].items()]
        opt = torch.optim.AdamW([p for _,p in named], lr=1e-4, betas=(.9,.999), eps=1e-8,
                               weight_decay=0., foreach=False, fused=False)
        validate(named, opt, state, rank=rank, world_size=4, completed_updates=step, contract=contract)


def audit(*, collection, transport, launch, experiment, schedule, initial_policy):
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.context_training_schedule import load_schedule
    from multimodal_jev.context_policy_artifacts import read_execution_bundle, verify_native_adapter
    root = Path(collection);state = transport['state']
    image = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
    if (state['Status'] != 'exited' or state['ExitCode'] != 0 or state['OOMKilled']
            or transport['container_id'] != launch['container_id'] or transport['image'] != launch['image']
            or launch['image'] != image):
        raise ValueError('successful exact fixed-image terminal container required')
    if (transport.get('status') != 'transport_complete_independent_audit_required'
            or transport.get('bad_pins') != [] or transport.get('bad_package') != []
            or transport.get('launch_binding_mismatches') != []):
        raise ValueError('completed unchanged-package transport required')
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('collection symlink forbidden')
        if path.is_file():bound(root, transport, str(path.relative_to(root)))
    for name in transport['files']:bound(root, transport, name)
    def read(name):return json.loads(bound(root, transport, name).read_text())
    c = read('config.json')
    if c.get('binary_policy_arithmetic') != PROFILE:
        raise ValueError('repaired formal arithmetic required')
    if transport['files']['config.json'] != launch['config_sha256']:
        raise ValueError('executed config differs from launch')
    if (c['stage'] != 'task' or c['algorithm'] not in ('grpo_v1','rloo_v1')
            or c['engineering_prefix'] is not False or c['training_dispatch_allowed'] is not True
            or c['start_update'] != 0 or c['stop_update'] != 634 or c['checkpoint_interval'] != 64
            or c['global_batch'] != 128 or c['microbatch'] != 8 or c['physical_devices'] != [4,5,6,7]
            or c['release_sha256'] != experiment['release_sha256']):
        raise ValueError('fixed formal full634 task configuration required')
    terminal = read('supervision/terminal.json')
    if (terminal['passed'] is not True or terminal['actor_started'] is not True or terminal['forced_cleanup']
            or terminal['actor_exit_code'] != 0 or terminal['service_exit_code'] != 0
            or terminal['child_exit_codes'] != [0,0] or read('service/terminal.json')['exit_code'] != 0):
        raise ValueError('both supervised processes must complete normally')
    result = read('run/result.json')
    if any(result.get(k) != v for k,v in dict(passed=True, start_update=0, stop_update=634,
                  schedule_updates=634, learning_population=40513, engineering_prefix=False).items()):
        raise ValueError('full training result differs')
    initial = read_execution_bundle(initial_policy, expected_policy_id=c['source_policy_id'],
                    expected_contract=c['source_contract'], expected_base_id=c['base_id'])
    current = read('service/ready.json')['policy_id']
    if current != c['source_policy_id']:raise ValueError('initial behavior differs from fixed source')
    rows, task, _ = load(c['release'], c['release_sha256'], 'learning', purpose='optimize')
    population = {r['id']:r for r in rows}
    if len(population) != 40513:raise ValueError('full clean learning population required')
    batches, _ = load_schedule(schedule, c['schedule_sha256'], release_sha256=c['release_sha256'], population=population)
    if len(batches) != 634:raise ValueError('complete two-epoch schedule required')
    lifecycle = audit_lifecycle(root/'run', files={p[4:]:h for p,h in transport['files'].items() if p.startswith('run/')},
                                start_update=0, stop_update=634, checkpoint_interval=64)
    summaries = []
    for index, planned in enumerate(batches):
        receipt = read(f'run/rollout-{index:04d}.json')
        targets = update_targets(read(f'run/batch-{index:04d}.json'), receipt, planned, population, task,
                                 update=index, policy_id=current, reference_id=c['source_policy_id'])
        groups = torch.load(bound(root, transport, f'run/groups-{index:04d}.pt'), map_location='cpu', weights_only=True)
        summary = audit_groups(groups, receipt['queries'], targets, algorithm=c['algorithm'], update=index,
                               policy_id=current, reference_policy_id=c['source_policy_id'])
        step = index+1;current = lifecycle['policy_ids'][index]
        directory = root/f'run/policy-{step:04d}'
        bundle = read_execution_bundle(directory, expected_policy_id=current,
                       expected_contract=c['source_contract'], expected_base_id=c['base_id'])
        if not torch.equal(initial['tensors']['decision_token_row'], bundle['tensors']['decision_token_row']):
            raise ValueError('frozen marker changed')
        verify_native_adapter(root/f'run/adapter-{step:04d}', bundle)
        times = []
        for rank in range(4):
            record = read(f'run/rank-{rank}/update-{step:04d}.json')
            parity = record['behavior_parity_history']
            if (record['real_queries'] != summary['real_queries'] or len(parity) != 4
                    or any(len(group) != 4 for group in parity)
                    or any(p['passed'] is not True or p['threshold_flips'] != 0 for group in parity for p in group)):
                raise ValueError('actor population or four-microbatch behavior parity differs')
            times.append(record['total_seconds'])
        if any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in times):
            raise ValueError('finite positive step durations required')
        summary['slowest_rank_step_seconds'] = max(times);summaries.append(summary)
        if step in lifecycle['recovery_steps']:
            commit = read(f'run/checkpoint-{step:04d}-committed.json');contract = commit['contract']
            expected = {k:c[k] for k in ('base_id','source_policy_id','source_contract','runtime_profile',
                       'release_sha256','schedule_sha256','global_batch','microbatch','stage','algorithm','behavior_bounds','binary_policy_arithmetic')}
            expected.update(optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),
                objective=dict(group_size=16,reward='complete_candidate_set_f1',kl=.01,auxiliary='cosine_0.3_to_0.1_first_80pct'),grad_clip=1.,world_size=4)
            if contract != expected:raise ValueError('checkpoint objective contract differs')
            verify_checkpoint(root/'run', step, contract, bundle)
    for rank in range(4):
        report = read(f'run/rank-{rank}/result.json')
        if (report['passed'] is not True or report['start_update'] != 0 or report['stop_update'] != 634
                or len(report['executions']) != 113 or set(report['executions'].values()) != {'bfloat16'}):
            raise ValueError('rank completion or BF16 execution differs')
    if sum(r['real_queries'] for r in summaries) != 2*40513:
        raise ValueError('two full learning epochs required')
    return dict(passed=True, container_id=launch['container_id'], state=state, image=image,
                files=transport['files'], algorithm=c['algorithm'], updates=summaries,
                binary_policy_arithmetic=PROFILE,
                checkpoint_steps=lifecycle['recovery_steps'], source_policy_id=c['source_policy_id'],
                scope='Collected full learning schedule, sampling/advantages, published policies and saved optimizer audited; '
                      'source Stage1 admission/runtime release and task efficacy require separate evidence')


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--spec', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();spec = json.loads(a.spec.read_text())
    for name in ('transport','launch','experiment'):spec[name] = json.loads(Path(spec[name]).read_text())
    report = audit(**spec)
    with a.output.open('x') as stream:json.dump(report, stream, indent=2, allow_nan=False);stream.write('\n')
