"""Audit saved continuous updates as a future recovery comparison reference.

Read-only CPU checks. No NPU replay or throughput improvement certification.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from multimodal_jev.local_config import load_config
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from multimodal_jev.context_training_schedule import load_schedule
from multimodal_jev.slurp_official_training import load
from tools.jev.audit_context_rl_training import update_targets
from tools.jev.context_rl_group_audit import audit as audit_groups
from tools.reports.collect_committed_context_checkpoint import inventory
from tools.reports.prepare_context_dual_throughput_pair import sha, write


def check_update(records, activation, export, summary, policy_id):
    ack = activation['receipt']
    if (activation['error'] is not None or activation['policy_id'] != policy_id
            or ack['policy_id'] != policy_id or ack['head_marker_exact'] is not True
            or ack['native_loaded_tensors'] != 112 or export['recovery_checkpoint'] is not False):
        raise ValueError('successful policy activation without intermediate recovery required')
    times = []
    for record in records:
        parity = record['behavior_parity_history']
        if (record['checkpoint'] is not False or record['real_queries'] != summary['real_queries']
                or len(parity) != 4 or any(len(group) != 4 for group in parity)
                or any(p['passed'] is not True or p['threshold_flips'] != 0
                       for group in parity for p in group)):
            raise ValueError('four-rank population and behavior parity required')
        times.append(record['total_seconds'])
    if len(records) != 4 or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError('four finite positive full-step durations required')
    return max(times)


def audit(c):
    snapshot = Path(c['snapshot']); root = snapshot/'collection'; run = root/'run'
    proof = json.loads((snapshot/'COMPLETE.json').read_text())
    attempt = json.loads((snapshot/'attempt.json').read_text())['config']
    launch = json.loads(Path(attempt['launch']).read_text()); step = attempt['step']
    if (sha(attempt['launch']) != attempt['launch_sha256']
            or proof['schema'] != 'context-committed-following-snapshot-v1'
            or attempt['following_steps'] != 2 or proof['step'] != step
            or proof['comparison_stop_update'] != step+2
            or proof['following_optimizer_states_collected'] is not False
            or proof['container_id'] != launch['container_id'] or proof['image'] != launch['image']
            or proof['collection'] != str(root) or sha(snapshot/'checkpoint.tar') != proof['archive_sha256']):
        raise ValueError('following snapshot provenance differs')
    actual = inventory(root, step, launch, following_steps=2)
    if actual != json.loads((snapshot/'inventory.json').read_text()) or any(
            actual[k] != proof[k] for k in actual):
        raise ValueError('following snapshot inventory differs')
    config = json.loads((root/'config.json').read_text())
    if config['algorithm'] != 'rloo_v1' or any(s % config['checkpoint_interval'] == 0 for s in (step+1, step+2)):
        raise ValueError('original RLOO non-checkpoint comparison updates required')
    rows, task, _ = load(c['release'], config['release_sha256'], 'learning', purpose='optimize')
    population = {r['id']: r for r in rows}
    batches, _ = load_schedule(c['schedule'], config['schedule_sha256'],
                              release_sha256=config['release_sha256'], population=population)
    if len(population) != 40513 or len(batches) != 634:
        raise ValueError('full original population and schedule required')
    current = proof['policy_id']
    initial = read_execution_bundle(run/f'policy-{step:04d}', expected_policy_id=current,
        expected_contract=config['source_contract'], expected_base_id=config['base_id'])
    summaries = []
    for index in range(step, step+2):
        def read(name):
            return json.loads((run/name).read_text())
        batch = read(f'batch-{index:04d}.json'); receipt = read(f'rollout-{index:04d}.json')
        targets = update_targets(batch, receipt, batches[index], population, task,
            update=index, policy_id=current, reference_id=config['source_policy_id'])
        groups = torch.load(run/f'groups-{index:04d}.pt', map_location='cpu', weights_only=True)
        expected_order = [batch['records'][i]['request_id']
                          for indices in batches[index]['rank_indices'] for i in indices]
        if [g['request_id'] for g in groups] != expected_order:
            raise ValueError('actual group order differs from scheduled rank concatenation')
        summary = audit_groups(groups, receipt['queries'], targets, algorithm=config['algorithm'],
            update=index, policy_id=current, reference_policy_id=config['source_policy_id'])
        current = proof['following_policy_ids'][index-step]
        bundle = read_execution_bundle(run/f'policy-{index+1:04d}', expected_policy_id=current,
            expected_contract=config['source_contract'], expected_base_id=config['base_id'])
        if not torch.equal(initial['tensors']['decision_token_row'], bundle['tensors']['decision_token_row']):
            raise ValueError('frozen marker changed')
        records = [read(f'rank-{rank}/update-{index+1:04d}.json') for rank in range(4)]
        summary['slowest_rank_step_seconds'] = check_update(records, read(f'activation-{index+1:04d}.json'),
            read(f'policy-export-{index+1:04d}.json'), summary, current)
        summary['rank0_timing'] = {k: records[0][k] for k in ('preparation_seconds', 'actor_seconds', 'total_seconds')}
        summary['rollout_timing'] = {k: receipt[k] for k in ('reference_seconds', 'total_seconds', 'reference_misses')}
        summaries.append(summary)
    return dict(schema='context-following-reference-audit-v1', passed=True,
        source_container=launch['container_id'], start_update=step, stop_update=step+2,
        snapshot_manifest_sha256=sha(snapshot/'COMPLETE.json'), implementation_sha256=sha(__file__),
        updates=summaries, policy_ids=proof['following_policy_ids'],
        native_adapter_tensor_audit=False, following_optimizer_states_available=False,
        fresh_process_npu_resume_verified=False, throughput_gain_verified=False,
        stop_training_allowed=False,
        scope='Saved continuous sample/reward/parity/publication reference; no new NPU run or comparison of unrecorded Adam states')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    a = p.parse_args(); c, provenance = load_config(a.config, a.local); torch.set_num_threads(1)
    result = audit(c); result['config_provenance'] = provenance; write(c['output'], result)
    print(json.dumps(result, indent=2))
