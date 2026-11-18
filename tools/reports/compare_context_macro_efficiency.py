"""Re-audit matched DP2 arms and compare full-update throughput.

Does not select a model using target data or assert identical BF16 training
trajectories across microbatch partitions. No launch or cleanup side effects.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.audit_context_macro_training import audit_collected
from tools.reports.context_macro_efficiency_summary import summarize
from tools.reports.prepare_context_dual_replay import relocate


def controls(training, services, remote):
    cfg = relocate(training, remote, '/comparison')
    micro = cfg.pop('microbatch')
    if micro not in (8, 16, 32):
        raise ValueError('registered microbatch required')
    svcs = [relocate(s, remote, '/comparison') for s in services]
    return cfg, svcs


def compare(c):
    plan_path = Path(c['plan'])
    if file_hash(plan_path) != c['plan_sha256']:
        raise ValueError('frozen comparison plan changed')
    plan = json.loads(plan_path.read_text())
    names = list(c['arms'])
    if (len(names) < 2 or len(set(names)) != len(names) or 'micro8' not in names
            or any(n not in plan['arms'] for n in names)):
        raise ValueError('baseline and at least one declared arm required')
    if type(c['warmup_updates']) is not int or not 0 <= c['warmup_updates'] < plan['steps_per_arm']:
        raise ValueError('explicit common warmup window required')
    results = {}; baseline_controls = None; baseline_records = None; baseline_cache = None
    images = set()
    for name in names:
        spec = c['arms'][name]
        ops, _ = load_config(spec['config'], spec['local'])
        expected = plan['arms'][name]
        if ops['manifest_sha256'] != expected['manifest_sha256'] or ops['remote_root'] != expected['remote_root']:
            raise ValueError('actual arm not bound to plan')
        # Full bytes, container scope, targets, groups, parity and checkpoints.
        audit = audit_collected(ops)
        if audit['status'] != 'finite_dp2_efficiency_audited':
            raise ValueError('completed independently audited efficiency arm required')
        package = Path(ops['prepared'])/'package';run = Path(ops['operations_root'])/'collected/run'
        t = json.loads((package/'training.json').read_text())
        services = [json.loads((package/f'service-{i}.json').read_text()) for i in range(2)]
        if (t['microbatch'] != expected['microbatch'] or t['stop_update'] != plan['steps_per_arm']
                or t['start_update'] != 0 or file_hash(package/'RUNTIME-SOURCES.json') != plan['runtime_sources_sha256']):
            raise ValueError('budget, capacity or common runtime differs')
        matched = controls(t, services, ops['remote_root'])
        records = [json.loads((run/f'batch-{i:04d}.json').read_text())['records'] for i in range(t['stop_update'])]
        stats = summarize(run,start_update=0,stop_update=t['stop_update'],warmup_updates=c['warmup_updates'])
        cache = [s['reference_misses'] for s in stats['steps']]
        if baseline_controls is None:
            baseline_controls, baseline_records, baseline_cache = matched, records, cache
        elif matched != baseline_controls or records != baseline_records or cache != baseline_cache:
            raise ValueError('confounded configuration, actual samples or reference cache condition')
        images.add(audit['image'])
        results[name] = dict(microbatch=t['microbatch'],statistics=stats,
            audit_manifest_sha256=audit['files_sha256'],transport_sha256=audit['transport_sha256'],
            containers=audit['containers'])
    if len(images) != 1:
        raise ValueError('different runtime images')
    baseline = results['micro8']['statistics']
    for name, result in results.items():
        stats = result['statistics']
        result['all_updates_speedup'] = baseline['all_updates']['complete_seconds']/stats['all_updates']['complete_seconds']
        result['after_warmup_speedup'] = baseline['measured']['complete_seconds']/stats['measured']['complete_seconds']
        result['paired_step_speedup'] = [a['complete_seconds']/b['complete_seconds']
            for a,b in zip(baseline['steps'],stats['steps'],strict=True)]
    return dict(schema='context-macro-efficiency-comparison-v1',plan_sha256=c['plan_sha256'],
        arms=results,image=next(iter(images)),warmup_updates=c['warmup_updates'],
        matched_actual_queries=True,matched_reference_misses=True,independent_audits=True,
        automatic_training_switch=False,efficacy_claim=False,
        scope='Matched complete optimizer updates through dual activation. BF16 accumulation can differ across microbatches; '
              'no exact cross-arm parameter claim. Startup, evaluation and inference memory excluded.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    import torch
    torch.set_num_threads(1)
    result=compare(c);result['config_provenance']=provenance
    with Path(c['output']).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:{n:v[n] for n in ('microbatch','all_updates_speedup','after_warmup_speedup')} for k,v in result['arms'].items()},indent=2))
