"""Fixed-script DP2 efficiency statistics for at most 100 optimizer updates.

This reads collected timers and policy acknowledgements. Correctness/recovery
admission is a separate audit; throughput statistics do not prove efficacy.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.audit_context_macro_training import activation


def summarize(root, *, start_update, stop_update, warmup_updates=0):
    if (any(type(v)!=int for v in (start_update,stop_update,warmup_updates))
            or not 0<=start_update<stop_update<=100
            or not 0<=warmup_updates<stop_update-start_update):
        raise ValueError('explicit optimizer interval within 100 updates and nonempty measured window required')
    root=Path(root);hashes={};rows=[]
    def read(name):
        p=root/name
        if p.is_symlink():raise ValueError('regular collected file required')
        hashes[name]=file_hash(p);return json.loads(p.read_text())
    result=read('result.json')
    if (result['passed'] is not True or result['start_update']!=start_update
            or result['stop_update']!=stop_update):raise ValueError('complete requested interval required')
    expected_updates={f'update-{i:04d}.json' for i in range(start_update+1,stop_update+1)}
    for rank in range(2):
        if {p.name for p in (root/f'rank-{rank}').glob('update-*.json')}!=expected_updates:
            raise ValueError('extra or missing optimizer update; enforce 100-step limit')
    for index in range(start_update,stop_update):
        step=index+1;ranks=[read(f'rank-{rank}/update-{step:04d}.json') for rank in range(2)]
        r=ranks[0];rollout=r['rollout'];ack=read(f'activation-{step:04d}.json')
        if ack['error'] is not None or ack['policy_id']!=r['next_policy_id']:raise ValueError('step missing policy activation')
        activation(ack['receipt'],r['next_policy_id'])
        for peer in ranks:
            if (peer['update']!=step or peer['real_queries']!=128 or peer['next_policy_id']!=r['next_policy_id']
                    or peer['rollout']!=rollout):raise ValueError('complete matching DP2 global update required')
            for key in ('total_seconds','preparation_seconds','actor_seconds','peak_allocated'):
                value=peer[key]
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError('finite nonnegative timer or memory required')
            if (peer['total_seconds']<=0 or peer['actor_seconds']<=0
                    or peer['preparation_seconds']+peer['actor_seconds']>peer['total_seconds']+1e-6):
                raise ValueError('phase timer exceeds complete optimizer step')
        if (rollout['update']!=index or not 0<=rollout['reference_misses']<=128
                or type(rollout['reference_misses'])!=int):raise ValueError('rollout cursor or reference count differs')
        for key in ('total_seconds','reference_seconds','behavior_and_credit_seconds'):
            if type(rollout[key]) not in (int,float) or not math.isfinite(rollout[key]) or rollout[key]<0:
                raise ValueError('invalid nested rollout timer')
        if (rollout['reference_seconds']+rollout['behavior_and_credit_seconds']>rollout['total_seconds']+1e-6
                or rollout['total_seconds']>r['preparation_seconds']+1e-6):
            raise ValueError('nested rollout phases exceed parent timer')
        rows.append(dict(update=step,real_queries=128,complete_seconds=max(p['total_seconds'] for p in ranks),
            rank0_preparation_seconds=r['preparation_seconds'],rank0_actor_seconds=r['actor_seconds'],
            rank0_export_checkpoint_sync_seconds=r['total_seconds']-r['preparation_seconds']-r['actor_seconds'],
            rollout_seconds=rollout['total_seconds'],reference_seconds=rollout['reference_seconds'],
            behavior_and_credit_seconds=rollout['behavior_and_credit_seconds'],reference_misses=rollout['reference_misses'],
            peak_actor_allocated_bytes=max(p['peak_allocated'] for p in ranks),policy_id=r['next_policy_id']))
    measured=rows[warmup_updates:]
    def window(items):
        if not items:return None
        total=sum(r['complete_seconds'] for r in items)
        keys=('complete_seconds','rank0_preparation_seconds','rank0_actor_seconds',
              'rank0_export_checkpoint_sync_seconds','rollout_seconds','reference_seconds','behavior_and_credit_seconds')
        return dict(updates=len(items),real_query_exposures=128*len(items),complete_seconds=total,
            queries_per_second=128*len(items)/total,
            mean_seconds={k:statistics.mean(r[k] for r in items) for k in keys},
            median_complete_seconds=statistics.median(r['complete_seconds'] for r in items),
            peak_actor_allocated_bytes=max(r['peak_actor_allocated_bytes'] for r in items),
            reference_misses=sum(r['reference_misses'] for r in items))
    return dict(schema='context-macro-efficiency-summary-v1',optimizer_steps=len(rows),
        warmup_updates=warmup_updates,all_updates=window(rows),measured=window(measured),
        reference_miss_updates=window([r for r in measured if r['reference_misses']]),
        reference_hit_only_updates=window([r for r in measured if not r['reference_misses']]),
        steps=rows,files=hashes,correctness_audit_required=True,
        scope='Complete DP2 update timers through dual activation; nested phases are not additive. '
              'Startup, evaluation and inference memory not measured; no device-hours or efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=summarize(c['run_root'],start_update=c['start_update'],stop_update=c['stop_update'],warmup_updates=c['warmup_updates'])
    result['config_provenance']=provenance
    output=Path(c['output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result['measured'],indent=2))
