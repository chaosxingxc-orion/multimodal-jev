"""Read committed formal-training timers; never starts or changes training."""
import json
import math
from pathlib import Path
import statistics
from multimodal_jev.baseline import file_hash
from tools.reports.audit_context_replica_training import activation


def summarize(root,*,start_update,stop_update,warmup_updates,world_size=2,replicas=2):
    if (any(type(v)is not int for v in (start_update,stop_update,warmup_updates,world_size,replicas))
            or not 0<=start_update<stop_update or not 0<=warmup_updates<stop_update-start_update
            or world_size<1 or replicas<1):raise ValueError('explicit formal sampling window required')
    root=Path(root);rows=[];hashes={}
    def read(name):
        p=root/name
        if p.is_symlink():raise ValueError('regular timer evidence required')
        raw=p.read_bytes();import hashlib
        hashes[name]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    def seconds(value):
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('finite nonnegative timing required')
        return value
    for update in range(start_update+1,stop_update+1):
        ranks=[read(f'rank-{rank}/update-{update:04d}.json') for rank in range(world_size)]
        first=ranks[0];rollout=first['rollout'];ack=read(f'activation-{update:04d}.json')
        if ack['error'] is not None or ack['policy_id']!=first['next_policy_id']:raise ValueError('committed activation required')
        activation(ack['receipt'],first['next_policy_id'],replicas)
        for peer in ranks:
            if (peer['update']!=update or peer['next_policy_id']!=first['next_policy_id'] or peer['rollout']!=rollout
                    or peer['real_queries']!=first['real_queries'] or type(peer['real_queries'])is not int
                    or peer['real_queries']<=0):raise ValueError('matching complete rank timers required')
            for key in ('total_seconds','preparation_seconds','actor_seconds','peak_allocated'):seconds(peer[key])
            if peer['total_seconds']<=0 or peer['preparation_seconds']+peer['actor_seconds']>peer['total_seconds']+1e-6:
                raise ValueError('phase exceeds complete update')
        for key in ('total_seconds','reference_seconds','behavior_and_credit_seconds'):seconds(rollout[key])
        if (rollout['update']!=update-1 or rollout['total_seconds']>first['preparation_seconds']+1e-6
                or rollout['reference_seconds']+rollout['behavior_and_credit_seconds']>rollout['total_seconds']+1e-6):
            raise ValueError('nested rollout timing differs')
        rows.append(dict(update=update,real_queries=first['real_queries'],
            complete_seconds=max(v['total_seconds'] for v in ranks),
            actor_seconds=max(v['actor_seconds'] for v in ranks),rollout_seconds=rollout['total_seconds'],
            rank0_export_checkpoint_sync_seconds=max(0.,first['total_seconds']-first['preparation_seconds']-first['actor_seconds']),
            peak_actor_allocated_bytes=max(v['peak_allocated'] for v in ranks)))
    measured=rows[warmup_updates:];elapsed=sum(r['complete_seconds'] for r in measured)
    return dict(schema='jev-formal-efficiency-sample-v1',start_update=start_update,stop_update=stop_update,
        warmup_updates=warmup_updates,steps=rows,files=hashes,
        real_query_exposures=sum(r['real_queries'] for r in measured),
        queries_per_second=sum(r['real_queries'] for r in measured)/elapsed,
        mean_seconds={k:statistics.mean(r[k] for r in measured) for k in
            ('complete_seconds','actor_seconds','rollout_seconds','rank0_export_checkpoint_sync_seconds')},
        scope='Sampled formal updates only; nested phases are not additive; export/checkpoint/sync is an aggregate residual, not independent timers.',
        separate_training=False,model_calls=0,npu_executed=False)
