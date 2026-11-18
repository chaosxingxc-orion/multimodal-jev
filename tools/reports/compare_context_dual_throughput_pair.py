"""Compare authenticated two-step collections; never stop or start training."""
import argparse
import json
import math
from pathlib import Path

import torch

from multimodal_jev.local_config import load_config
from tools.reports.prepare_context_dual_throughput_pair import sha,write
from tools.reports.prepare_context_dual_training_candidate import rebase


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def summarize(root,transport):
    root=Path(root);wire=json.loads(Path(transport).read_text());state=wire['state']
    if (wire['status']!='transport_complete_independent_audit_required' or
            state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled'] or
            any(wire[k] for k in ('bad_pins','bad_package','launch_binding_mismatches'))):
        raise ValueError('successful unchanged terminal collection required')
    def bound(name):
        path=root/name
        if name not in wire['files'] or sha(path)!=wire['files'][name]:
            raise ValueError('changed or unbound collected evidence: '+name)
        return path
    def read(name):return json.loads(bound(name).read_text())
    config=read('config.json');service=read('service-config.json');prepared=read('PREPARED.json')
    if (config['start_update']!=0 or config['stop_update']!=2 or config['checkpoint_interval']!=1 or
            config['engineering_prefix'] is not True or config['training_dispatch_allowed'] is not False):
        raise ValueError('exact initial two-step prefix required')
    terminal=read('supervision/terminal.json')
    if (not terminal['passed'] or terminal['forced_cleanup'] or
            terminal['actor_exit_code']!=0 or terminal['service_exit_code']!=0 or
            terminal['child_exit_codes']!=[0,0] or read('service/terminal.json')['exit_code']!=0 or
            read('run/result.json')['passed'] is not True):
        raise ValueError('normal completed actor and service required')
    steps=[];shards=[]
    for step in (1,2):
        ranks=[]
        commit=read(f'run/checkpoint-{step:04d}-committed.json')
        for rank in range(4):
            row=read(f'run/rank-{rank}/update-{step:04d}.json')
            if row['update']!=step or row['checkpoint'] is not True or row['real_queries']!=128:
                raise ValueError('complete DP4 real-query step required')
            for key in ('total_seconds','preparation_seconds','actor_seconds'):
                v=row[key]
                if type(v) not in (int,float) or not math.isfinite(v) or v<=0:
                    raise ValueError('finite positive stage timers required')
            if row['actor_seconds']+row['preparation_seconds']>row['total_seconds']+1e-6:
                raise ValueError('stage timing exceeds full update')
            parity=row['behavior_parity_history']
            if len(parity)!=4 or any(len(group)!=4 or any(p['passed'] is not True for p in group) for group in parity):
                raise ValueError('complete per-microbatch collective parity required')
            path=bound(f'run/rank-{rank}/checkpoint-{step:04d}.pt')
            if commit['rank_shards'][f'rank-{rank}/checkpoint-{step:04d}.pt']!=sha(path):
                raise ValueError('all-rank checkpoint commit mismatch')
            shards.append(path);ranks.append(row)
        ids={v['next_policy_id'] for v in ranks}
        if len(ids)!=1:raise ValueError('cross-rank activated policy differs')
        rollout=ranks[0]['rollout']
        steps.append(dict(update=step,total_seconds=max(v['total_seconds'] for v in ranks),
            actor_seconds=max(v['actor_seconds'] for v in ranks),
            preparation_seconds=max(v['preparation_seconds'] for v in ranks),
            reference_misses=rollout['reference_misses'],policy_id=next(iter(ids))))
    return dict(config=config,service=service,prepared=prepared,steps=steps,shards=shards,
        image=wire['image'],container=wire['container_id'])


def compare(c):
    pair=json.loads(Path(c['pair']).read_text())
    if sha(c['pair'])!=c['pair_sha256']:raise ValueError('comparison plan changed')
    arms=[summarize(c[name]['collection'],c[name]['transport']) for name in ('serial','batch4')]
    configs=[];services=[]
    for name,arm in zip(('serial','batch4'),arms):
        expected=pair['arms'][name]
        if sha(Path(c[name]['collection'])/'PREPARED.json')!=expected['manifest_sha256']:
            raise ValueError('executed arm differs from prepared plan')
        remote=arm['prepared']['remote_root']
        cfg=rebase(arm['config'],remote,'/comparison');svc=rebase(arm['service'],remote,'/comparison')
        if svc['policy'].pop('max_batch_queries')!=expected['max_batch_queries']:
            raise ValueError('actual service capacity differs')
        configs.append(cfg);services.append(svc)
    if configs[0]!=configs[1] or services[0]!=services[1] or arms[0]['image']!=arms[1]['image']:
        raise ValueError('confounded objective, runtime or schedule comparison')
    for a,b in zip(arms[0]['steps'],arms[1]['steps'],strict=True):
        if a['reference_misses']!=b['reference_misses'] or a['policy_id']!=b['policy_id']:
            raise ValueError('reference cache condition or resulting policy differs')
    for a,b in zip(arms[0]['shards'],arms[1]['shards'],strict=True):
        x,y=[torch.load(p,map_location='cpu',weights_only=True) for p in (a,b)]
        if not equal(x,y):raise ValueError('training masters, Adam or checkpoint contract differ')
    seconds=[sum(s['total_seconds'] for s in arm['steps']) for arm in arms]
    return dict(schema='context-dual-throughput-prefix-comparison-v1',
        actual_collections=True,exact_two_step_checkpoints=True,
        serial_seconds=seconds[0],batch4_seconds=seconds[1],speedup=seconds[0]/seconds[1],
        per_step_speedup=[a['total_seconds']/b['total_seconds'] for a,b in zip(arms[0]['steps'],arms[1]['steps'])],
        containers=[arm['container'] for arm in arms],pair_sha256=c['pair_sha256'],
        switch_old_training_allowed=False,
        pending=['independent full training audit','fresh-process replay','representative sustained timing'],
        scope='Two matched full-update timers and exact checkpoints; excludes startup, cannot alone authorize a training switch')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);r=compare(c)
    r['config_provenance']=provenance;write(c['output'],r)
    print(json.dumps(r,indent=2))
