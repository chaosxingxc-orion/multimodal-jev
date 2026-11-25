"""Verify an actual DP8 continuation against the retired run's overlapping updates."""
import json
from pathlib import Path
import shlex
import subprocess

from multimodal_jev.local_checkpoint import commit_json
from tools.reports.collect_card_first_updates import summarize
from tools.reports.watch_stage1_first_updates import REMOTE


def audit(package, steps, timeout):
    read=lambda p:json.loads(p.read_bytes())
    ops=read(package/'operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    root=Path(ops['operations'])
    launch=read(root/'formal-launch.json')
    if launch['remote_root']!=ops['remote_root'] or launch['host']!=ops['host']:
        raise ValueError('wrong continuation handle')
    def remote(*args):
        return json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',ops['host'],shlex.join(args)]))
    leaf=read(package/'entry/formal.json')['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    origin=leaf['continuation']
    cursor=origin['step']
    devices=ops['devices']
    mappings=dict(m.split(':')[:2][::-1] for m in ops['read_only_mounts'])
    declared=Path(origin['root'])
    mount=max((Path(m) for m in mappings if declared.is_relative_to(m)),key=lambda m:len(m.parts))
    prior=Path(mappings[str(mount)])/declared.relative_to(mount)
    def updates(directory):
        cfg=dict(root=str(directory),world=len(devices),updates=steps,start_update=cursor,timeout=timeout)
        return remote('python3','-c',REMOTE.replace('CONFIG',repr(cfg),1))['records']
    formal=Path(ops['remote_root'])/'formal'
    rows=updates(formal)
    old=updates(prior)
    cfg=remote('cat',str(formal/'config.json'))
    if cfg['continuation']!=origin or cfg['physical_devices']!=devices or cfg['global_batch']!=cfg['microbatch']*len(devices):
        raise ValueError('incorrect resumed configuration')
    shard=remote('cat',str(prior/'run/recovery'/f'step-{cursor}.json'))
    proofs=[remote('cat',str(formal/'run'/f'rank-{k}'/f'restored-{cursor:04d}.json')) for k in range(len(devices))]
    if any(not (p['exact_masters'] and p['exact_adam'] and p['step']==p['ema_updates']==cursor
                and p['rank']==r and p['pending_version']==cursor-1 and p['shard_sha256']==shard['shards'][r])
           for r,p in enumerate(proofs)):
        raise ValueError('incomplete exact restore proof')
    result=[]
    for s in range(steps):
        group=rows[s*len(devices):(s+1)*len(devices)]
        prev=old[s*len(devices):(s+1)*len(devices)]
        if any(a['metrics']!=b['metrics'] or a['learning_rate']!=b['learning_rate'] or a['optimizer_audit']!=b['optimizer_audit']
               for a,b in zip(group,prev,strict=True)):
            raise ValueError('resumed distributed update differs from predecessor')
        result.append(summarize(group,cursor+s+1,cfg))
    obj,=remote('docker','inspect',launch['container_id'])
    if obj['State']['Status']!='running' or obj['Image']!=ops['image'] or obj['Id']!=launch['container_id']:
        raise ValueError('continuation is not live on the declared image')
    proof=dict(passed=True,container_id=obj['Id'],restored=cursor,world=len(devices),engine=cfg.get('training_acoustic_engine'),
               updates=result,endpoint=cfg['stop_update'],old_metrics_matched=True)
    commit_json(root/'cutover-startup.json',proof)
    return proof
