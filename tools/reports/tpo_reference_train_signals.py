"""Summarize actual TRAIN slots and pre-clip norms from two formal TPO endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from statistics import median

import torch


def read(p):
    return json.loads(p.read_bytes())


def verify(p, expected):
    if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
        raise ValueError(f'changed input: {p.name}')


def summarize(local, lock):
    c=read(local)['macros']; cfg=read(lock)
    e=Path(c['TPO_PACKAGE'])/'entry'; data=e/'targets'
    m=read(data/'MANIFEST.json'); b=cfg['binding']; spec=read(e/'next-macro.json')
    for name,path,h in [('shift',e/'reward-shift.pt',spec['reward_shift']),('spec',e/'next-macro.json',b['next_macro_sha']),
                        ('manifest',data/'MANIFEST.json',b['target_sha'])]:
        verify(path,h)
    for name,h in m['files'].items(): verify(data/name,h)
    batches=read(data/'batches.json'); rows=read(data/'rows.json')
    shifts=torch.load(e/'reward-shift.pt',map_location='cpu',weights_only=True)
    ref=Path(c['TPO_REFERENCE']); verify(ref/'MANIFEST.json',m['reference_sha256'])
    verify(ref/'records.json',read(ref/'MANIFEST.json')['files']['records.json'])
    records=read(ref/'records.json')
    shifts_used=[]; positives=[]; hardest=[]
    for batch in batches:
        if batch['mode']!='noul':continue
        for slot in batch['slots']:
            if slot['padding']:continue
            row=rows[slot['id']]; y=row['gold']
            v=shifts[batch['epoch'],row['index']]
            ps=records[row['request_key']]['prob']
            h=max((i for i in range(len(v)) if i!=y),key=ps.__getitem__)
            shifts_used.append(torch.cat([v[:y],v[y+1:]]));positives.append(float(v[y]));hardest.append(float(v[h]))
    ns=torch.stack(shifts_used); n=len(positives)
    result=dict(noul_real=n,old_gold_median_shift=median(positives),old_strongest_negative_median_shift=median(hardest),
                other_and_hard_negatives_median_shift=float(ns.median()),positive_shift_fraction_on_negatives=float((ns>0).double().mean()),
                train_tasks=m['task_counts'],reward_coefficients=m['rewards'],arms={})
    remote="""import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'tpo'
r=json.loads((p/'receipt.json').read_bytes())
assert r['passed'] and r['window']=='formal' and r['endpoint']==sys.argv[2]
print(json.dumps([json.loads((p/f'update-{i:04d}.json').read_bytes()) for i in range(1,r['updates']+1)]))
"""
    for arm,root in [('prior',c['TPO_CONTROL_ROOT']),('tpo',c['TPO_DEPLOYMENT']['remote_root'])]:
        updates=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',c['TPO_DEPLOYMENT']['host'],
            shlex.join(['python3','-',root,cfg['arms'][arm]])],input=remote,text=True))
        verify(Path(c['TPO_PARENT_CONFIG']),b['parent_sha'])
        clip=read(Path(c['TPO_PARENT_CONFIG']))['view_execution']['grad_clip']
        if len(updates)!=len(batches) or any(u['task']!=bt['mode'] or u['step']!=i or u['real']!=sum(not s['padding'] for s in bt['slots']) for i,(u,bt) in enumerate(zip(updates,batches),1)):
            raise ValueError('update schedule mismatch')
        def summary(task):
            us=[u for u in updates if u['task']==task]
            norms=[u['metrics']['actor/grad_norm'] for u in us]
            return dict(steps=len(us),median_grad_norm=median(norms),grad_clip=clip,norm_above_clip=sum(v>clip for v in norms),
                        median_projection_loss=median(u['metrics']['jev/tpo_loss'] for u in us))
        result['arms'][arm]={task:summary(task) for task in ('noul','open')}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('local',type=Path);p.add_argument('lock',type=Path);args=p.parse_args()
    print(json.dumps(summarize(args.local,args.lock),indent=2))
