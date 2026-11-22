"""One finite or formal torchrun window on the existing isolated job runner."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json


def run(c):
    def read(p):
        return json.loads(p.read_bytes())
    t=c['training']
    root=Path(t['output'])
    world=t['world_size']
    bound={k:t[k] for k in ('target_sha','parent_sha','world_size')}
    bound['source']=c['source']['sha256']
    if 'next_macro_sha' in t: bound['next_macro_sha']=t['next_macro_sha']
    if t['window']=='formal':
        receipt=read(Path(t['finite_receipt']))
        if receipt['passed'] is not True or receipt['binding']!=bound:
            raise ValueError('matching independent finite receipt required')
    env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=','.join(str(i) for i in range(world)),
             HCCL_DETERMINISTIC='true',LCCL_DETERMINISTIC='1')
    command=[sys.executable,'-m','torch.distributed.run','--master_addr','127.0.0.1',
             '--master_port',str(read(Path(t['parent']))['view_execution']['master_port']),
             '--nproc_per_node',str(world),'--module','tools.reports.train_tpo_stage2']
    for k in ('parent','targets','output','parent_sha','target_sha','arm','window'):
        command.extend(('--'+k.replace('_','-'),str(t[k])))
    if 'next_macro_sha' in t: command.extend(('--next-macro-sha',t['next_macro_sha']))
    subprocess.run(command,env=env,check=True)
    if t['window']=='reference':
        hashes=[file_hash(root/f'rank-{r}'/'choice.pt') for r in range(world)]
        receipt=dict(bound,rank_files=hashes,passed=True,window='reference')
        commit_json(root/'receipt.json',receipt)
        return receipt
    result=read(root/'result.json')
    m=read(Path(t['targets'])/'MANIFEST.json')
    steps=m['updates'] if t['window']=='formal' else 2
    if (result['updates']!=steps or result['policy']==m['source_policy'] or
            result['policy']!=read(root/'policy/MANIFEST.json')['policy_id']):
        raise ValueError('new endpoint not independently collected')
    if t['window']=='finite':
        for rank in range(world):
            parity=read(root/f'rank-{rank}'/'initial-parity.json')
            if len(parity)!=4 or not all(r['passed'] and r['rows']==m['schedule']['microbatch'] for r in parity):
                raise ValueError('initial producer/actor probability mismatch')
    pair_spec = m.get('actor_pair_margin')
    pair_by_epoch = {}
    if pair_spec is not None:
        from multimodal_jev.context_tpo_pair_margin import verify_update
    for step in range(1,steps+1):
        metric=read(root/f'update-{step:04d}.json')
        if 'next_macro_sha' in t:
            ref=metric['reference']
            if (ref['policy_update']!=step-1 or ref['macro_slots']!=m['schedule']['global_batch']
                    or ref['seconds_max']<=0 or metric['metrics']['jev/tpo_loss']<=0):
                raise ValueError('incomplete next-macro update')
        if pair_spec is not None:
            verify_update(metric, m['batch_real'], pair_spec['epochs'], pair_spec['batch_eligible'])
            pair_by_epoch.setdefault(str(metric['epoch']), Counter()).update(metric['pair_margin'])
        elif 'probability_gap' in m:
            from multimodal_jev.context_tpo_probability_gap import verify_gap_update
            verify_gap_update(metric, step, m['batch_real'])
        elif m['schema']=='jev-tpo-reward-view-v1':
            projection=metric['metrics'].get('jev/reward_projection_kl')
            if (projection is None or not projection>0 or projection!=metric['metrics']['jev/tpo_loss']
                    or any('jev/'+key in metric['metrics'] for key in ('hard_negative','margin','contrast','existence','consistency'))):
                raise ValueError('expected only positive reward-projection KL')
        if metric['step']!=step or metric['metrics']['actor/grad_norm']<=0:
            raise ValueError('failed optimizer update')
    receipt=dict(binding=bound,passed=True,window=t['window'],arm=t['arm'],updates=steps,
                 endpoint=result['policy'],manifest_sha256=file_hash(root/'policy/MANIFEST.json'))
    if pair_spec is not None:
        receipt['pair_margin_by_epoch'] = pair_by_epoch
    commit_json(root/'receipt.json',receipt)
    return receipt
