"""One finite or formal torchrun window on the existing isolated job runner."""
from collections import Counter
import json
import math
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
    if 'optimizer_sha' in t: bound['optimizer_sha'] = t['optimizer_sha']
    if 'next_macro_sha' in t: bound['next_macro_sha']=t['next_macro_sha']
    for check in t.get('card_inference_parity', []):
        proof = read(Path(check['path']))
        if (proof['physical_device'] != check['physical_device'] or proof['passed'] is not True
                or any(proof[k] != bound[k] for k in ('target_sha','parent_sha','source'))
                or proof.get('optimizer_sha') != t.get('optimizer_sha')
                or len(proof['passes']) != 4 or not all(r['passed'] for r in proof['passes'])):
            raise ValueError('inference receipt does not bind this card TPO window')
    if t['window']=='formal':
        receipt=read(Path(t['finite_receipt']))
        if receipt['passed'] is not True or receipt['binding']!=bound:
            raise ValueError('matching independent finite receipt required')
    env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=','.join(str(i) for i in range(world)),
             HCCL_DETERMINISTIC='true',LCCL_DETERMINISTIC='1')
    command=[sys.executable,'-m','torch.distributed.run','--master_addr','127.0.0.1',
             '--master_port',str(read(Path(t['parent']))['view_execution']['master_port']),
             '--nproc_per_node',str(world),'--module','tools.reports.train_tpo_stage2']
    if t['window'] == 'formal':
        parent = read(Path(t['parent']))
        if 'recovery' in parent:
            command[3:3] = ['--max-restarts',str(parent['recovery']['max_restarts'])]
    for k in ('parent','targets','output','parent_sha','target_sha','arm','window'):
        command.extend(('--'+k.replace('_','-'),str(t[k])))
    if 'next_macro_sha' in t: command.extend(('--next-macro-sha',t['next_macro_sha']))
    plan = None
    if 'optimizer_sha' in t:
        from tools.reports.card_tpo_bucket_package import load
        data = Path(t['targets'])
        m = read(data/'MANIFEST.json')
        plan = load(data.parent, t['optimizer_sha'], t['target_sha'], world, m['schedule']['epochs'])
        command.extend(('--optimizer-sha',t['optimizer_sha']))
    subprocess.run(command,env=env,check=True)
    if t['window']=='reference':
        hashes=[file_hash(root/f'rank-{r}'/'choice.pt') for r in range(world)]
        receipt=dict(bound,rank_files=hashes,passed=True,window='reference')
        commit_json(root/'receipt.json',receipt)
        return receipt
    result=read(root/'result.json')
    m=read(Path(t['targets'])/'MANIFEST.json')
    batches = None
    if plan:
        batches = plan['updates'] if t['window']=='formal' else [plan['updates'][i] for i in plan['finite']]
    steps = len(batches) if batches is not None else (m['updates'] if t['window']=='formal' else 2)
    counts = m['batch_real'] if batches is None else [b['real'] for b in batches]
    if (result['updates']!=steps or result['policy']==m['source_policy'] or
            result['policy']!=read(root/'policy/MANIFEST.json')['policy_id']):
        raise ValueError('new endpoint not independently collected')
    if t['window']=='finite':
        for rank in range(world):
            parity=read(root/f'rank-{rank}'/'initial-parity.json')
            if len(parity) != 4:
                raise ValueError('four independent actor parity passes required')
            for i, probe in enumerate(parity):
                if batches is None:
                    micro = m['schedule']['microbatch']
                else:
                    probe_batch = i//2
                    micro = batches[probe_batch]['microbatch']
                    comparison = 'frozen_stage1' if probe_batch == 0 else 'current_native_actor'
                    if (probe['step'], probe['comparison']) != (probe_batch+1, comparison):
                        raise ValueError('Stage2 parity tested the wrong actor batch')
                if not probe['passed'] or probe['rows'] != micro:
                    raise ValueError('producer/actor probability mismatch')
    pair_spec = m.get('actor_pair_margin')
    pair_by_epoch = {}
    bucket_timing = {}
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
            verify_gap_update(metric, step, counts)
        elif m['schema']=='jev-tpo-reward-view-v1':
            projection=metric['metrics'].get('jev/reward_projection_kl')
            if (projection is None or not projection>0 or projection!=metric['metrics']['jev/tpo_loss']
                    or any('jev/'+key in metric['metrics'] for key in ('hard_negative','margin','contrast','existence','consistency'))):
                raise ValueError('expected only positive reward-projection KL')
        if batches is not None:
            batch = batches[step-1]
            timing = metric['timing']
            if (metric['optimizer_sha'] != t['optimizer_sha'] or metric['epoch'] != batch['epoch'] or metric['real'] != batch['real']
                    or metric['bucket'] != {k: batch[k] for k in metric['bucket']}
                    or set(metric['bucket']) != {'bucket','local_batch','microbatch','max_tokens','epoch_end'}
                    or set(timing) != {'input_wait_seconds','train_seconds'}
                    or not all(math.isfinite(v) and v >= 0 for v in timing.values())
                    or timing['train_seconds'] <= 0 or type(metric['peak_bytes']) is not int or metric['peak_bytes'] <= 0):
                raise ValueError('invalid Stage2 batch geometry or timing evidence')
            summary = bucket_timing.setdefault(str(batch['bucket']), dict(updates=0, recordings=0,
                input_wait_seconds=0., train_seconds=0., peak_bytes=0))
            summary['updates'] += 1
            summary['recordings'] += batch['real']
            summary['peak_bytes'] = max(summary['peak_bytes'], metric['peak_bytes'])
            for key in timing:
                summary[key] += timing[key]
        if metric['step']!=step or metric['metrics']['actor/grad_norm']<=0:
            raise ValueError('failed optimizer update')
    receipt=dict(binding=bound,passed=True,window=t['window'],arm=t['arm'],updates=steps,
                 endpoint=result['policy'],manifest_sha256=file_hash(root/'policy/MANIFEST.json'))
    if batches is not None:
        for summary in bucket_timing.values():
            summary['recordings_per_train_second'] = summary['recordings']/summary['train_seconds']
            summary['pairs_per_train_second'] = summary['recordings_per_train_second']*len(m['candidate_ids'])
        receipt.update(real_exposures=sum(counts), bucket_timing=bucket_timing)
    if pair_spec is not None:
        receipt['pair_margin_by_epoch'] = pair_by_epoch
    commit_json(root/'receipt.json',receipt)
    return receipt
