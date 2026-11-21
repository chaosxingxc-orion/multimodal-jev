"""Finite/formal and fixed-reference windows for the shared native launcher."""
import math
import os
from pathlib import Path
import subprocess
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.tpo_anchor_layout import checked_parts
from tools.reports.build_tpo_view import read, verified
from tools.reports.joint_tpo_reference import path_for


def check_update(row, step, plan, binding, previous):
    ref=row['reference']; loss=row['metrics']; total=loss['jev/tpo_loss']
    if (row['step']!=step or row['task']!='joint' or not 0<row['real']<=plan['global_batch'] or
            ref['mode']!='joint_next_macro' or ref['policy_update']!=step-1 or
            ref['recent_policy']!=previous or ref['anchor_policy']!=plan['source_policy'] or
            ref['macro_slots']!=plan['global_batch'] or ref['config_sha256']!=binding['joint_sha'] or
            not all(math.isfinite(v) and v>0 for v in (row['seconds'],ref['seconds_max'],total,
                loss['actor/grad_norm'],loss['jev/noul_joint_kl'],loss['jev/open_joint_kl'])) or
            not math.isclose(total,.5*(loss['jev/noul_joint_kl']+loss['jev/open_joint_kl']),rel_tol=1e-6)):
        raise ValueError('invalid committed joint optimizer update')


def run(c):
    t=c['training']; root=Path(t['output']); root.mkdir()
    spec=Path(t['plan']); plan=read(verified(spec,t['joint_sha']))
    parent=Path(t['parent']); config=read(verified(parent,plan['parent_sha']))
    world=len(plan['training_devices'])
    ref=plan['inference_devices']
    fresh=plan.get('initialization')=='pre_task_policy'
    bound=dict(joint_sha=t['joint_sha'],source=c['source']['sha256'],target_sha=plan['target_sha'],
               parent_sha=plan['parent_sha'],world_size=world,initial_policy=plan['source_policy'])
    worker_os=dict(os.environ,HCCL_DETERMINISTIC='true',LCCL_DETERMINISTIC='1')

    def env(name,ordinals):
        cache=Path(os.environ['TMPDIR'])/name
        cache.mkdir()
        out=dict(worker_os,ASCEND_RT_VISIBLE_DEVICES=ordinals)
        for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
                    'ASCEND_WORK_PATH','ASCEND_CACHE_PATH','VLLM_RPC_BASE_PATH'):
            out[key]=str(cache)
        return out

    if t['window'] in ('anchor','finite-anchor'):
        from concurrent.futures import ThreadPoolExecutor
        offline = plan.get('offline_reference_devices', ref)
        split = 'offline_reference_devices' in plan
        steps=2 if t['window']=='finite-anchor' else plan['updates']
        if fresh and t['window']=='anchor':
            accepted=read(root.parent/'finite/receipt.json')
            if (accepted['binding']!=bound or accepted['passed'] is not True or accepted['updates']!=2
                    or accepted['window']!='finite' or accepted['anchor_sha256']!=file_hash(root.parent/'finite-anchor/receipt.json')):
                raise ValueError('full anchor requires matching finite approval')
        def anchor(replica):
            sub=root/f'reference-{replica}'
            subprocess.run([sys.executable,'-m','tools.reports.build_joint_tpo_anchor','--parent',str(parent),
                '--plan',str(spec),'--output',str(sub),'--replica',str(replica),'--steps',str(steps)],
                env=env(f'anchor_{replica}',str(replica)),check=True)
            r=read(sub/'receipt.json')
            if split:
                return r
            if (r['card']!=offline[replica] or r['source_policy']!=plan['source_policy'] or
                    r['target_sha']!=plan['target_sha'] or r['joint_sha']!=t['joint_sha'] or r['updates']!=steps or
                    set(map(int,r['rank_files']))!=set(range(replica,world,len(ref)))):
                raise ValueError('incomplete fixed reference shard')
            for rank,sha in r['rank_files'].items(): verified(sub/f'rank-{rank}.pt',sha)
            return r
        with ThreadPoolExecutor(max_workers=len(offline)) as pool:
            receipts=list(pool.map(anchor,range(len(offline))))
        if split:
            checked_parts(receipts, plan, t['joint_sha'], steps)
            for replica, receipt in enumerate(receipts):
                for name, sha in receipt['parts'].items():
                    verified(root/f'reference-{replica}'/name, sha)
        receipt=dict(binding=bound,passed=True,window=t['window'],updates=steps,replicas=receipts)
        if split:
            receipt['offline_reference_devices'] = offline
        commit_json(root/'receipt.json',receipt)
        return receipt
    finite_anchor='finite-anchor' if fresh else 'anchor'
    anchor_role=finite_anchor if t['window']=='finite' else 'anchor'
    anchor_path=root.parent/anchor_role/'receipt.json'
    anchor=read(anchor_path)
    if anchor['binding']!=bound or anchor['passed'] is not True or anchor['window']!=anchor_role:
        raise ValueError('complete pinned fixed reference required')
    if 'offline_reference_devices' in plan:
        anchor_updates = 2 if anchor_role == 'finite-anchor' else plan['updates']
        if (anchor.get('offline_reference_devices') != plan['offline_reference_devices']
                or anchor['updates'] != anchor_updates):
            raise ValueError('fixed reference was built with a different execution plan')
        checked_parts(anchor['replicas'], plan, t['joint_sha'], anchor_updates)
    if t['window']=='formal':
        accepted=read(root.parent/'finite/receipt.json')
        if (accepted['binding']!=bound or not accepted['passed'] or accepted['window']!='finite' or
                accepted['updates']!=2 or accepted['anchor_sha256']!=file_hash(root.parent/finite_anchor/'receipt.json')):
            raise ValueError('matching finite admission required')
    elif t['window']!='finite':
        raise ValueError('unknown joint window')
    steps=2 if t['window']=='finite' else plan['updates']
    from context_process_supervisor import supervise_services
    services=[]
    common=['--parent',str(parent),'--plan',str(spec),'--output',str(root)]
    for i,card in enumerate(ref):
        name=f'reference_{i}'
        services.append(dict(name=name,env=env(name,str((plan['training_devices']+ref).index(card))),
            ready_path=root/f'reference-{i}/ready.json',socket_path=path_for(root,i),
            command=[sys.executable,'-m','tools.reports.joint_tpo_reference',*common,'--replica',str(i),'--steps',str(steps)]))
    command=[sys.executable,'-m','torch.distributed.run','--master_addr','127.0.0.1','--master_port',
        str(config['view_execution']['master_port']),'--nproc_per_node',str(world),'--module',
        'tools.reports.train_joint_tpo',*common,'--window',t['window'],'--joint-sha',t['joint_sha']]
    life=supervise_services(services=services,actor_command=command,output=root/'supervision',
        actor_env=env('actor',','.join(map(str,range(world)))))
    if not life['passed']:
        raise ValueError('joint actor or TP1 reference service failed')
    if t['window']=='finite':
        for rank in range(world):
            parity=read(root/f'rank-{rank}/initial-parity.json')
            if len(parity)!=4 or any(not r['passed'] or r['rows']!=plan['microbatch'] for r in parity):
                raise ValueError('four initial probability parity passes required')
    previous=plan['source_policy']
    for step in range(1,steps+1):
        row=read(root/f'update-{step:04d}.json')
        check_update(row,step,plan,bound,previous)
        previous=row['reference']['next_policy']
    result=read(root/'result.json')
    endpoint=read(root/'policy/MANIFEST.json')
    if (result['updates']!=steps or result['policy']!=endpoint['policy_id'] or
            result['policy']==plan['source_policy'] or result['parent_config_sha256']!=plan['parent_sha'] or
            result['target_manifest_sha256']!=plan['target_sha']):
        raise ValueError('joint endpoint identity differs')
    receipt=dict(binding=bound,passed=True,window=t['window'],arm='tpo',updates=steps,endpoint=result['policy'],
                 anchor_sha256=file_hash(anchor_path),manifest_sha256=file_hash(root/'policy/MANIFEST.json'))
    commit_json(root/'receipt.json',receipt)
    return receipt
