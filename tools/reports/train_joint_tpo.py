"""Joint TPO actor: one optimizer update for two authenticated views per recording."""
import argparse
from datetime import timedelta
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_choice_inputs import identity
from multimodal_jev.context_joint_tpo import targets
from multimodal_jev.local_checkpoint import commit_file, commit_json
from multimodal_jev.local_runtime_storage import require_training_storage
from tools.reports.build_tpo_view import read, verified
from tools.reports.joint_tpo_inputs import load, stream
from tools.reports.joint_tpo_reference import score


def publish_recent(wrapped, model, root, step, rank):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from multimodal_jev.context_fsdp_export import canonical_execution
    with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
        if rank==0:
            execution=canonical_execution({n:p.detach().cpu().to(torch.bfloat16).clone()
                for n,p in model.named_parameters() if p.requires_grad})
            path=root/f'activation-{step:04d}.pt'
            commit_file(path,lambda f:torch.save(execution,f))
            commit_json(root/f'activation-{step:04d}.json',dict(update=step,sha256=file_hash(path)))
    dist.barrier()


def run(args):
    root=args.output
    require_training_storage(root)
    plan=read(verified(args.plan,args.joint_sha))
    c,m,task,rows,pop,batches=load(args.parent,args.plan.parent/'targets',plan)
    rank,world,local=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    size=c['batch_config']['per_dp_replica_batch']; micro=plan['microbatch']
    if world!=len(plan['training_devices']) or world*size!=plan['global_batch'] or size%micro:
        raise ValueError('joint batch geometry differs')
    devices=plan['inference_devices']
    fresh=plan.get('initialization') == 'pre_task_policy'
    expected_steps=2 if args.window=='finite' else plan['updates']
    anchor_role='finite-anchor' if fresh and args.window=='finite' else 'anchor'
    anchor=root.parent/anchor_role/f'reference-{rank%len(devices)}'
    seal=read(anchor/'receipt.json')
    if (seal['source_policy']!=plan['source_policy'] or seal['target_sha']!=plan['target_sha'] or
            seal['card']!=devices[rank%len(devices)] or seal['joint_sha']!=args.joint_sha):
        raise ValueError('unbound fixed reference')
    anchors=torch.load(verified(anchor/f'rank-{rank}.pt',seal['rank_files'][str(rank)]),map_location='cpu',weights_only=True)
    if fresh and (seal['updates']!=expected_steps or len(anchors)!=expected_steps):
        raise ValueError('fixed anchor does not cover the selected window')
    from multimodal_jev.model_provider import resolve_provider
    provider = resolve_provider(c)
    provider.initialize_actor_runtime()
    from omegaconf import OmegaConf
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from multimodal_jev.context_native_actor import build_actor
    from multimodal_jev.context_tpo_actor import MODE,EXTENSION,pack
    from multimodal_jev.context_fsdp_export import canonical_execution
    from multimodal_jev.context_checkpoint_policy import export_checkpoint_policy
    from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope
    e=c['view_execution']; source=c['source_policy_id']
    torch.set_num_threads(e['actor_threads'])
    torch.npu.set_device(local)
    torch.manual_seed(e['actor_seed']); torch.npu.manual_seed_all(e['actor_seed'])
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('hccl',timeout=timedelta(seconds=e['collective_timeout_seconds']))
    try:
        out=root/f'rank-{rank}'; out.mkdir()
        transition = provider.adapter_transition(c,fresh)
        base,processor,model,wrapped,opt,execution,handle=provider.load_actor(
            c,torch.device('npu',local),out,e,transition)
        validate_trainable_scope([(n,p) for n,p in wrapped.named_parameters() if p.requires_grad])
        collate=provider.collation(base,processor,model)
        cfg=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,
            ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,
            entropy_from_logits_with_chunking=False,use_torch_compile=False,fsdp_config={'dtype':'bfloat16'},
            calculate_sum_pi_squared=False,entropy_checkpointing=False,ppo_mini_batch_size=2*size,
            ppo_micro_batch_size_per_gpu=micro,ppo_epochs=1,use_kl_loss=False,calculate_entropy=False,
            entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,policy_loss={'loss_mode':MODE},
            global_batch_info={},grad_clip=e['grad_clip'],jev_policy_id=source,jev_joint_tpo=plan['objective'],
            jev_context_parity=c['behavior_bounds']))
        actor=build_actor(cfg,wrapped,opt,objective=EXTENSION,collation=collate,forward=provider.forward)
        end=2 if args.window=='finite' else len(batches)
        bounds=c['behavior_bounds']['probability_max_abs']
        policy=source
        with stream(c,task,rows,pop,batches,processor,collate,provider=provider,rank=rank,start=0,stop=end) as consume:
            for step in range(end):
                now=time.monotonic()
                warmup=plan.get('warmup_updates',0)
                if warmup:
                    for group in opt.param_groups:
                        group['lr']=e['optimizer']['lr'] * min(1., (step+1)/warmup)
                cfg.jev_policy_id=policy
                n,choice,meta,slots=consume(step,policy)
                items=[]
                for start in range(0,size,micro):
                    items.extend(n[start:start+micro])
                    items.extend(choice[start:start+micro])
                denominator=sum(not x['padding'] for x in batches[step]['slots'])
                recent,seconds=score(root,out,items,rank=rank,step=step,version=step,policy=policy,
                    devices=devices,timeout=e['collective_timeout_seconds'])
                parity=[]
                for group,start in enumerate(range(0,size,micro)):
                    ng=n[start:start+micro]; cg=choice[start:start+micro]
                    fixed=anchors[step][start:start+micro]
                    if any((uid,key,ck)!=(meta[start+i]['id'],identity(ng[i],source),identity(cg[i],source))
                           for i,(uid,key,ck,_,_) in enumerate(fixed)):
                        raise ValueError('fixed reference input differs from paired actor input')
                    a=torch.stack([x[3] for x in fixed]); ca=torch.stack([x[4] for x in fixed])
                    nr=torch.stack(recent[2*group*micro:(2*group+1)*micro])
                    cr=torch.stack(recent[(2*group+1)*micro:(2*group+2)*micro])
                    prior_logit=a if fresh else a.logit()
                    if step==0:
                        a_prob=a.sigmoid() if fresh else a
                        error=max(float((nr.sigmoid()-a_prob).abs().max()),float((cr.sigmoid()-ca.sigmoid()).abs().max()))
                        flips=int(((nr>=0)!=((a>=0) if fresh else (a_prob>=.5))).sum()+((cr>=0)!=(ca>=0)).sum())
                        parity.append(dict(rows=len(ng),max_abs=error,threshold_flips=flips,passed=error<=bounds and flips==0))
                    indices=torch.tensor([s['slate'] for s in slots[start:start+micro]],dtype=torch.long)
                    gold=torch.tensor([x['gold'] for x in ng],dtype=torch.long)
                    qn,qc=targets(nr,cr,prior_logit,ca,indices,gold,plan['objective'])
                    for view,logits,q in ((ng,nr,qn),(cg,cr,qc)):
                        for x,z,target in zip(view,logits.unbind(),q.unbind(),strict=True):
                            x.update(macro_prior=z,log_target=target,global_real=denominator)
                if step==0:
                    commit_json(out/'initial-parity.json',parity[:4])
                    if not all(x['passed'] for x in parity):
                        raise ValueError(f'joint fixed/initial reference differs: {parity}')
                metrics=actor.update_policy(pack(items))
                torch.npu.synchronize()
                joined=[None]*world
                dist.all_gather_object(joined,dict(metrics=metrics,seconds=seconds))
                if step+1<end:
                    publish_recent(wrapped,model,root,step+1,rank)
                    next_policy=read(root/f'activation-{step+1:04d}.json')['sha256']
                else:
                    next_policy=None
                if rank==0:
                    def mean(key):
                        return sum(float(torch.as_tensor(r['metrics'][key],dtype=torch.float64).mean()) for r in joined)/world
                    values={k:mean(k) for k in ('actor/pg_loss','actor/grad_norm','jev/tpo_loss')}
                    for label in ('noul','open'):
                        values[f'jev/{label}_joint_kl']=mean(f'jev/{label}_joint_kl')/2
                    if (not all(math.isfinite(v) and v>0 for v in values.values()) or
                            not math.isclose(values['jev/tpo_loss'],.5*(values['jev/noul_joint_kl']+
                                values['jev/open_joint_kl']),rel_tol=1e-6,abs_tol=1e-8)):
                        raise ValueError(f'invalid joint loss metrics: {values}')
                    commit_json(root/f'update-{step+1:04d}.json',dict(step=step+1,seconds=time.monotonic()-now,
                        learning_rate=opt.param_groups[0]['lr'],
                        real=denominator,task='joint',metrics=values,reference=dict(mode='joint_next_macro',
                        policy_update=step,recent_policy=policy,anchor_policy=source,next_policy=next_policy,
                        macro_slots=size*world,seconds_max=max(r['seconds'] for r in joined),
                        config_sha256=args.joint_sha)))
                    if step:
                        (root/f'activation-{step:04d}.pt').unlink()
                if args.window=='formal' and step+1 in plan.get('intermediate_export_updates',[]):
                    with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
                        if rank==0:
                            weights=canonical_execution({n:p.detach().cpu().to(torch.bfloat16).clone()
                                for n,p in model.named_parameters() if p.requires_grad})
                            export_checkpoint_policy(c['source_policy'],root/f'policy-step-{step+1}',weights,
                                policy_id=source,contract=c['source_contract'],base_id=c['base_id'])
                policy=next_policy
        with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
            if rank==0:
                full=canonical_execution({n:p.detach().cpu().to(torch.bfloat16).clone()
                    for n,p in model.named_parameters() if p.requires_grad})
                joint=export_checkpoint_policy(c['source_policy'],root/'policy',full,policy_id=source,
                    contract=c['source_contract'],base_id=c['base_id'])
                commit_json(root/'result.json',dict(arm='tpo',window=args.window,updates=end,policy=joint['policy_id'],
                    target_manifest_sha256=plan['target_sha'],parent_config_sha256=plan['parent_sha']))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('parent','plan','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--joint-sha',required=True)
    p.add_argument('--window',choices=('finite','formal'),required=True)
    run(p.parse_args())
