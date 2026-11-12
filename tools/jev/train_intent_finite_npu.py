#!/usr/bin/env python3
"""One accepted-policy finite-support RL update on four BF16 LoRA/FSDP ranks.

Native verl owns accumulation and Adam stepping. Each subsequent update needs
fresh accepted behavior probabilities and exact restored FP32 master/Adam.
Continuation requires an independently accepted predecessor and exact shards.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import time
from multimodal_jev.baseline import atomic_json,file_hash


def validate_config(config):
    from multimodal_jev.intent_optimizer import optimizer_kwargs
    optimizer_kwargs(config)
    if (config['arm'] not in ('D','Q','P') or config['global_requests']!=8
            or type(config['step'])!=int or config['step']<0
            or config.get('initial_optimizer_checkpoint')
            or config['lora_execution_dtype']!='bfloat16'):
        raise ValueError('accepted-policy DP4/BF16 update required; no implicit optimizer reset')
    if config['step']==0:
        if config.get('previous_root') or config.get('previous_suite'):
            raise ValueError('step zero requires fresh optimizer without predecessor')
    elif not config.get('previous_root') or not config.get('previous_suite'):
        raise ValueError('continuation requires accepted predecessor and optimizer')
    import math
    if type(config['lr']) not in (int,float) or not math.isfinite(config['lr']) or config['lr']<=0:
        raise ValueError('positive finite learning rate required')
    if type(config['seed'])!=int:raise ValueError('fixed seed required')


def run(config,root,out,rank):
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision,ShardingStrategy
    from torch.distributed.fsdp.wrap import CustomPolicy
    from safetensors.torch import load_file,save_file
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3,configure_actor_profile
    from multimodal_jev.span_fsdp import unshared_linear_modules
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.intent_actor_inputs import prepare
    from probe_fsdp_lora_npu import digest
    validate_config(config)
    for path,h in config['pins'].items():
        if file_hash(path)!=h:raise ValueError('pinned training asset changed: '+path)
    if torch.npu.device_count()!=4:raise ValueError('four isolated NPUs required')
    torch.set_num_threads(2);torch.npu.set_device(rank);device=torch.device('npu',rank)
    torch.manual_seed(config['seed']);torch.npu.manual_seed_all(config['seed'])
    initialize_verl_preserving_qwen3();configure_actor_profile(config['runtime_profile'])
    from omegaconf import OmegaConf
    from multimodal_jev.span_finite_actor import make_global_batch,build_actor,admit_batch
    from multimodal_jev.span_native_actor import ACTOR_SHA256
    dist.init_process_group('hccl',timeout=timedelta(seconds=300))
    begin=time.monotonic()
    progress=lambda phase:atomic_json(out/'progress.json',dict(phase=phase,updated_unix=time.time()))
    try:
        lineage=[None]
        if config['step']>0:
            if rank==0:
                try:
                    from multimodal_jev.intent_resume import authenticate
                    lineage[0]=dict(error=None,evidence=authenticate(config))
                    atomic_json(root/'resume-lineage.json',lineage[0]['evidence'])
                except Exception as exc:lineage[0]=dict(error=repr(exc))
            dist.broadcast_object_list(lineage,src=0)
            if lineage[0]['error']:raise ValueError('continuation admission failed: '+lineage[0]['error'])
        progress('load_accepted_policy')
        base_id=checkpoint_identity(config['model_path'])[0]
        policy_id=base_id+':'+checkpoint_identity(config['initial_adapter'])[0]
        processor,base=load_runtime(config['model_path'],device,'bfloat16')
        adapted=initialize_adapter(base,config).eval();model=NativeAudioSpanModel(adapted.get_base_model()).eval()
        # Authenticate and encode once on rank zero, then transport equal complete rows.
        payload=[None];chunks=None
        if rank==0:
            try:
                items,selection=prepare(config,processor,policy_id,config['step'])
                batch=make_global_batch(items);chunks=list(batch.split(2))
                payload[0]=dict(error=None,selection=selection)
                atomic_json(root/'selection.json',selection)
                atomic_json(root/'supervision.json',[dict(request_id=i['request_id'],targets=i['targets'].tolist(),
                    supervised=i['supervised'].tolist(),span_weights=i['span_weights'].tolist(),request_weight=i['request_weight']) for i in items])
            except Exception as exc:payload[0]=dict(error=repr(exc))
        dist.broadcast_object_list(payload,src=0)
        if payload[0]['error']:raise ValueError('controller admission failed: '+payload[0]['error'])
        incoming=[None];dist.scatter_object_list(incoming,chunks,src=0);batch=incoming[0]
        selection=payload[0]['selection'];local_ids=selection['recording_ids'][rank*2:rank*2+2]
        atomic_json(out/'selection.json',dict(**selection,rank=rank,local_recording_ids=local_ids))
        for p in model.parameters():
            if p.requires_grad:p.data=p.data.float()
        executions={}
        def hook(name):
            def check(module,args):
                executions[name]=[str(module.weight.dtype),str(args[0].dtype)]
                if executions[name]!=['torch.bfloat16','torch.bfloat16']:raise ValueError('BF16 LoRA execution required')
            return check
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Linear) and ('lora_A.' in name or 'lora_B.' in name):module.register_forward_pre_hook(hook(name))
        units=unshared_linear_modules(model)
        wrapped=FSDP(model,auto_wrap_policy=CustomPolicy(lambda m:m in units),device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,sync_module_states=True,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,cast_forward_inputs=True,keep_low_precision_grads=False),limit_all_gathers=True)
        parameters=[p for p in wrapped.parameters() if p.requires_grad]
        if any(p.dtype!=torch.float32 for p in parameters):raise ValueError('FP32 master shards required')
        from multimodal_jev.intent_optimizer import optimizer_kwargs
        optimizer=torch.optim.AdamW(parameters,**optimizer_kwargs(config))
        restored=None
        if config['step']>0:
            from multimodal_jev.span_fsdp_resume import restore_local_shard
            path=Path(config['previous_root'])/'run'/f'rank-{rank}'/'optimizer-local.pt'
            state=torch.load(path,map_location='cpu',weights_only=True)
            restored=restore_local_shard(wrapped,optimizer,state,rank=rank,world_size=4,updates=config['step'])
            torch.save(dict(names=[n for n,p in wrapped.named_parameters() if p.requires_grad],
                parameters={n:p.detach().cpu().clone() for n,p in wrapped.named_parameters() if p.requires_grad},
                optimizer=optimizer.state_dict()),out/'restored-local.pt')
            atomic_json(out/'optimizer-resume.json',restored)
        actor_config=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,
            ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,
            entropy_from_logits_with_chunking=False,use_torch_compile=False,
            fsdp_config={'dtype':'bfloat16'},calculate_sum_pi_squared=False,entropy_checkpointing=False,
            ppo_mini_batch_size=2,ppo_micro_batch_size_per_gpu=1,ppo_epochs=1,use_kl_loss=False,
            calculate_entropy=False,entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,
            policy_loss={'loss_mode':'vanilla'},clip_ratio=.2,clip_ratio_low=None,clip_ratio_high=None,
            clip_ratio_c=3.,global_batch_info={},grad_clip=1.,jev_policy_id=policy_id))
        admission=admit_batch(batch,actor_config)
        frozen={n:digest(p) for n,p in wrapped.named_parameters() if not p.requires_grad}
        before={n:digest(p) for n,p in wrapped.named_parameters() if p.requires_grad}
        actor=build_actor(actor_config,wrapped,optimizer)
        progress('full_candidate_forward_admission')
        computed=actor.compute_log_prob(batch)['log_probs'].detach().cpu()
        old=batch.batch['old_log_probs'].clone()
        if not torch.isfinite(computed).all():raise ValueError('nonfinite candidate probability')
        atomic_json(out/'before.json',dict(old=old.tolist(),actor=computed.tolist(),admission=admission))
        progress('native_finite_support_update')
        metrics=actor.update_policy(batch)
        if not torch.equal(old,batch.batch['old_log_probs']):raise ValueError('behavior replaced')
        norms=metrics['actor/grad_norm']
        if len(norms)!=1 or not torch.isfinite(torch.tensor(norms)).all() or norms[0]<=0:raise ValueError('one finite nonzero gradient required')
        for p in parameters:
            state=optimizer.state[p]
            if p.dtype!=torch.float32 or not torch.isfinite(p).all() or state['step'].item()!=config['step']+1:raise ValueError('finite master/step required')
            for key in ('exp_avg','exp_avg_sq'):
                if state[key].dtype!=torch.float32 or not torch.isfinite(state[key]).all():raise ValueError('finite FP32 Adam required')
        if frozen!={n:digest(p) for n,p in wrapped.named_parameters() if not p.requires_grad}:raise ValueError('base changed')
        changed=torch.tensor([sum(before[n]!=digest(p) for n,p in wrapped.named_parameters() if p.requires_grad)],device=device)
        dist.all_reduce(changed)
        if not changed.item():raise ValueError('no adapter change')
        torch.save(dict(world_size=4,rank=rank,updates=config['step']+1,names=[n for n,p in wrapped.named_parameters() if p.requires_grad],
            parameters={n:p.detach().cpu().clone() for n,p in wrapped.named_parameters() if p.requires_grad},
            optimizer=optimizer.state_dict()),out/'optimizer-local.pt')
        after=[]
        with torch.no_grad():
            for uid,item in zip(local_ids,batch.non_tensor_batch['multi_modal_inputs'],strict=True):
                inputs={k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in item['inputs'].items()}
                probabilities=wrapped(inputs,item['positions'].to(device),item['candidates'])[1]
                if any(not torch.isfinite(p).all() for p in probabilities):raise ValueError('nonfinite updated policy')
                after.append(dict(id=uid,previous_request_id=item['request_id'],candidate_logprobs=[p[0].cpu().tolist() for p in probabilities]))
        atomic_json(out/'after.json',after)
        progress('export_bf16_adapter')
        with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
            tensors={n.replace('_fsdp_wrapped_module.','').replace('.default.','.'):
                p.detach().to(device='cpu',dtype=torch.bfloat16).clone()
                for n,p in adapted.named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
        original=load_file(str(Path(config['initial_adapter'])/'adapter_model.safetensors'))
        if tensors.keys()!=original.keys() or any(tensors[n].shape!=v.shape for n,v in original.items()):raise ValueError('export layout differs')
        hashes={n:digest(p) for n,p in tensors.items()};peers=[None]*4;dist.all_gather_object(peers,hashes)
        if any(peer!=hashes for peer in peers):raise ValueError('rank exports differ')
        if rank==0:
            dest=root/'adapter';dest.mkdir();save_file(tensors,str(dest/'adapter_model.safetensors'))
            # safetensors may create mode 0600 despite the shared directory ACL.
            # The host controller must hash the export before independent audit.
            (dest/'adapter_model.safetensors').chmod(0o644)
            shutil.copyfile(Path(config['initial_adapter'])/'adapter_config.json',dest/'adapter_config.json')
        dist.barrier();adapter_id=checkpoint_identity(root/'adapter')[0]
        atomic_json(out/'result.json',dict(passed=True,rank=rank,world_size=4,updates=config['step']+1,arm=config['arm'],restored=restored,
            base_id=base_id,initial_policy_id=policy_id,adapter_id=adapter_id,recording_ids=local_ids,
            admission=admission,metrics=metrics,execution_dtypes=executions,changed_shards=int(changed.item()),
            frozen_base_unchanged=True,cross_rank_export_exact=True,actor_sha256=ACTOR_SHA256,
            seconds=time.monotonic()-begin,scope='One finite-support native verl DP4/FSDP BF16 learning update. Independent export/backend verification still required; no continuous trainer or task-gain claim.'))
    finally:dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=4 or int(os.environ['LOCAL_RANK'])!=rank:raise ValueError('single-node DP4 required')
    out=args.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    try:
        run(json.loads(args.config.read_text()),args.output,out,rank)
        atomic_json(out/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(out/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
