#!/usr/bin/env python3
"""Four-rank real-audio Span FSDP update through the pinned native verl actor.

Fresh FSDP optimizer from a pinned accepted BF16 adapter, not single-rank
optimizer continuation. Saved vLLM probabilities stay bound to exact inputs.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import time

from multimodal_jev.baseline import atomic_json, file_hash
from train_span_rl_step import prepare_actor


def run(config, root, out, rank):
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import CustomPolicy
    from multimodal_jev.span_fsdp import unshared_linear_modules
    from safetensors.torch import load_file, save_file
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3, configure_actor_profile, verify_suite_admission
    from multimodal_jev.span_actor_update import record_rollout, admit_actor
    from multimodal_jev.policy_identity import checkpoint_identity
    from probe_fsdp_lora_npu import digest
    torch.set_num_threads(2)
    if torch.npu.device_count()!=4:raise ValueError('four isolated NPUs required')
    torch.npu.set_device(rank);device=torch.device('npu',rank)
    initialize_verl_preserving_qwen3();configure_actor_profile(config['runtime_profile'])
    from omegaconf import OmegaConf
    from multimodal_jev.span_native_actor import make_batch, build_actor, ACTOR_SHA256
    from multimodal_jev.span_native_dp import admit_rank_batch
    for path,sha in config['pins'].items():
        if file_hash(path)!=sha:raise ValueError('pinned asset changed: '+path)
    verify_suite_admission(config)
    if config.get('initial_optimizer_checkpoint'):raise ValueError('fresh FSDP optimizer pilot only')
    config={**config,'batch_index':config['batch_index']+rank}
    begin=time.monotonic()
    progress=lambda phase:atomic_json(out/'progress.json',dict(phase=phase,updated_unix=time.time()))
    dist.init_process_group('hccl',timeout=timedelta(seconds=300))
    try:
        progress('prepare_audio_actor')
        adapted,model,base_id,policy_id,prepared,_=prepare_actor(config,out,Path(config['initial_adapter']),device=device)
        suite=json.loads(Path(config['runtime_acceptance']).read_text())
        rows={r['request_id']:r for r in suite['records']}
        rollouts=[]
        for i,p in enumerate(prepared):
            row=rows[p['request_id']]
            if row['request_identity']!=p['identity']:raise ValueError('exact accepted input differs')
            behavior=tuple(torch.tensor([v],dtype=torch.float32) for v in row['candidate_logprobs'])
            rollouts.append(record_rollout(policy_id,[p['request_id']],behavior,32,
                torch.Generator().manual_seed(config['seed']+rank*100+i)))
        for p in model.parameters():
            if p.requires_grad:p.data=p.data.float()
        executions={}
        def hook(name):
            def inspect(m,args):
                executions[name]=[str(m.weight.dtype),str(args[0].dtype)]
                if m.weight.dtype!=torch.bfloat16 or args[0].dtype!=torch.bfloat16:
                    raise ValueError('LoRA compute dtype differs: '+name)
            return inspect
        for name,m in model.named_modules():
            if isinstance(m,torch.nn.Linear) and ('lora_A.' in name or 'lora_B.' in name):
                m.register_forward_pre_hook(hook(name))
        units=unshared_linear_modules(model)
        wrapped=FSDP(model,auto_wrap_policy=CustomPolicy(lambda module: module in units),device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,sync_module_states=True,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,cast_forward_inputs=True,keep_low_precision_grads=False),
            limit_all_gathers=True)
        parameters=[p for p in wrapped.parameters() if p.requires_grad]
        if any(p.dtype!=torch.float32 for p in parameters):raise ValueError('FP32 FSDP LoRA masters required')
        optimizer=torch.optim.AdamW(parameters,lr=config['lr'],weight_decay=0.,foreach=False,fused=False)
        def forward(p):
            inputs={k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in p['inputs'].items()}
            return wrapped(inputs,p['positions'].to(device),p['candidates'])[1]
        progress('frozen_probability_admission')
        audits=[]
        with torch.no_grad():
            for p,rollout in zip(prepared,rollouts,strict=True):
                prob=forward(p)
                audits.extend(admit_actor(rollout,prob,policy_id,[p['request_id']]))
        atomic_json(out/'admission.json',dict(passed=True,audits=audits,execution_dtypes=executions))
        dist.barrier()
        frozen={n:digest(p) for n,p in wrapped.named_parameters() if not p.requires_grad}
        before={n:digest(p) for n,p in wrapped.named_parameters() if p.requires_grad}
        # Keep one complete request per microbatch: one FSDP forward/backward
        # before the next request, matching the independently verified probe.
        items=[]
        for p,rollout in zip(prepared,rollouts,strict=True):
            target=torch.tensor(p['targets'])
            rewards=(rollout.actions==target).float().mean(-1)
            items.append(dict(inputs=p['inputs'],positions=p['positions'],candidates=p['candidates'],
                              rollout=rollout,rewards=rewards))
        actor_config=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,
            ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,
            entropy_from_logits_with_chunking=False,use_torch_compile=False,
            fsdp_config={'dtype':'bfloat16'},calculate_sum_pi_squared=False,entropy_checkpointing=False,
            ppo_mini_batch_size=64,ppo_micro_batch_size_per_gpu=32,ppo_epochs=1,use_kl_loss=False,
            calculate_entropy=False,entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,
            policy_loss={'loss_mode':'vanilla'},clip_ratio=.2,clip_ratio_low=None,clip_ratio_high=None,
            clip_ratio_c=3.,global_batch_info={},grad_clip=1.,jev_group_size=32,jev_policy_id=policy_id))
        batch=make_batch(items,32)
        partition=admit_rank_batch(batch,actor_config)
        actor=build_actor(actor_config,wrapped,optimizer)
        computed=actor.compute_log_prob(batch)['log_probs'].detach().cpu()
        if not torch.isfinite(computed).all():raise ValueError('nonfinite native actor probabilities')
        old=batch.batch['old_log_probs'].clone()
        progress('native_verl_actor_update')
        metrics=actor.update_policy(batch)
        if not torch.equal(old,batch.batch['old_log_probs']):raise ValueError('behavior probabilities replaced')
        norms=metrics['actor/grad_norm']
        if len(norms)!=1 or not torch.isfinite(torch.tensor(norms)).all() or norms[0]<=0:
            raise ValueError('one finite nonzero native optimizer step required')
        norm=norms[0]
        if any(optimizer.state[p].get('step',torch.tensor(-1)).item()!=1 for p in parameters):
            raise ValueError('native Adam step one required')
        for state in optimizer.state.values():
            for key in ('exp_avg','exp_avg_sq'):
                if state[key].dtype!=torch.float32 or not torch.isfinite(state[key]).all():
                    raise ValueError('finite FP32 native Adam moments required')
        atomic_json(out/'native-actor.json',dict(actor_sha256=ACTOR_SHA256,partition=partition,
            metrics=metrics,old_probabilities_unchanged=True,
            compute_maximum_old_logprob_error=float((computed-old).abs().max()),
            update_loop='DataParallelPPOActor.update_policy',execution_dtype='bfloat16'))
        if frozen!={n:digest(p) for n,p in wrapped.named_parameters() if not p.requires_grad}:
            raise ValueError('frozen base changed')
        changed=sum(before[n]!=digest(p) for n,p in wrapped.named_parameters() if p.requires_grad)
        total_changed=torch.tensor([changed],device=device);dist.all_reduce(total_changed)
        if not total_changed.item():raise ValueError('no LoRA update')
        torch.save(dict(world_size=4,rank=rank,updates=1,
            names=[n for n,p in wrapped.named_parameters() if p.requires_grad],
            parameters={n:p.detach().cpu().clone() for n,p in wrapped.named_parameters() if p.requires_grad},
            optimizer=optimizer.state_dict()),out/'optimizer-local.pt')
        expected=[]
        with torch.no_grad():
            for p in prepared:
                prob=forward(p)
                expected.append(dict(recording_id=p['bundle']['id'],input_request=p['bundle']['request'].model_input(),
                    candidate_logprobs=[v[0].cpu().tolist() for v in prob]))
        atomic_json(out/'after.json',expected)
        progress('export_adapter')
        # use_orig_params retains the live PEFT parameter handles through FSDP.
        with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
            tensors={n.replace('_fsdp_wrapped_module.','').replace('.default.','.'):
                p.detach().to(device='cpu',dtype=torch.bfloat16).clone()
                for n,p in adapted.named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
        original=load_file(str(Path(config['initial_adapter'])/'adapter_model.safetensors'))
        if tensors.keys()!=original.keys() or any(tensors[n].shape!=v.shape for n,v in original.items()):
            raise ValueError('PEFT export names/shapes changed')
        hashes={n:digest(v) for n,v in tensors.items()}
        peers=[None]*4;dist.all_gather_object(peers,hashes)
        if any(h!=hashes for h in peers):raise ValueError('ranks disagree on exported adapter')
        if rank==0:
            dest=root/'adapter';dest.mkdir()
            save_file(tensors,str(dest/'adapter_model.safetensors'))
            shutil.copyfile(Path(config['initial_adapter'])/'adapter_config.json',dest/'adapter_config.json')
            for p in dest.iterdir():p.chmod(0o644)
            dest.chmod(0o777)
        dist.barrier()
        adapter_id=checkpoint_identity(root/'adapter')[0]
        atomic_json(out/'result.json',dict(status='passed',rank=rank,world_size=4,
            optimizer_updates=1,optimizer_start='fresh FP32 AdamW from pinned BF16 adapter; not resume',
            recordings=[p['bundle']['id'] for p in prepared],base_id=base_id,initial_policy_id=policy_id,
            adapter_id=adapter_id,admission=audits,execution_dtypes=executions,
            gradient_norm=float(norm),native_actor_metrics=metrics,native_actor_sha256=ACTOR_SHA256,
            partition=partition,changed_shards=int(total_changed.item()),
            frozen_base_unchanged=True,cross_rank_export_exact=True,seconds=time.monotonic()-begin,
            scope='Four-rank real-audio FSDP forward admission against accepted TP1 vLLM, one native verl actor update. '
            'Independent exported-adapter reload/vLLM verification still required; not Ray trainer or TP4 acceptance.'))
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=4 or rank!=int(os.environ['LOCAL_RANK']):raise ValueError('four-rank single node required')
    out=args.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    try:run(json.loads(args.config.read_text()),args.output,out,rank)
    except BaseException as exc:
        atomic_json(out/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(out/'terminal.json',dict(exit_code=0))
