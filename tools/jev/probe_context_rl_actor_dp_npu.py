#!/usr/bin/env python3
"""Actual audio GRPO/RLOO single-update acceptance from authenticated vLLM behavior.

Each algorithm starts from the same updated joint policy and fresh Adam. Replay
checks determinism; this is not a multi-step RL curve or online policy switching.
"""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash


def run(c,root):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision,ShardingStrategy
    from torch.distributed.fsdp.wrap import CustomPolicy
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3,configure_actor_profile,SHORT_AUDIO_PROFILE
    # Must precede all verl imports; preserve HF equations in the same process.
    initialize_verl_preserving_qwen3();configure_actor_profile(SHORT_AUDIO_PROFILE)
    from omegaconf import OmegaConf
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.context_decision_model import ContextDecisionModel
    from multimodal_jev.context_decision_head import ContextDecisionHead
    from multimodal_jev.context_runtime_profile import PROFILE,install_actor
    from multimodal_jev.context_fsdp import context_fsdp_units
    from multimodal_jev.context_policy_snapshot import load_policy
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    from multimodal_jev.context_native_actor import MODE,build_actor,make_batch
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.context_candidate_task import compile_candidates,bind_candidate_targets
    from multimodal_jev.context_decision_contract import encode_context_request
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.local_checkpoint import commit_file
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.span_request_identity import tensor_identity
    from multimodal_jev.context_optimizer_audit import audit_adam_shards
    from multimodal_jev.policy_identity import checkpoint_identity
    require_training_storage(root)
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if (world!=4 or c['global_batch']!=128 or c['microbatch']!=8 or
            c['runtime_profile']!=PROFILE or len(c['audio'])!=256 or
            len({a['id'] for a in c['audio']})!=256 or c['physical_devices']!=[4,5,6,7] or
            sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci4','davinci5','davinci6','davinci7']):
        raise ValueError('isolated DP4 global128 micro8 learning256 required')
    for path,digest in c['pins'].items():
        if file_hash(path)!=digest:raise ValueError('dependency changed: '+path)
    torch.set_num_threads(2);torch.npu.set_device(local);device=torch.device('npu',local)
    torch.manual_seed(317);torch.npu.manual_seed_all(317);torch.use_deterministic_algorithms(True)
    dist.init_process_group('hccl',timeout=timedelta(seconds=900))
    out=root/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    progress=lambda phase,**kw:atomic_json(out/'progress.json',dict(phase=phase,unix=time.time(),**kw))
    try:
        progress('load')
        if checkpoint_identity(c['model'])[0]!=c['base_id']:raise ValueError('base changed')
        bundle=read_execution_bundle(c['source_policy'],expected_policy_id=c['source_policy_id'],
            expected_contract=c['source_contract'],expected_base_id=c['base_id'])
        processor=Qwen3ASRProcessor.from_pretrained(bundle['processor'],local_files_only=True)
        _,base=load_runtime(c['model'],device,'bfloat16')
        adapted=initialize_adapter(base,dict(lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16')).eval()
        head=ContextDecisionHead(1024,8,1024,.1).to(device=device,dtype=torch.bfloat16).eval()
        model=ContextDecisionModel(adapted.get_base_model().model,head,bundle['contract']['token']['id']).eval()
        load_policy(c['source_policy'],model,expected_policy_id=c['source_policy_id'],
            expected_contract=c['source_contract'],expected_base_id=c['base_id'])
        runtime_handle,profile=install_actor(adapted,model,PROFILE)
        atomic_json(out/'runtime.json',profile)
        # Entire head is one FSDP unit. MHA calls out_proj functionally.
        units=context_fsdp_units(model)
        for p in model.parameters():
            if p.requires_grad:p.data=p.data.float()
        executions={}
        def capture_dtype(name):
            def hook(module,args):
                if args[0].dtype!=torch.bfloat16 or any(p.dtype!=torch.bfloat16 for p in module.parameters()):
                    raise ValueError('BF16 execution required: '+name)
                executions[name]='bfloat16'
            return hook
        # These hooks are registered after FSDP so all-gather/casting occurs first.
        wrapped=FSDP(model,auto_wrap_policy=CustomPolicy(lambda m:m in units),device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,sync_module_states=True,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,cast_forward_inputs=True,keep_low_precision_grads=False),limit_all_gathers=True)
        model.decision_head.module.register_forward_pre_hook(capture_dtype('head'))
        for n,m in model.named_modules():
            if type(m) is torch.nn.Linear and ('lora_A.' in n or 'lora_B.' in n):
                m.register_forward_pre_hook(capture_dtype(n))
        parameters=[p for p in wrapped.parameters() if p.requires_grad]
        def optimizer():return torch.optim.AdamW(parameters,lr=1e-4,betas=(.9,.999),eps=1e-8,
            weight_decay=0.,foreach=False,fused=False)
        opt=optimizer()
        cfg=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,
            ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,
            entropy_from_logits_with_chunking=False,use_torch_compile=False,fsdp_config={'dtype':'bfloat16'},
            calculate_sum_pi_squared=False,entropy_checkpointing=False,ppo_mini_batch_size=32,
            ppo_micro_batch_size_per_gpu=8,ppo_epochs=1,use_kl_loss=False,calculate_entropy=False,
            entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,policy_loss={'loss_mode':MODE},
            clip_ratio=.2,clip_ratio_low=None,clip_ratio_high=None,clip_ratio_c=3.,global_batch_info={},grad_clip=1.,
            jev_policy_id=c['source_policy_id'],jev_context_parity=c['behavior_bounds']))
        options=dict(stage='task',ranking_coefficient=.3,probability_coefficient=.3,algorithm='grpo_v1',
            kl_coefficient=.01,reference_policy_id=c['source_policy_id'])
        cfg.jev_context_objective=options
        collation=dict(decision_token_id=model.decision_token_id,pad_token_id=processor.tokenizer.pad_token_id,
            audio_token_id=base.config.audio_token_id,audio_chunk_size=base.config.audio_config.n_window*2,
            audio_infer_window=base.config.audio_config.n_window_infer)
        progress('prepare_learning256')
        prepared=[None]
        if rank==0:
            try:
                rows,task,_=load(c['release'],c['release_sha256'],'learning',purpose='optimize')
                population={r['id']:r for r in rows};items=[];receipts=[]
                candidates=[dict(id=x['id'],definition=x['description']) for x in task['schema']['candidates']]
                from multimodal_jev.context_rollout_batch import bind_rollout
                behavior_rows=json.loads(Path(c['behavior_records']).read_text())['records']
                behavior={r['id']:r for r in behavior_rows}
                if len(behavior)!=len(behavior_rows) or set(behavior)!={a['id'] for a in c['audio']}:
                    raise ValueError('complete engine behavior population required')
                for asset in c['audio']:
                    row=population[asset['id']]
                    if row['audio']!=asset['audio'] or row['audio_sha256']!=asset['sha256'] or file_hash(row['audio'])!=asset['sha256']:
                        raise ValueError('learning audio identity differs')
                    request=compile_candidates(row['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
                    inputs,positions=encode_context_request(processor,request,load_audio(row['audio']),
                        decision_token_id=model.decision_token_id,audio_token_id=base.config.audio_token_id,max_tokens=4096)
                    inputs={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in inputs.items()}
                    target=bind_candidate_targets(request,semantics='exclusive_event',exclusive_event=row['intent'])
                    item=dict(request_id=row['id'],policy_id=c['source_policy_id'],inputs=inputs,positions=positions,
                        labels=torch.tensor([target['labels']],dtype=torch.bool),observed=torch.ones(1,60,dtype=torch.bool),
                        request_weight=1.,real_request=True)
                    bound=bind_rollout(behavior[row['id']],inputs,positions,policy_id=c['source_policy_id'],group_size=16,seed=317+len(items))
                    item.update(behavior=bound['behavior'],actions=bound['actions'],reference_policy=bound['behavior'],reference_policy_id=c['source_policy_id'])
                    items.append(item);receipts.append(dict(id=row['id'],target=target,rollout=bound['receipt'],
                        inputs={k:tensor_identity(v) for k,v in inputs.items()},positions=positions.tolist()))
                atomic_json(root/'input-receipts.json',receipts)
                prepared=[dict(error=None)]
            except Exception as exc:prepared=[dict(error=repr(exc))]
        dist.broadcast_object_list(prepared,src=0)
        if prepared[0]['error']:raise ValueError(prepared[0]['error'])
        chunks=[]
        for algorithm in ('grpo_v1','rloo_v1'):
            current=dict(options,algorithm=algorithm)
            parts=None
            if rank==0:
                complete=make_batch(items[:128],**current)
                audit=[{k:p[k] for k in ('request_id','policy_id','labels','observed','behavior','actions','old_joint','advantages','rewards')}
                       for p in complete.non_tensor_batch['multi_modal_inputs']]
                commit_file(root/(algorithm+'-groups.pt'),lambda stream:torch.save(audit,stream))
                parts=list(complete.split(32))
            target=[None];dist.scatter_object_list(target,parts,src=0);chunks.append(target[0])
        names=[n for n,p in wrapped.named_parameters() if p.requires_grad]
        frozen={n:tensor_identity(p.detach().cpu()) for n,p in wrapped.named_parameters() if not p.requires_grad}
        def snapshot():
            return dict(names=names,parameters={n:p.detach().cpu().clone() for n,p in wrapped.named_parameters() if p.requires_grad},
                optimizer=copy.deepcopy(opt.state_dict()),rank=rank,world_size=4)
        def restore(state):
            nonlocal opt
            if state['names']!=names or state['rank']!=rank:raise ValueError('checkpoint shard differs')
            with torch.no_grad():
                for n,p in wrapped.named_parameters():
                    if p.requires_grad:p.copy_(state['parameters'][n].to(device))
            opt=optimizer();opt.load_state_dict(state['optimizer'])
        before=snapshot();records=[]
        commit_file(out/'initial.pt',lambda stream:torch.save(before,stream))
        for execution,batch_index in enumerate((0,1,0)):
            if execution:
                restore(torch.load(out/'initial.pt',map_location='cpu',weights_only=True))
            step=1
            cfg.jev_policy_id=c['source_policy_id']
            cfg.jev_context_objective=dict(options,algorithm=('grpo_v1','rloo_v1')[batch_index])
            batch=copy.deepcopy(chunks[batch_index])
            for p in batch.non_tensor_batch['multi_modal_inputs']:p['policy_id']=cfg.jev_policy_id
            actor=build_actor(cfg,wrapped,opt,collation=collation)
            progress('native_update',execution=execution,step=step)
            torch.npu.reset_peak_memory_stats();torch.npu.synchronize();started=time.monotonic()
            metrics=actor.update_policy(batch)
            torch.npu.synchronize();seconds=time.monotonic()-started
            named=[(n,p) for n,p in wrapped.named_parameters() if p.requires_grad]
            atomic_json(out/f'optimizer-layout-{execution}.json',dict(
                execution=execution,seconds=seconds,metrics=metrics,
                parameters={n:dict(shape=list(p.shape),dtype=str(p.dtype),grad_none=p.grad is None,
                    fields=sorted(opt.state.get(p,{}))) for n,p in named}))
            optimizer_audit=audit_adam_shards(named,opt,step)
            if frozen!={n:tensor_identity(p.detach().cpu()) for n,p in wrapped.named_parameters() if not p.requires_grad}:
                raise ValueError('frozen backbone changed')
            state=snapshot()
            if execution==0:
                commit_file(out/'grpo.pt',lambda stream:torch.save(state,stream));reference=state
            elif execution==1:
                commit_file(out/'rloo.pt',lambda stream:torch.save(state,stream))
            else:
                commit_file(out/'grpo-replay.pt',lambda stream:torch.save(state,stream))
                for n,value in reference['parameters'].items():
                    if not torch.equal(value,state['parameters'][n]):raise ValueError('resumed parameter differs: '+n)
                for key,fields in reference['optimizer']['state'].items():
                    for name,value in fields.items():
                        if not torch.equal(value.cpu(),state['optimizer']['state'][key][name].cpu()):
                            raise ValueError('resumed optimizer differs')
            changed={n:bool(not torch.equal(v,before['parameters'][n])) for n,v in state['parameters'].items()}
            records.append(dict(execution=execution,algorithm=('grpo_v1','rloo_v1')[batch_index],behavior_parity=actor.last_behavior_parity,behavior_parity_history=actor.behavior_parity_history,step=step,seconds=seconds,metrics=metrics,admission=actor.last_admission,
                peak_allocated=torch.npu.max_memory_allocated(),changed=changed,optimizer_audit=optimizer_audit))
            atomic_json(out/'updates.json',records)
        # Compare assembled BF16 head/LoRA on all ranks, independent of shard layout.
        with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
            full={n:p.detach().cpu().to(torch.bfloat16).clone() for n,p in model.named_parameters() if p.requires_grad}
            identity={n:tensor_identity(v) for n,v in full.items()}
            peers=[None]*4;dist.all_gather_object(peers,identity)
            if any(p!=identity for p in peers):raise ValueError('cross-rank full execution differs')
            if rank==0:commit_file(root/'full-execution.pt',lambda stream:torch.save(full,stream))
        atomic_json(out/'result.json',dict(passed=True,updates=records,replay_exact=True,executions=executions,
            full_bf16_hashes=identity,scope='Actual audio GRPO/RLOO from authenticated offline vLLM behavior; one update per algorithm, exact GRPO replay; no online policy refresh or task gain.'))
        dist.barrier()
        if rank==0:atomic_json(root/'result.json',dict(passed=True,ranks=4,global_batch=128,microbatch=8,
            prepared_learning_queries=256,updated_learning_queries=128,updates_per_algorithm=1,replay_executions=1,training_dispatch_allowed=False))
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    run(json.loads(a.config.read_text()),a.output)
