#!/usr/bin/env python3
"""Restore accepted CE checkpoint, then one real-audio sampled native RL update.

The prefix adapter is an engineering fixture, not the final Stage1 reference.
Accepted actual vLLM probabilities supply behavior. Fresh rollout, updated-policy
parity and restart acceptance remain separate; this tool never promotes training.
"""
import argparse,json,os,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def actor_config(policy_id):
    from omegaconf import OmegaConf
    from multimodal_jev.span_hybrid_actor import MODE
    return OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,ulysses_sequence_parallel_size=1,
            use_dynamic_bsz=False,use_prefix_grouper=False,entropy_from_logits_with_chunking=False,use_torch_compile=False,
            fsdp_config={'dtype':'bfloat16'},calculate_sum_pi_squared=False,entropy_checkpointing=False,
            ppo_mini_batch_size=32,ppo_micro_batch_size_per_gpu=1,ppo_epochs=1,use_kl_loss=False,
            calculate_entropy=False,entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,
            policy_loss={'loss_mode':MODE},clip_ratio=.2,clip_ratio_low=None,clip_ratio_high=None,clip_ratio_c=3.,
            global_batch_info={},grad_clip=1.,jev_policy_id=policy_id))


def restore_accepted_fixture(config, root, rank, *, restore_only=False, after_restore=None):
    return load_accepted_actor(config, root, rank, restore_only=restore_only, after_restore=after_restore)


def initialize_stage2_actor(config, root, rank, *, after_initialize):
    """Use the same admitted actor construction with a real Stage1 boundary."""
    if after_initialize is None:
        raise ValueError('explicit Stage2 initialization consumer required')
    return load_accepted_actor(config, root, rank, restore_only=True,
                               after_restore=after_initialize, stage2=True)


def resume_stage2_actor(config, root, rank, *, after_resume):
    if after_resume is None:
        raise ValueError('explicit formal recovery consumer required')
    return load_accepted_actor(config, root, rank, restore_only=True,
                               after_restore=after_resume, stage2=True, resuming=True)


def load_accepted_actor(config, root, rank, *, restore_only=False, after_restore=None, stage2=False, resuming=False):
    """Load accepted bytes and exact Adam, without stochastic CE reconstruction."""
    if after_restore is not None and not restore_only:
        raise ValueError("continuation callback requires accepted updated-state restore")
    if resuming and (not stage2 or not restore_only):
        raise ValueError('formal Stage2 recovery requires explicit restore mode')
    import shutil
    import torch
    import torch_npu
    import torch.distributed as dist
    from datetime import timedelta
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import CustomPolicy
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3,configure_actor_profile
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_fsdp import unshared_linear_modules
    from multimodal_jev.span_fsdp_resume import restore_local_shard
    from multimodal_jev.intent_resident_replay import compare_state
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.local_checkpoint import commit_file,seal_adapter
    from multimodal_jev.slurp_resident_admission import check_restored_policy
    from verify_slurp_simple_parity import verify
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('fixture dependency changed: '+p)
    initialization=None;recovered=None
    if resuming:
        from multimodal_jev.simple_stage2_resume import authenticate
        recovered=authenticate(config,verify_parity=verify)
        initialization=recovered['initial']
        gate_path=config['stage2_initialization']['gate']
        gate=dict(recovered,training_root=config['continuous_resume']['training_root'])
        updates=recovered['completed_updates']
    elif stage2:
        from multimodal_jev.simple_stage2_initialization import authenticate
        initialization=authenticate(config,verify_parity=verify)
        gate_path=config['stage2_initialization']['gate']
        # authenticate already recomputed full-selection parity; retain its
        # accepted identity without another full verifier invocation.
        gate=dict(initialization,training_root=str(Path(initialization['shard_paths'][0]).parents[2]))
        updates=634
    else:
        gate_path=config['sampled_probe']['behavior_gate']
        gate=verify(gate_path)
        updates=1 if restore_only else 2
        if gate['updates']!=updates or gate['requests']!=256:raise ValueError('accepted prefix256 and exact stage cursor required')
        if restore_only and gate['sampled_resident_parity'] is None:
            raise ValueError('independently accepted updated RL policy required')
    original=json.loads((Path(gate['training_root'])/'config.json').read_text())
    from multimodal_jev.slurp_instruction_resume import FIELDS
    for field in FIELDS:
        if original[field]!=config[field]:raise ValueError('restore factor differs: '+field)
    if restore_only and not stage2 and original['sampled_probe']['algorithm']!=config['sampled_probe']['algorithm']:
        raise ValueError('restored RL algorithm differs')
    if torch.npu.device_count()!=4:raise ValueError('fixed DP4 required')
    torch.set_num_threads(2);torch.npu.set_device(rank);device=torch.device('npu',rank)
    torch.manual_seed(config['seed']);torch.npu.manual_seed_all(config['seed'])
    initialize_verl_preserving_qwen3();configure_actor_profile(config['runtime_profile'])
    dist.init_process_group('hccl',timeout=timedelta(seconds=900))
    try:
        base_id=checkpoint_identity(config['model_path'])[0]
        if base_id!=config['base_id']:raise ValueError('base identity changed')
        processor,base=load_runtime(config['model_path'],device,'bfloat16')
        adapted=initialize_adapter(base,dict(config,initial_adapter=gate['adapter'])).eval()
        model=NativeAudioSpanModel(adapted.get_base_model()).eval()
        for p in model.parameters():
            if p.requires_grad:p.data=p.data.float()
        executions={}
        def hook(name):
            def check(module,args):
                executions[name]=[str(module.weight.dtype),str(args[0].dtype)]
                if executions[name]!=['torch.bfloat16','torch.bfloat16']:raise ValueError('BF16 execution required')
            return check
        for n,m in model.named_modules():
            if isinstance(m,torch.nn.Linear) and ('lora_A.' in n or 'lora_B.' in n):m.register_forward_pre_hook(hook(n))
        units=unshared_linear_modules(model)
        wrapped=FSDP(model,auto_wrap_policy=CustomPolicy(lambda m:m in units),device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,sync_module_states=True,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32,
                cast_forward_inputs=True,keep_low_precision_grads=False),limit_all_gathers=True)
        optimizer=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=config['lr'],
            betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
        relative=(f'loop-rank-{rank}/checkpoint-{updates:04d}.pt' if resuming else
                  f'rank-{rank}/checkpoint-0634.pt' if stage2 else
                  f'rl-rank-{rank}/checkpoint-0001.pt' if restore_only else f'rank-{rank}/checkpoint-0002.pt')
        saved=torch.load(Path(gate['training_root'])/'run'/relative,map_location='cpu',weights_only=True)
        if resuming:
            from multimodal_jev.simple_stage2_resume import restore
            actual,restored=restore(wrapped,optimizer,recovered,rank=rank)
        elif stage2:
            from multimodal_jev.simple_stage2_initialization import restore_stage_boundary
            restored=restore_stage_boundary(wrapped,optimizer,saved,rank=rank)
        else:
            restored=restore_local_shard(wrapped,optimizer,saved,rank=rank,world_size=4,updates=updates)
        out=root/f'rank-{rank}';atomic_json(out/'fixture-restored.json',restored)
        if resuming:
            commit_file(out/'stage2-resumed.pt',lambda stream:torch.save(actual,stream))
            atomic_json(out/'stage2-resume-source.json',{k:v for k,v in recovered.items()
                                                       if k not in ('states','initial')})
            after_restore(config['stage2_initialization'],config=config,root=root,rank=rank,
                processor=processor,adapted=adapted,wrapped=wrapped,optimizer=optimizer,
                actor_config=actor_config(recovered['policy_id']),base_id=base_id,
                adapter_id=gate['adapter_id'],device=device,execution_dtypes=executions,
                initialization=initialization,recovery=recovered)
            atomic_json(out/'stage2-execution.json',dict(execution_dtypes=executions))
            return
        if stage2:
            # The continuous collector checks behavior parity on the complete
            # first learning batch before the first optimizer update. Full9226
            # HF/vLLM selection parity is a prerequisite, not repeated here.
            atomic_json(out/'stage2-source.json',{k:v for k,v in initialization.items() if k!='files'})
            initial_state=dict(rank=rank,world_size=4,updates=0,stage='simple_stage2_initial',
                names=[n for n,p in wrapped.named_parameters() if p.requires_grad],
                parameters={n:p.detach().cpu().clone() for n,p in wrapped.named_parameters() if p.requires_grad},
                optimizer=optimizer.state_dict(),adapter_id=gate['adapter_id'],
                reference_policy_id=initialization['reference_policy_id'],schedule_sha256=config['schedule_sha256'],
                next_update=0)
            commit_file(out/'stage2-initial.pt',lambda stream:torch.save(initial_state,stream))
            after_restore(config['stage2_initialization'],config=config,root=root,rank=rank,
                processor=processor,adapted=adapted,wrapped=wrapped,optimizer=optimizer,
                actor_config=actor_config(initialization['policy_id']),base_id=base_id,
                adapter_id=gate['adapter_id'],device=device,execution_dtypes=executions,
                initialization=initialization)
            atomic_json(out/'stage2-execution.json',dict(execution_dtypes=executions))
            return
        check_restored_policy(wrapped,processor,dict(config,previous_gate=gate_path),out,rank=rank,device=device)
        actual=dict(saved,parameters={n:p.detach().cpu().clone() for n,p in wrapped.named_parameters() if p.requires_grad},optimizer=optimizer.state_dict())
        compared=compare_state(saved,actual,rank=rank,updates=updates)
        commit_file(out/f'checkpoint-{updates:04d}.pt',lambda stream:torch.save(actual,stream))
        if restore_only:
            atomic_json(out/'result.json',dict(passed=True,rank=rank,updates=updates,
                base_id=base_id,adapter_id=gate['adapter_id'],exact_state=compared,
                execution_dtypes=executions,checkpoint_source=str(Path(gate['training_root'])/'run'/relative),
                scope='Fresh-process fixed-DP4 restore of accepted RL master/Adam and full256 resident/HF parity; no optimizer update or training promotion.'))
            if after_restore is not None:
                after_restore(config['sampled_probe'],config=config,root=root,rank=rank,
                    processor=processor,adapted=adapted,wrapped=wrapped,optimizer=optimizer,
                    actor_config=actor_config(base_id+':'+gate['adapter_id']),base_id=base_id,
                    adapter_id=gate['adapter_id'],device=device,execution_dtypes=executions)
            return
        if rank==0:
            dest=root/'adapter-step0002';dest.mkdir()
            for name in ('adapter_config.json','adapter_model.safetensors'):shutil.copyfile(Path(gate['adapter'])/name,dest/name)
            seal_adapter(dest)
        dist.barrier()
        cfg=actor_config(base_id+':'+gate['adapter_id'])
        sampled_update(config['sampled_probe'],config=config,root=root,rank=rank,processor=processor,adapted=adapted,
            wrapped=wrapped,optimizer=optimizer,actor_config=cfg,base_id=base_id,
            adapter_id=checkpoint_identity(root/'adapter-step0002')[0],device=device,execution_dtypes=executions)
    finally:dist.destroy_process_group()


def sampled_update(plan, **ctx):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from safetensors.torch import save_file
    from multimodal_jev.intent_resident_replay import compare_state
    from multimodal_jev.simple_audio_batch import compile_batch
    from multimodal_jev.simple_sampled_rollout import actor_batch
    from multimodal_jev.span_hybrid_actor import make_batch,build_actor
    from multimodal_jev.span_training_session import SpanTrainingSession
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.slurp_training import content_hash
    from multimodal_jev.span_adapter_export import adapter_key
    from multimodal_jev.policy_identity import checkpoint_identity, compare_adapter_artifacts
    from multimodal_jev.local_checkpoint import commit_file,commit_json,seal_adapter
    from multimodal_jev.span_fsdp_resume import restore_local_shard
    from probe_fsdp_lora_npu import digest
    from verify_slurp_simple_parity import verify
    config=ctx['config'];root=ctx['root'];rank=ctx['rank'];model=ctx['wrapped']
    out=root/f'rl-rank-{rank}';out.mkdir(exist_ok=False)
    def gather(value):
        peers=[None]*4;dist.all_gather_object(peers,value);return peers
    def collective(name,fn):
        error=None;result=None
        try:result=fn()
        except Exception as exc:error=repr(exc)
        errors=gather(error)
        if any(errors):raise ValueError(name+': '+str(errors))
        return result
    verified=collective('accepted fixture',lambda:verify(plan['behavior_gate']))
    if verified['updates']!=2 or verified['requests']!=256:
        raise ValueError('exact accepted prefix endpoint required')
    reference_root=Path(verified['training_root'])
    artifacts=collective('exact prefix weights and config',lambda:compare_adapter_artifacts(
        reference_root/'run/adapter-step0002',root/'adapter-step0002'))
    if (artifacts['reference_id']!=verified['adapter_id']
            or artifacts['candidate_id']!=ctx['adapter_id']):
        raise ValueError('prefix artifact binding differs')
    reference_config=json.loads((reference_root/'config.json').read_text())
    from multimodal_jev.slurp_instruction_resume import FIELDS
    for field in FIELDS:
        if reference_config[field]!=config[field]:
            raise ValueError('accepted fixture training factor differs: '+field)
    if config.get('microbatch_per_rank',1)!=reference_config.get('microbatch_per_rank',1):
        raise ValueError('accepted fixture microbatch differs')
    def exact_state():
        expected=torch.load(reference_root/'run'/f'rank-{rank}'/'checkpoint-0002.pt',map_location='cpu',weights_only=True)
        actual=dict(rank=rank,world_size=4,updates=2,names=[n for n,p in model.named_parameters() if p.requires_grad],
                    parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
                    optimizer=ctx['optimizer'].state_dict())
        return compare_state(expected,actual,rank=rank,updates=2)
    atomic_json(out/'prefix-exact.json',collective('prefix master and Adam',exact_state))
    # The accepted behavior retains its original ID. Reconstructed actor bytes
    # and full FP32 state were checked above; retain the distinct artifact IDs.
    atomic_json(out/'prefix-artifact-equivalence.json',artifacts)
    policy=ctx['base_id']+':'+verified['adapter_id']
    chunks=None;receipt=None
    def prepare():
        nonlocal chunks,receipt
        if rank!=0:return
        rows,task,_=load(config['release'],config['release_sha256'],'learning',purpose='optimize')
        schedule=json.loads(Path(config['epoch_file']).read_text())
        if content_hash(schedule['batches'])!=schedule['report']['batches_sha256']:
            raise ValueError('frozen schedule changed')
        entries,renderer,compiled=compile_batch({r['id']:r for r in rows},task,schedule['batches'][0],
            ctx['processor'],policy_id=policy,global_batch=128)
        wanted={r[4] for r in renderer}
        behavior=json.loads((Path(plan['behavior_gate'])/'run/vllm/result.json').read_text())
        behavior=dict(behavior,records=[r for r in behavior['records'] if r['request_id'] in wanted])
        items,receipt=actor_batch(entries,behavior,behavior,policy_id=policy,reference_policy_id=policy,
                                  seed=config['seed'],update=0)
        batch=make_batch(items,stage='task',coefficient=1.,reduction='real_request_mean',
                         algorithm=plan['algorithm'],kl_coefficient=.01,reference_policy_id=policy)
        chunks=list(batch.split(32))
        atomic_json(root/'rl-inputs.json',dict(compiled=compiled,sampling=receipt))
    collective('prepare full audio batch',prepare)
    incoming=[None];dist.scatter_object_list(incoming,chunks,src=0);batch=incoming[0]
    # Stage boundary deliberately resets Adam; prefix masters stay exact.
    ctx['optimizer'].zero_grad(set_to_none=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],
        betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    cfg=ctx['actor_config'];cfg.jev_policy_id=policy;cfg.jev_advantage_algorithm=plan['algorithm']
    cfg.jev_kl_coefficient=.01;cfg.jev_reference_policy_id=policy
    actor=build_actor(cfg,model,optimizer)
    session=SpanTrainingSession(actor,policy_id=policy,completed_updates=0,gather=gather)
    frozen={n:digest(p) for n,p in model.named_parameters() if not p.requires_grad}
    torch.npu.synchronize();start=time.monotonic();metrics=session.update(batch);torch.npu.synchronize()
    seconds=time.monotonic()-start
    def check_update():
        if frozen!={n:digest(p) for n,p in model.named_parameters() if not p.requires_grad}:
            raise ValueError('frozen base changed')
        if len(metrics['actor/grad_norm'])!=1 or not 0<float(metrics['actor/grad_norm'][0])<float('inf'):
            raise ValueError('finite nonzero gradient required')
        for p in optimizer.param_groups[0]['params']:
            state=optimizer.state[p]
            if (p.dtype!=torch.float32 or not torch.isfinite(p).all() or state['step'].item()!=1
                    or any(state[k].dtype!=torch.float32 or not torch.isfinite(state[k]).all() for k in ('exp_avg','exp_avg_sq'))):
                raise ValueError('finite FP32 master and fresh-stage Adam required')
    collective('RL update state',check_update)
    def probabilities():
        records=[]
        with torch.no_grad():
            for payload in batch.non_tensor_batch['multi_modal_inputs']:
                inputs={k:v.to(ctx['device'],dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                        for k,v in payload['inputs'].items()}
                logp=model(inputs,payload['positions'].to(ctx['device']),payload['candidates'])[1]
                records.append(dict(request_id=payload['request_id'],candidate_logprobs=[v[0].cpu().tolist() for v in logp]))
        return records
    after=collective('updated resident probabilities',probabilities)
    atomic_json(out/'after.json',dict(records=after))
    with FSDP.summon_full_params(model,writeback=False,rank0_only=False):
        tensors={adapter_key(n):p.detach().cpu().to(torch.bfloat16).clone()
                 for n,p in ctx['adapted'].named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
    hashes={n:digest(p) for n,p in tensors.items()}
    if any(p!=hashes for p in gather(hashes)):raise ValueError('cross-rank adapter export differs')
    dest=root/'rl-adapter-step0001'
    def export():
        if rank==0:
            dest.mkdir();save_file(tensors,str(dest/'adapter_model.safetensors'))
            ctx['adapted'].peft_config['default'].save_pretrained(dest)
            (dest/'adapter_model.safetensors').chmod(0o644);seal_adapter(dest)
    collective('export',export)
    adapter_id=checkpoint_identity(dest)[0];session.publish(ctx['base_id']+':'+adapter_id)
    state=dict(rank=rank,world_size=4,updates=1,stage='sampled_rl_engineering',algorithm=plan['algorithm'],
        names=[n for n,p in model.named_parameters() if p.requires_grad],
        parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
        optimizer=optimizer.state_dict(),adapter_id=adapter_id,reference_policy_id=policy)
    collective('persist shard',lambda:commit_file(out/'checkpoint-0001.pt',lambda stream:torch.save(state,stream)))
    def commit():
        if rank==0:
            commit_json(root/'rl-checkpoint-0001-committed.json',dict(adapter_id=adapter_id,world_size=4,
                files={str(p.relative_to(root)):file_hash(p) for p in [root/f'rl-rank-{r}/checkpoint-0001.pt' for r in range(4)]}))
    collective('commit all shards',commit)
    optimizer.zero_grad(set_to_none=True)
    restored_optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],
        betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    def restore():
        saved=torch.load(out/'checkpoint-0001.pt',map_location='cpu',weights_only=True)
        return restore_local_shard(model,restored_optimizer,saved,rank=rank,world_size=4,updates=1)
    restored=collective('local master Adam reload',restore)
    recovered=collective('reloaded resident probabilities',probabilities)
    def check_recovered():
        if recovered!=after:raise ValueError('resident probabilities changed on reload')
    collective('exact recovered probabilities',check_recovered)
    atomic_json(out/'restored.json',dict(**restored,
        exact_resident_probabilities=True,requests=len(recovered),records=recovered,
        scope='Actual serialized shard and fresh Adam reload into same resident model with exact post-update probabilities; no fresh-process FSDP or independent HF/vLLM acceptance.'))
    atomic_json(out/'result.json',dict(passed=True,rank=rank,algorithm=plan['algorithm'],updates=1,
        base_id=ctx['base_id'],adapter_id=adapter_id,reference_policy_id=policy,metrics=metrics,
        admission=actor.last_admission,seconds=seconds,phase=session.phase,execution_dtypes=ctx['execution_dtypes'],
        scope='One real-audio BF16 native DP4 sampled RL+CE+KL update using accepted actual vLLM prefix behavior. Export awaits independent reload/parity; no final Stage1 reference or efficacy claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text());plan=config['sampled_probe'];rank=int(os.environ['RANK'])
    if (plan['algorithm'] not in ('grpo_v1','rloo_v1') or config['updates']!=2 or config['global_batch']!=128
            or config.get('initial_adapter') or int(os.environ['WORLD_SIZE'])!=4 or int(os.environ['LOCAL_RANK'])!=rank):
        raise ValueError('fresh two-step fixture then one sampled DP4 update required')
    from multimodal_jev.local_runtime_storage import require_training_storage
    require_training_storage(a.output)
    out=a.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    try:
        if plan.get('fixture_initialization')!='exact_checkpoint_restore':
            raise ValueError('explicit accepted-checkpoint restore required; no stochastic reconstruction')
        restore_accepted_fixture(config,a.output,rank)
        atomic_json(out/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(out/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
