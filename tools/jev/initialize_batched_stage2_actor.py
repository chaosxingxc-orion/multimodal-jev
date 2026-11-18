#!/usr/bin/env python3
"""Candidate Stage2 actor initialization; no standalone launch admission."""
from pathlib import Path
import json
from datetime import timedelta
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_batch_training_config import batch_layout


def initialize_stage2_actor(config, root, rank, *, after_initialize):
    if after_initialize is None:
        raise ValueError('explicit candidate Stage2 consumer required')
    layout=batch_layout(config)
    if (config.get('physical_devices')!=[4,5,6,7]
            or sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))
                !=['davinci4','davinci5','davinci6','davinci7']):
        raise ValueError('isolated DP4 candidate devices required')
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import CustomPolicy
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3,configure_actor_profile
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_fsdp import unshared_linear_modules
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.local_checkpoint import commit_file
    from verify_slurp_batched_parity import verify
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('fixture dependency changed: '+p)
    from multimodal_jev.batched_stage2_initialization import authenticate
    recovering='continuous_resume' in config
    recovered=None
    if recovering:
        from multimodal_jev.batched_stage2_resume import authenticate as resume
        recovered=resume(config,verify_parity=verify)
        initialization=recovered['initial']
        gate=dict(recovered,training_root=config['continuous_resume']['training_root'])
    else:
        initialization=authenticate(config,verify_parity=verify)
        gate=dict(initialization,training_root=str(Path(initialization['shard_paths'][0]).parents[2]))
    original=json.loads((Path(gate['training_root'])/'config.json').read_text())
    from multimodal_jev.slurp_instruction_resume import FIELDS
    for field in FIELDS:
        if original[field]!=config[field]:raise ValueError('restore factor differs: '+field)
    if torch.npu.device_count()!=4:raise ValueError('fixed DP4 required')
    torch.set_num_threads(2);torch.npu.set_device(rank);device=torch.device('npu',rank)
    torch.manual_seed(config['seed']);torch.npu.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    initialize_verl_preserving_qwen3();configure_actor_profile(config['runtime_profile'])
    dist.init_process_group('hccl',timeout=timedelta(seconds=900))
    try:
        base_id=checkpoint_identity(config['model_path'])[0]
        if base_id!=config['base_id']:raise ValueError('base identity changed')
        processor,base=load_runtime(config['model_path'],device,'bfloat16')
        adapted=initialize_adapter(base,dict(config,initial_adapter=gate['adapter'])).eval()
        model=NativeAudioSpanModel(adapted.get_base_model()).eval()
        from multimodal_jev.span_batch_candidate_runtime import install_candidate
        candidate_handle,candidate_profile=install_candidate(adapted,model)
        atomic_json(root/f'rank-{rank}'/'candidate-profile.json',candidate_profile)
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
        if recovering:
            from multimodal_jev.batched_stage2_resume import restore
            actual,restored=restore(wrapped,optimizer,recovered,rank=rank)
            commit_file(root/f'rank-{rank}'/'stage2-resumed.pt',lambda stream:torch.save(actual,stream))
        else:
            saved=torch.load(initialization['shard_paths'][rank],map_location='cpu',weights_only=True)
            from multimodal_jev.batched_stage2_initialization import restore_stage_boundary
            restored=restore_stage_boundary(wrapped,optimizer,saved,rank=rank)
        out=root/f'rank-{rank}';atomic_json(out/'fixture-restored.json',restored)
        from probe_simple_sampled_npu import actor_config
        cfg=actor_config(gate['policy_id'])
        cfg.ppo_micro_batch_size_per_gpu=layout['microbatch_per_rank']
        collation=dict(pad_token_id=processor.tokenizer.pad_token_id,
            audio_token_id=base.config.audio_token_id,
            audio_chunk_size=base.config.audio_config.n_window*2,
            audio_infer_window=base.config.audio_config.n_window_infer)
        atomic_json(out/'stage2-source.json',{k:v for k,v in initialization.items() if k!='files'})
        if not recovering:
            initial_state=dict(rank=rank,world_size=4,updates=0,stage='simple_stage2_initial',
                names=[n for n,p in wrapped.named_parameters() if p.requires_grad],
                parameters={n:p.detach().cpu().clone() for n,p in wrapped.named_parameters() if p.requires_grad},
                optimizer=optimizer.state_dict(),adapter_id=gate['adapter_id'],
                reference_policy_id=initialization['reference_policy_id'],schedule_sha256=config['schedule_sha256'],
                next_update=0,candidate_profile=config['candidate_profile'],batch_layout=layout,
                batch_order=config['batch_order'])
            commit_file(out/'stage2-initial.pt',lambda stream:torch.save(initial_state,stream))
        if recovering:
            from multimodal_jev.batched_stage2_recovery_policy import check_restored_policy
            check_restored_policy(wrapped,processor,config,recovered,root,
                                  rank=rank,device=device,collation=collation)
        after_initialize(config['stage2_initialization'],config=config,root=root,rank=rank,
            processor=processor,adapted=adapted,wrapped=wrapped,optimizer=optimizer,
            actor_config=cfg,base_id=base_id,
            adapter_id=gate['adapter_id'],device=device,execution_dtypes=executions,
            initialization=initialization,collation=collation,
            **(dict(recovery=recovered) if recovering else {}))
        atomic_json(out/'stage2-execution.json',dict(execution_dtypes=executions))
    finally:
        dist.destroy_process_group()
