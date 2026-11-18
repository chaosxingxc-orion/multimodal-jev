#!/usr/bin/env python3
"""Isolated cached-input DP4 admission prefix, not dispatched production.

Derived from train_slurp_batched_instruction_candidate.py SHA256 d1b6d63f577374dcf74f9b0852a02dde1e304eb696a69ee0b1d58a05e1028924.
Only input service and its cleanup differ; actor/export/restore remain identical.

Candidate DP4 true-batch instruction prefix on the authenticated full schedule.

Native verl owns backward/accumulation/optimizer. Checkpoints are immutable;
continuation requires independent state/reload/parity and full processor evidence.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash


def run(config, root, rank, *, after_training=None):
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import CustomPolicy
    from safetensors.torch import save_file
    from omegaconf import OmegaConf
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3, configure_actor_profile
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_fsdp import unshared_linear_modules
    from multimodal_jev.span_contract import encode_span_request, span_labels
    from multimodal_jev.slurp_official_training import load, example
    from multimodal_jev.slurp_training import content_hash
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_hybrid_actor import MODE, make_batch, build_actor
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.span_adapter_export import adapter_key
    from multimodal_jev.slurp_instruction_resume import load_full_schedule, schedule_cursor
    from multimodal_jev.span_batched_instruction_resume import authenticate, check_restored_policy
    from multimodal_jev.span_fsdp_resume import restore_local_shard
    from multimodal_jev.local_checkpoint import commit_file, commit_json, seal_adapter
    from probe_fsdp_lora_npu import digest
    if config.get('input_pipeline') != 'cached_batch_prefetch_v1' or config['updates'] != 2:
        raise ValueError('isolated two-update cached-input admission required')
    from multimodal_jev.local_runtime_storage import check_paths
    check_paths({'encoded_audio_cache': config['encoded_audio_cache']})
    input_pipeline = None
    continuing=bool(config.get('initial_adapter'));start_update=config.get('start_update',0)
    if config.get('task_mode') == 'direct_final' and config.get('loss_reduction') != 'real_request_mean':
        raise ValueError('simple baseline requires population-weighted real-request reduction')
    from multimodal_jev.span_batch_training_config import batch_layout, reorder_global_items
    layout = batch_layout(config)
    if (torch.npu.device_count() != 4
            or config['lora_execution_dtype'] != 'bfloat16'):
        raise ValueError('BF16 DP4 required')
    if not continuing and (config['updates']!=2 or start_update!=0):
        raise ValueError('fresh training requires the two-update admission prefix')
    lineage=authenticate(config) if continuing else None
    checkpoint_interval=config.get('checkpoint_interval',1)
    if type(checkpoint_interval) is not int or checkpoint_interval<1:
        raise ValueError('positive checkpoint interval required')
    frozen_audit = config.get('frozen_audit', 'every_update')
    if frozen_audit not in ('every_update', 'checkpoint'):
        raise ValueError('explicit frozen parameter audit cadence required')
    for path, expected in config['pins'].items():
        if file_hash(path) != expected:
            raise ValueError('frozen instruction dependency changed: '+path)
    if (config.get('physical_devices') != [4,5,6,7]
            or sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))
            != ['davinci4','davinci5','davinci6','davinci7']):
        raise ValueError('isolated physical4..7 candidate prefix required')
    for key in ('runtime_gate','runtime_collection'):
        if config['pins'].get(config[key]) != file_hash(config[key]):
            raise ValueError('pinned candidate runtime evidence required')
    gate=json.loads(Path(config['runtime_gate']).read_text())
    collected=json.loads(Path(config['runtime_collection']).read_text())
    if (gate.get('reference') != 'full_candidate_HF'
            or gate.get('reference_comparison_passed') is not True
            or len(gate.get('original_hf_comparison',[])) != 256
            or collected['state']['Status'] != 'exited' or collected['state']['ExitCode'] != 0):
        raise ValueError('completed candidate HF/vLLM engineering gate required')
    torch.set_num_threads(2);torch.npu.set_device(rank);device=torch.device('npu',rank)
    torch.manual_seed(config['seed']);torch.npu.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    initialize_verl_preserving_qwen3();configure_actor_profile(config['runtime_profile'])
    dist.init_process_group('hccl',timeout=timedelta(seconds=900))
    out=root/f'rank-{rank}';start=time.monotonic()
    progress=lambda phase,**kw:atomic_json(out/'progress.json',dict(phase=phase,updated_unix=time.time(),**kw))
    try:
        progress('load_model')
        base_id=checkpoint_identity(config['model_path'])[0]
        if base_id!=config['base_id']:raise ValueError('base identity changed')
        processor,base=load_runtime(config['model_path'],device,'bfloat16')
        adapted=initialize_adapter(base,config).eval();model=NativeAudioSpanModel(adapted.get_base_model()).eval()
        from multimodal_jev.span_batch_candidate_runtime import install_candidate
        candidate_handle,candidate_profile=install_candidate(adapted,model)
        atomic_json(out/'candidate-profile.json',candidate_profile)
        # Preserve initial BF16 LoRA independently of subsequent optimizer masters.
        initial_tensors={adapter_key(n):p.detach().cpu().to(torch.bfloat16).clone()
                         for n,p in adapted.named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
        if rank==0 and not continuing:
            initial=root/'adapter-step0000';initial.mkdir()
            save_file(initial_tensors,str(initial/'adapter_model.safetensors'))
            adapted.peft_config['default'].save_pretrained(initial)
            (initial/'adapter_model.safetensors').chmod(0o644)
            seal_adapter(initial)
        dist.barrier();initial_id=checkpoint_identity(config['initial_adapter'] if continuing else root/'adapter-step0000')[0]
        if continuing and initial_id!=lineage['adapter_id']:
            raise ValueError('loaded predecessor identity differs')
        for p in model.parameters():
            if p.requires_grad:p.data=p.data.float()
        executions={}
        def hook(name):
            def verify(module,args):
                executions[name]=[str(module.weight.dtype),str(args[0].dtype)]
                if executions[name]!=['torch.bfloat16','torch.bfloat16']:
                    raise ValueError('BF16 LoRA execution required')
            return verify
        for n,m in model.named_modules():
            if isinstance(m,torch.nn.Linear) and ('lora_A.' in n or 'lora_B.' in n):m.register_forward_pre_hook(hook(n))
        units=unshared_linear_modules(model)
        wrapped=FSDP(model,auto_wrap_policy=CustomPolicy(lambda m:m in units),device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,sync_module_states=True,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32,
                                          cast_forward_inputs=True,keep_low_precision_grads=False),limit_all_gathers=True)
        parameters=[p for p in wrapped.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=config['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
        if continuing:
            state=torch.load(Path(config['previous_root'])/'run'/f'rank-{rank}'/f'checkpoint-{start_update:04d}.pt',map_location='cpu',weights_only=True)
            restored=restore_local_shard(wrapped,optimizer,state,rank=rank,world_size=4,updates=start_update)
            atomic_json(out/'restored.json',dict(**restored,lineage=lineage))
        local=layout['local_batch']
        cfg=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,ulysses_sequence_parallel_size=1,
            use_dynamic_bsz=False,use_prefix_grouper=False,entropy_from_logits_with_chunking=False,use_torch_compile=False,
            fsdp_config={'dtype':'bfloat16'},calculate_sum_pi_squared=False,entropy_checkpointing=False,
            ppo_mini_batch_size=local,ppo_micro_batch_size_per_gpu=layout['microbatch_per_rank'],ppo_epochs=1,use_kl_loss=False,
            calculate_entropy=False,entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,
            policy_loss={'loss_mode':MODE},clip_ratio=.2,clip_ratio_low=None,clip_ratio_high=None,clip_ratio_c=3.,
            global_batch_info={},grad_clip=1.,jev_policy_id=base_id+':'+initial_id))
        from multimodal_jev.span_batched_hybrid_actor import build_batched_actor
        collation=dict(
            pad_token_id=processor.tokenizer.pad_token_id,audio_token_id=base.config.audio_token_id,
            audio_chunk_size=base.config.audio_config.n_window*2,
            audio_infer_window=base.config.audio_config.n_window_infer)
        actor=build_batched_actor(cfg,wrapped,optimizer,collation=collation)
        frozen={n:digest(p) for n,p in wrapped.named_parameters() if not p.requires_grad}
        data_payload=[None];population=None;batches=None;task=None;epoch_updates=None
        if rank==0:
            try:
                rows,task,release=load(config['release'],config['release_sha256'],'learning',purpose='optimize')
                population={r['id']:r for r in rows}
                batches,epoch_updates,schedule_report=load_full_schedule(config)
                if schedule_report['unique_recordings']!=len(rows):
                    raise ValueError('full population differs')
                from transformers import Qwen3ASRProcessor
                from multimodal_jev.encoded_audio_cache import EncodedAudioCache
                from multimodal_jev.cached_span_preparation import CachedSpanPreparer
                from multimodal_jev.cached_instruction_batches import CachedInstructionBatches
                # Bind all authenticated source/processor pins in this candidate.
                preparer=CachedSpanPreparer(
                    lambda: Qwen3ASRProcessor.from_pretrained(config['model_path'],local_files_only=True),
                    EncodedAudioCache(config['encoded_audio_cache']), processor_sha256=content_hash(config['pins']))
                input_pipeline=CachedInstructionBatches(batches,population,task,preparer,
                    start_update=start_update,stop_update=start_update+config['updates'],
                    context_limit=config['context_limit'],global_batch=config['global_batch'])
                data_payload[0]=dict(error=None,recordings=len(rows),schedule=schedule_report,epoch_updates=epoch_updates)
            except Exception as exc:data_payload[0]=dict(error=repr(exc))
        dist.broadcast_object_list(data_payload,src=0)
        if data_payload[0]['error']:raise ValueError(data_payload[0]['error'])
        epoch_updates=data_payload[0]['epoch_updates']
        atomic_json(out/'population.json',data_payload[0]);step_metrics=[]
        if continuing and (state['epoch'],state['next_batch'])!=schedule_cursor(start_update,epoch_updates):
            raise ValueError('checkpoint next epoch/batch differs from full schedule')
        if continuing:
            progress('restored_policy_admission')
            check_restored_policy(wrapped,processor,dict(config,candidate_collation=collation),out,rank=rank,device=device)
            # A read-only admission must preserve the exact recovered master state.
            for name,p in wrapped.named_parameters():
                if p.requires_grad and not torch.equal(p.detach().cpu(),state['parameters'][name]):
                    raise ValueError('admission changed restored FP32 master')
        for step in range(start_update,start_update+config['updates']):
            step_begin=time.monotonic()
            progress('prepare_global_batch',step=step)
            payload=[None];chunks=None;input_timing=None
            if rank==0:
                try:
                    items,metadata=input_pipeline.consume(step,cfg.jev_policy_id)
                    input_timing=dict(input_pipeline.last_timing)
                    pack_begin=time.monotonic()
                    items,permutation=reorder_global_items(items,config)
                    chunks=list(make_batch(items,stage='instruction',
                        reduction=config.get('loss_reduction','legacy_weight_mean')).split(local))
                    input_timing['pack_seconds']=time.monotonic()-pack_begin
                    metadata_begin=time.monotonic()
                    atomic_json(root/f'batch-{step:04d}.json',dict(policy_id=cfg.jev_policy_id,records=metadata,rank_indices=permutation))
                    input_timing['metadata_write_seconds']=time.monotonic()-metadata_begin
                    payload[0]=dict(error=None)
                except Exception as exc:payload[0]=dict(error=repr(exc))
            broadcast_begin=time.monotonic()
            dist.broadcast_object_list(payload,src=0)
            broadcast_seconds=time.monotonic()-broadcast_begin
            if payload[0]['error']:raise ValueError(payload[0]['error'])
            scatter_begin=time.monotonic()
            incoming=[None];dist.scatter_object_list(incoming,chunks,src=0);batch=incoming[0]
            scatter_seconds=time.monotonic()-scatter_begin
            prepared_seconds=time.monotonic()-step_begin
            progress('native_actor_update',step=step);torch.npu.synchronize();begin=time.monotonic()
            metrics=actor.update_policy(batch);torch.npu.synchronize();seconds=time.monotonic()-begin
            audit_begin=time.monotonic()
            if not all(p.dtype==torch.float32 and torch.isfinite(p).all() for p in parameters):
                raise ValueError('nonfinite or nonFP32 masters')
            norms=metrics['actor/grad_norm']
            if len(norms)!=1 or not torch.isfinite(torch.tensor(norms)).all() or norms[0]<=0:
                raise ValueError('finite nonzero gradient required')
            for p in parameters:
                s=optimizer.state[p]
                if s['step'].item()!=step+1 or any(s[k].dtype!=torch.float32 or not torch.isfinite(s[k]).all() for k in ('exp_avg','exp_avg_sq')):
                    raise ValueError('optimizer state differs')
            next_epoch,next_batch=schedule_cursor(step+1,epoch_updates)
            save_now=((step+1)%checkpoint_interval==0 or next_batch==0
                      or step+1==start_update+config['updates'])
            check_frozen = frozen_audit == 'every_update' or save_now
            if check_frozen and {n:digest(p) for n,p in wrapped.named_parameters() if not p.requires_grad}!=frozen:
                raise ValueError('frozen base changed')
            record=dict(update=step+1,seconds=seconds,metrics=metrics,admission=actor.last_admission,
                peak_memory_allocated=torch.npu.max_memory_allocated(),
                peak_memory_reserved=torch.npu.max_memory_reserved(),batch_layout=layout,
                preparation_seconds=prepared_seconds,audit_seconds=time.monotonic()-audit_begin,
                input_pipeline_timing=input_timing,broadcast_seconds=broadcast_seconds,
                scatter_seconds=scatter_seconds,
                frozen_hash_checked=check_frozen,frozen_audit=frozen_audit)
            if not save_now:
                record['total_step_seconds']=time.monotonic()-step_begin
                step_metrics.append(record);atomic_json(out/f'update-{step+1:04d}.json',record)
                cfg.jev_policy_id=base_id+':'+initial_id+f':instruction-update{step+1}'
                continue
            export_begin=time.monotonic();progress('export_checkpoint',step=step)
            dest=root/f'adapter-step{step+1:04d}'
            with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
                tensors={adapter_key(n):p.detach().cpu().to(torch.bfloat16).clone()
                         for n,p in adapted.named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
            hashes={n:digest(p) for n,p in tensors.items()};peers=[None]*4;dist.all_gather_object(peers,hashes)
            if any(v!=hashes for v in peers):raise ValueError('cross-rank BF16 exports differ')
            if tensors.keys()!=initial_tensors.keys():raise ValueError('export adapter layout changed')
            if rank==0:
                dest.mkdir();save_file(tensors,str(dest/'adapter_model.safetensors'))
                adapted.peft_config['default'].save_pretrained(dest);(dest/'adapter_model.safetensors').chmod(0o644)
                seal_adapter(dest)
            dist.barrier();adapter_id=checkpoint_identity(dest)[0]
            checkpoint = dict(world_size=4,rank=rank,updates=step+1,names=[n for n,p in wrapped.named_parameters() if p.requires_grad],
                parameters={n:p.detach().cpu().clone() for n,p in wrapped.named_parameters() if p.requires_grad},
                optimizer=optimizer.state_dict(),epoch=next_epoch,next_batch=next_batch,schedule_sha256=config['schedule_sha256'],
                release_sha256=config['release_sha256'],adapter_id=adapter_id)
            commit_file(out/f'checkpoint-{step+1:04d}.pt', lambda stream: torch.save(checkpoint, stream))
            dist.barrier()
            if rank == 0:
                commit_json(root/f'checkpoint-{step+1:04d}-committed.json', dict(
                    updates=step+1, adapter_id=adapter_id, world_size=4,
                    files={str(p.relative_to(root)):file_hash(p) for p in
                           [root/f'rank-{r}'/f'checkpoint-{step+1:04d}.pt' for r in range(4)]},
                    scope='Local durable adapter and all four optimizer shards; not independent reload acceptance.'))
            dist.barrier()
            record['checkpoint_seconds']=time.monotonic()-export_begin
            record['total_step_seconds']=time.monotonic()-step_begin
            step_metrics.append(dict(**record,adapter_id=adapter_id))
            atomic_json(out/f'update-{step+1:04d}.json',step_metrics[-1]);cfg.jev_policy_id=base_id+':'+adapter_id
        atomic_json(out/'result.json',dict(passed=True,rank=rank,updates=start_update+config['updates'],executed_updates=config['updates'],base_id=base_id,
            initial_adapter_id=initial_id,final_adapter_id=adapter_id,step_metrics=step_metrics,
            execution_dtypes=executions,seconds=time.monotonic()-start,
            scope='Candidate true-batch native verl DP4 BF16 instruction CE prefix on full authenticated schedule. Completed update count only; independent endpoint reload/rollout and selection efficacy require separate acceptance.'))
        if after_training is not None:
            after_training(config=config,root=root,rank=rank,processor=processor,adapted=adapted,
                           wrapped=wrapped,optimizer=optimizer,actor_config=cfg,base_id=base_id,
                           adapter_id=adapter_id,device=device,execution_dtypes=executions)
    finally:
        try:
            if input_pipeline is not None:input_pipeline.close()
        finally:
            dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=4 or int(os.environ['LOCAL_RANK'])!=rank:raise ValueError('DP4 topology required')
    # Admission runs before mkdir, checkpoint writes, or importing torch/NPU.
    # Do not remove fsync or reuse historical Ceph-output launch snapshots.
    from multimodal_jev.local_runtime_storage import require_training_storage
    storage = require_training_storage(a.output)
    out=a.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    try:
        atomic_json(out/'storage-admission.json',dict(paths=storage,
            scope='Local hot paths and capacity only; not kernel or training acceptance.'))
        run(json.loads(a.config.read_text()),a.output,rank)
        atomic_json(out/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(out/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
