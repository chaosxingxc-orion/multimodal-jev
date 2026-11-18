#!/usr/bin/env python3
"""Two-rank native verl/FSDP BF16 capacity/gradient probe, not DP4 release.

Uses real learning labels/weights; every case resets identical FP32 LoRA masters
and Adam. Saves raw tensors for independent comparison, including failures.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time


def main():
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import CustomPolicy
    from peft import PeftModel
    from omegaconf import OmegaConf
    from multimodal_jev.baseline import atomic_json, file_hash
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3, configure_actor_profile
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.simple_prepared_inputs import prepared
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_fsdp import unshared_linear_modules
    from multimodal_jev.span_batch_candidate_runtime import install_candidate
    from multimodal_jev.span_length_groups import length_group_indices
    from multimodal_jev.span_batched_hybrid_actor import build_batched_actor
    from multimodal_jev.span_hybrid_actor import MODE, make_batch
    from multimodal_jev.slurp_official_training import load, example
    from multimodal_jev.slurp_instruction_schedule import epoch_schedule
    from multimodal_jev.span_audio_collate import forward_span_audio_batch
    from multimodal_jev.span_parity import compare_span_policies
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text())
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2 or rank!=local_rank or c['physical_devices']!=[5,6]:raise ValueError('explicit DP2 physical5,6 required')
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci5','davinci6']:raise ValueError('isolated DP2 device nodes required')
    device=f'npu:{local_rank}'
    root=Path(a.output)/f'rank-{rank}';root.mkdir(parents=True,exist_ok=False)
    for name,digest in c['pins'].items():
        if file_hash(name)!=digest:raise ValueError('pinned dependency changed: '+name)
    torch.set_num_threads(4);torch.manual_seed(317);torch.npu.set_device(local_rank)
    if c.get('deterministic_algorithms', False):
        torch.use_deterministic_algorithms(True)
    initialize_verl_preserving_qwen3();configure_actor_profile(c['runtime_profile'])
    dist.init_process_group('hccl',init_method='env://',timeout=timedelta(seconds=900))
    try:
        processor,base=load_runtime(c['model'],device,'bfloat16')
        adapted=PeftModel.from_pretrained(base,c['adapter'],is_trainable=True,autocast_adapter_dtype=False).eval()
        model=NativeAudioSpanModel(adapted.get_base_model()).eval()
        handle,profile=install_candidate(adapted,model)
        atomic_json(root/'candidate-profile.json',profile)
        data,task,_=load(c['release'],c['release_sha256'],'learning',purpose='optimize')
        population={r['id']:r for r in data}
        _,schedule=epoch_schedule(data,seed=317,epoch=0,task_mode='direct_final',class_weighting='sqrt_cap4')
        artifact=json.loads(Path(c['inputs']).read_text())
        rows=sorted(list(prepared(c,artifact,processor=processor)),key=lambda r:(int(r[1]['input_features_mask'].sum()),r[0]['id']))
        chosen=[rows[round(i*(len(rows)-1)/63)] for i in range(64)]
        # Fixed random permutation independent of labels and bucket candidates.
        chosen=[chosen[i] for i in torch.randperm(64,generator=torch.Generator().manual_seed(317)).tolist()]
        items=[];records=[]
        for row in chosen:
            source_id='slurp:'+Path(row[0]['request']['audio']).name
            source=population[source_id]
            if not row[0]['id'].startswith(source_id+'::') or source['audio_sha256']!=row[0]['audio_sha256']:
                raise ValueError('scheduled request/source waveform binding differs')
            request,target,_=example(source,task,'direct','native')
            if json.loads(json.dumps(request.model_input()))!=row[0]['request']:raise ValueError('label-free task binding differs')
            item=dict(request_id=row[4],policy_id=c['base_id']+':'+c['adapter_id'],inputs=row[1],positions=row[2],
                      candidates=row[3],targets=torch.tensor([target['targets']]),supervised=torch.tensor([target['supervised']]),
                      span_weights=torch.tensor([target['span_weights']],dtype=torch.float32),
                      request_weight=schedule['class_weights'][source['intent']],real_request=True)
            items.append(item);records.append(dict(id=source['id'],request_id=row[4],request_identity=row[5],
                tokens=row[1]['input_ids'].shape[-1],frames=int(row[1]['input_features_mask'].sum()),
                targets=target['targets'],request_weight=item['request_weight']))
        atomic_json(root/'inputs.json',dict(records=records,selection='64 frame quantiles of authenticated learning256, fixed seed permutation',
            release_sha256=c['release_sha256'],full_population=len(data),global_requests_in_probe=64,world_size=world,rank=rank))
        options=dict(pad_token_id=processor.tokenizer.pad_token_id,audio_token_id=base.config.audio_token_id,
            audio_chunk_size=base.config.audio_config.n_window*2,audio_infer_window=base.config.audio_config.n_window_infer)
        behavior=None
        if c.get('reference_before_fsdp', False):
            with torch.no_grad():
                behavior=[tuple(v.cpu().clone() for v in forward_span_audio_batch(model,[i],device=device,dtype=torch.bfloat16,**options)[0]) for i in items]
        for param in model.parameters():
            if param.requires_grad:param.data=param.data.float()
        units=unshared_linear_modules(model)
        wrapped=FSDP(model,auto_wrap_policy=CustomPolicy(lambda m:m in units),device_id=torch.device(device),
            sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,sync_module_states=True,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32,
                cast_forward_inputs=True,keep_low_precision_grads=False),limit_all_gathers=True)
        if c.get('diagnose_shared_views', False):
            # Observe the original FSDP operation; never repair or suppress it.
            from torch.distributed.fsdp._flat_param import FlatParamHandle
            original_views = FlatParamHandle._use_unsharded_views
            module_names = {id(m): n for n,m in wrapped.named_modules()}
            def describe(module, name):
                value = getattr(module, name, None)
                return dict(module=module_names.get(id(module), type(module).__name__), name=name,
                    type=type(value).__name__, registered_type=type(module._parameters.get(name)).__name__,
                    object_id=id(value), shape=list(value.shape) if isinstance(value,torch.Tensor) else None)
            def shared_views(handle, as_params):
                before = [dict(primary=describe(s.prim_module,s.prim_param_name),
                               shared=describe(s.module,s.param_name))
                          for s in handle.flat_param._shared_param_infos]
                try:
                    return original_views(handle, as_params)
                except Exception as exc:
                    atomic_json(root/'shared-view-failure.json',dict(error=repr(exc),as_params=as_params,
                        training_state=str(handle._training_state),strategy=str(handle._sharding_strategy),
                        before=before,after=[dict(primary=describe(s.prim_module,s.prim_param_name),
                            shared=describe(s.module,s.param_name)) for s in handle.flat_param._shared_param_infos],
                        primary_parameters=[describe(s.module,s.param_name) for s in handle.flat_param._param_infos]))
                    raise
            FlatParamHandle._use_unsharded_views = shared_views
        parameters={n:p for n,p in wrapped.named_parameters() if p.requires_grad}
        assert len(parameters)==112 and all(p.dtype==torch.float32 for p in parameters.values())
        initial={n:p.detach().cpu().clone() for n,p in parameters.items()}
        if behavior is None:
            with torch.no_grad():
                behavior=[tuple(v.cpu().clone() for v in forward_span_audio_batch(wrapped,[i],device=device,dtype=torch.bfloat16,**options)[0]) for i in items]
        lengths=[(r['tokens'],r['frames']) for r in records]
        cases=[(1,'original'),(8,'original'),(8,'length_grouped'),(16,'original'),(16,'length_grouped'),(32,'original'),(32,'length_grouped')]
        if 'cases' in c:
            requested=[tuple(row) for row in c['cases']]
            if not requested or requested[0]!=(1,'original') or any(row not in cases for row in requested):
                raise ValueError('explicit supported diagnostic cases starting with micro1 required')
            cases=requested
        baseline=None;reports=[]
        for micro,order in cases:
            tag=f'm{micro:02d}-{order}';indices=length_group_indices(lengths,microbatch=micro,world_size=world,order=order)
            indices=indices[rank]
            repetitions=[]
            for repetition in range(c['repetitions']):
                with torch.no_grad():
                    for n,param in parameters.items():param.copy_(initial[n].to(param.device))
                opt=torch.optim.AdamW(list(parameters.values()),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
                cfg=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,ulysses_sequence_parallel_size=1,
                    use_dynamic_bsz=False,use_prefix_grouper=False,entropy_from_logits_with_chunking=False,use_torch_compile=False,
                    fsdp_config={'dtype':'bfloat16'},calculate_sum_pi_squared=False,entropy_checkpointing=False,
                    ppo_mini_batch_size=32,ppo_micro_batch_size_per_gpu=micro,ppo_epochs=1,use_kl_loss=False,
                    calculate_entropy=False,entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,
                    policy_loss={'loss_mode':MODE},clip_ratio=.2,clip_ratio_low=None,clip_ratio_high=None,clip_ratio_c=3.,
                    global_batch_info={},grad_clip=1.,jev_policy_id=items[0]['policy_id']))
                actor=build_batched_actor(cfg,wrapped,opt,collation=options)
                rank_indices=length_group_indices(lengths,microbatch=micro,world_size=world,order=order)
                ordered=[items[i] for ids in rank_indices for i in ids]
                batch=make_batch(ordered,stage='instruction',reduction='real_request_mean').split(32)[rank]
                captured={};original=actor._optimizer_step;preupdate=[]
                def capture_policy(module,args,out):
                    preupdate.extend([[p[i].detach().cpu().tolist() for p in out[1]] for i in range(out[1][0].shape[0])])
                output_handle=wrapped.register_forward_hook(capture_policy)
                def capture():
                    if any(p.grad is None and p.numel()!=0 for p in parameters.values()):raise ValueError('missing nonempty shard gradient')
                    captured['gradients']={n:p.grad.detach().float().cpu().clone() if p.grad is not None else torch.empty_like(p,device='cpu') for n,p in parameters.items()}
                    if not all(torch.isfinite(v).all() for v in captured['gradients'].values()):raise ValueError('nonfinite gradients')
                    return original()
                actor._optimizer_step=capture
                atomic_json(root/'progress.json',dict(phase='native_update',case=tag,repetition=repetition,updated_unix=time.time()))
                torch.npu.synchronize();torch.npu.reset_peak_memory_stats();dist.barrier();start=time.monotonic()
                metrics=actor.update_policy(batch);torch.npu.synchronize();dist.barrier();seconds=time.monotonic()-start
                output_handle.remove()
                peak_allocated=torch.npu.max_memory_allocated();peak_reserved=torch.npu.max_memory_reserved()
                if len(preupdate)!=32:raise ValueError('missing pre-update policy')
                parity=[compare_span_policies([v[0].tolist() for v in behavior[i]],out) for i,out in zip(indices,preupdate,strict=True)]
                atomic_json(root/f'{tag}-rep{repetition}-preupdate.json',dict(indices=indices,reference=[[v[0].tolist() for v in row] for row in behavior],candidate_logprobs=preupdate,parity=parity))
                if not all(p['passed'] for p in parity):raise ValueError('training forward differs from single-request inference')
                captured.update(parameters={n:p.detach().cpu().clone() for n,p in parameters.items()},
                    exp_avg={n:opt.state[p]['exp_avg'].cpu().clone() if p.numel() else torch.empty_like(p,device='cpu') for n,p in parameters.items()},
                    exp_avg_sq={n:opt.state[p]['exp_avg_sq'].cpu().clone() if p.numel() else torch.empty_like(p,device='cpu') for n,p in parameters.items()})
                assert all(opt.state[p]['step'].item()==1 for p in parameters.values() if p.numel())
                errors={}
                if baseline is None:baseline=captured
                for kind,values in captured.items():
                    delta=torch.cat([(values[n]-baseline[kind][n]).flatten() for n in values]);ref=torch.cat([baseline[kind][n].flatten() for n in values])
                    errors[kind]=dict(max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/ref.norm().clamp_min(1e-30)),
                        strict_cpu_tolerance_pass=all(torch.allclose(values[n],baseline[kind][n],atol=3e-6,rtol=3e-5) for n in values))
                path=root/f'{tag}-rep{repetition}.pt';torch.save(captured,path)
                if c.get('post_update_policies', False):
                    wrapped.eval()
                    with torch.no_grad():
                        after=[dict(index=i,logprobs=[v[0].float().cpu().tolist() for v in
                            forward_span_audio_batch(wrapped,[items[i]],device=device,dtype=torch.bfloat16,**options)[0]])
                            for i in range(rank*32,(rank+1)*32)]
                    atomic_json(root/f'{tag}-rep{repetition}-postupdate.json',dict(
                        records=after,scope='Same canonical rank slice after one Adam update; no action resampling, no promotion'))
                    if c.get('post_update_batch_parity', False):
                        canonical=list(range(rank*32,(rank+1)*32))
                        batched=[]
                        with torch.no_grad():
                            for start in range(0,len(canonical),micro):
                                ids=canonical[start:start+micro]
                                output=forward_span_audio_batch(wrapped,[items[i] for i in ids],
                                    device=device,dtype=torch.bfloat16,**options)
                                batched.extend(dict(index=i,logprobs=[v[0].float().cpu().tolist() for v in values])
                                               for i,values in zip(ids,output,strict=True))
                        checks=[compare_span_policies(single['logprobs'],batch['logprobs'])
                                for single,batch in zip(after,batched,strict=True)]
                        atomic_json(root/f'{tag}-rep{repetition}-postupdate-batched.json',dict(
                            records=batched,parity=checks,passed=all(x['passed'] for x in checks),
                            scope='Same post-Adam parameters, canonical inputs, single vs batched candidate forward; not vLLM acceptance'))
                repetitions.append(dict(repetition=repetition,seconds=seconds,requests_per_second=32/seconds,
                    maximum_allocated_bytes=peak_allocated,maximum_reserved_bytes=peak_reserved,
                    tensor_file=path.name,tensor_sha256=file_hash(path),errors=errors,metrics=metrics))
                atomic_json(root/f'{tag}.json',dict(microbatch=micro,order=order,indices=indices,repetitions=repetitions))
                del actor,opt,captured,batch
            reports.append(dict(microbatch=micro,order=order,repetitions=repetitions))
        atomic_json(root/'result.json',dict(completed=True,results=reports,training_promotion=False,
            scope='DP2 FSDP/native verl CE-only real learning64, local32, BF16 execution/FP32 master Adam, identical reset per update. Timings include gradient CPU capture; not DP4/global128 or steady state, no GRPO/RLOO or vLLM admission. Strict CPU tolerances are diagnostic, not a new BF16 acceptance threshold.'))
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
