"""Configured Stage1 instruction trainer for frozen card layouts."""
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
    from tools.reports.context_infonce_actor import load as load_actor
    triplet = c['instruction_objective']['kind'] == 'intent_triplet_infonce_v1'
    actor_module=load_actor(c['source_root'],temperature=c['instruction_objective']['temperature'],probability_margin=c['instruction_objective'].get('probability_margin'))
    if 'single_card' in c:
        from multimodal_jev.single_card_forward import IndependentForward
        card=c['single_card']
        if file_hash(card['path'])!=card['sha256']:raise ValueError('single-card prompts changed')
        spec=json.loads(Path(card['path']).read_bytes())
        pl=c['stage1_layout']['plan']
        if file_hash(pl['path'])!=spec['plan'] or pl['sha256']!=spec['plan']:
            raise ValueError('changed card exposure plan')
        if not triplet:
            actor_module.forward_context_audio_batch=IndependentForward(spec,json.loads(Path(pl['path']).read_bytes())['assignments'])
    MODE,build_actor,make_batch=actor_module.MODE,actor_module.build_actor,actor_module.make_batch
    from multimodal_jev.slurp_portable_release import load
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
    from tools.reports.context_infonce_trainer import topology
    from multimodal_jev.context_warmup import learning_rate,set_learning_rate
    layout=topology(c,world,[p.name for p in Path('/dev').glob('davinci[0-9]*')])
    if c['runtime_profile']!=PROFILE:raise ValueError('fixed actor runtime required')
    for path,digest in c['pins'].items():
        if file_hash(path)!=digest:raise ValueError('dependency changed: '+path)
    torch.set_num_threads(2);torch.npu.set_device(local);device=torch.device('npu',local)
    torch.manual_seed(317);torch.npu.manual_seed_all(317);torch.use_deterministic_algorithms(True)
    dist.init_process_group('hccl',timeout=timedelta(seconds=900))
    out=root/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    progress=lambda phase,**kw:atomic_json(out/'progress.json',dict(phase=phase,unix=time.time(),**kw))
    try:
        from multimodal_jev.context_training_checkpoint import capture,restore,validate
        from multimodal_jev.context_replica_schedule import load_schedule,rank_indices
        from multimodal_jev.cached_context_batches import CachedContextBatches
        from multimodal_jev.cached_context_preparation import CachedContextPreparer,processor_cache_identity
        from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache as EncodedAudioCache
        from multimodal_jev.slurp_training import content_hash
        from multimodal_jev.local_checkpoint import commit_json
        from multimodal_jev.context_fsdp_export import canonical_execution
        from multimodal_jev.context_checkpoint_policy import export_checkpoint_policy
        from multimodal_jev.context_audio_collate import forward_context_audio_batch
        start_update=c['start_update'];stop_update=c['stop_update']
        if (type(start_update) is not int or type(stop_update) is not int or
                not 0<=start_update<stop_update<=layout['total_updates'] or c['stage']!='instruction' or
                type(c['checkpoint_interval']) is not int or c['checkpoint_interval']<1):
            raise ValueError('bounded full two-epoch instruction cursor required')
        if c['engineering_prefix'] is not True and not c['training_dispatch_allowed']:
            raise ValueError('formal full training requires runtime admission')
        probe = c.get('batch_probe')
        if probe is not None:
            if (not triplet or not c.get('mixed') or start_update != 0 or
                    c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False or
                    c['parity_passes'] != 0 or not c['batch_config'].get('batch_rule') or
                    stop_update > c['batch_config']['efficiency_steps']):
                raise ValueError('batch comparison requires a bounded triplet window from initialization')
        elif c['engineering_prefix'] is True and stop_update-start_update>2:
            raise ValueError('engineering prefix is at most two updates')
        pressure = c.get('finite_pressure')
        if pressure is not None and (probe is not None or not c['engineering_prefix']
                or start_update != 0 or stop_update != 2 or not triplet or not c.get('mixed')):
            raise ValueError('pressure witness is restricted to the initial two-step finite window')
        contract={k:c[k] for k in ('base_id','source_policy_id','source_contract','runtime_profile',
            'release_sha256','schedule_sha256','global_batch','microbatch','stage')}
        online = triplet and 'class_logit_ema' in c
        if triplet:
            key = 'class_logit_ema' if online else 'epoch_mining'
            contract[key] = c[key]
        if online:
            contract.update(triplet_prefetch=c['triplet_prefetch'], sampler_seed=c['mixed']['seed'])
        if 'single_card' in c:contract['single_card_sha256']=c['single_card']['sha256']
        if 'stage1_layout' in c:
            from multimodal_jev.stage1_training_layout import Stage1TrainingLayout, checkpoint_layout
            contract['stage1_layout']=checkpoint_layout(c)
        contract.update(optimizer=dict(lr=c['lr_schedule']['peak_lr'],betas=[.9,.999],eps=1e-8,weight_decay=0.),
            objective=c['instruction_objective'],lr_schedule=c['lr_schedule'],grad_clip=1.,world_size=world,
            batch_config=c['batch_config'],parent_schedule_sha256=c['parent_schedule_sha256'],
            schedule_release_sha256=c['schedule_release_sha256'])
        if c.get('prompt_layout') is not None:contract['prompt_layout']=c['prompt_layout']
        if 'activation_checkpointing' in c:contract['activation_checkpointing']=c['activation_checkpointing']
        progress('load')
        if checkpoint_identity(c['model'])[0]!=c['base_id']:raise ValueError('base changed')
        bundle=read_execution_bundle(c['source_policy'],expected_policy_id=c['source_policy_id'],
            expected_contract=c['source_contract'],expected_base_id=c['base_id'])
        if not c['engineering_prefix'] and bundle['contract']['token']['registration_status']!='admitted':
            raise ValueError('formal training requires admitted token contract')
        if not c['engineering_prefix']:
            if (bundle['contract']['runtime_profile']!=PROFILE or
                    bundle['contract']['request_version']!='jev-context-exclusive_event-v1'):
                raise ValueError('formal training requires actual request and execution profile')
            from multimodal_jev.context_token_contract import validate_processor_extension
            validate_processor_extension(c['model'],bundle['processor'])
        processor=Qwen3ASRProcessor.from_pretrained(bundle['processor'],local_files_only=True)
        _,base=load_runtime(c['model'],device,'bfloat16')
        from multimodal_jev.context_stage_lora import from_policy
        lora=from_policy(bundle['contract'])
        adapted=initialize_adapter(base,dict(lora_rank=lora['rank'],lora_alpha=lora['alpha'],
            lora_execution_dtype=lora['execution_dtype'])).eval()
        from multimodal_jev.context_model_profile import from_hf,build_head
        model_profile=from_hf(c['model'])
        if any(bundle['contract']['branch'].get(k)!=v for k,v in model_profile.decision_head().items()):
            raise ValueError('initial decision head differs from actual model geometry')
        head=build_head(model_profile).to(device=device,dtype=torch.bfloat16).eval()
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
        if 'activation_checkpointing' in c:
            from tools.reports.stage2_view_actor import configure_activation_checkpointing
            configure_activation_checkpointing(model, c['activation_checkpointing'])
        model.decision_head.module.register_forward_pre_hook(capture_dtype('head'))
        for n,m in model.named_modules():
            if type(m) is torch.nn.Linear and ('lora_A.' in n or 'lora_B.' in n):
                m.register_forward_pre_hook(capture_dtype(n))
        parameters=[p for p in wrapped.parameters() if p.requires_grad]
        def optimizer():return torch.optim.AdamW(parameters,lr=learning_rate(c['lr_schedule'],max(0,start_update-1),layout['total_updates']),betas=(.9,.999),eps=1e-8,
            weight_decay=0.,foreach=False,fused=False)
        opt=optimizer()
        cfg=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,
            ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,
            entropy_from_logits_with_chunking=False,use_torch_compile=False,fsdp_config={'dtype':'bfloat16'},
            calculate_sum_pi_squared=False,entropy_checkpointing=False,ppo_mini_batch_size=layout['local_batch'],
            ppo_micro_batch_size_per_gpu=c['microbatch'],ppo_epochs=1,use_kl_loss=False,calculate_entropy=False,
            entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,policy_loss={'loss_mode':MODE},
            clip_ratio=.2,clip_ratio_low=None,clip_ratio_high=None,clip_ratio_c=3.,global_batch_info={},grad_clip=1.,
            jev_policy_id=c['source_policy_id'],jev_context_parity=dict(probability_max_abs=0.,binary_logprob_max_abs=0.)))
        options=dict(stage='instruction',ranking_coefficient=1.,probability_coefficient=0.,algorithm='grpo_v1',
            kl_coefficient=0.,reference_policy_id=None)
        cfg.jev_context_objective=options
        collation=dict(decision_token_id=model.decision_token_id,pad_token_id=processor.tokenizer.pad_token_id,
            audio_token_id=base.config.audio_token_id,audio_chunk_size=base.config.audio_config.n_window*2,
            audio_infer_window=base.config.audio_config.n_window_infer)
        progress('prepare_full_schedule')
        signal=[None];stream=None;class_ema=None
        if rank==0 or triplet:
            try:
                if c.get('mixed'):
                    from multimodal_jev.stage1_mixed_schedule import schedule
                    entry = Path(c['mixed']['entry'])
                    proof = json.loads((entry/'admission.json').read_bytes())
                    schedule_manifest = Path(c['mixed']['schedule_manifest']) if probe else entry/'mixed-schedule.json'
                    if (file_hash(entry/'admission.json')!=c['release_sha256'] or
                        file_hash(entry/'synthetic.json')!=proof['roster_sha256'] or
                        file_hash(schedule_manifest)!=c['schedule_sha256']):
                        raise ValueError('mixed release or schedule changed')
                    rows,_,_=load(c['release'],c['mixed']['real_release_sha256'],'learning',purpose='optimize')
                    extra = json.loads((entry/'synthetic.json').read_bytes())
                    population = {r['id']:r for r in rows+extra}
                    task = json.loads((entry/'mixed-task.json').read_bytes())
                    batches,manifest = schedule(population,c['batch_config'],c['mixed']['seed'])
                    if manifest!=json.loads(schedule_manifest.read_bytes()):
                        raise ValueError('mixed schedule not reproducible')
                    if probe is not None:
                        from multimodal_jev.stage1_batch_probe import select_batches
                        batches = select_batches(batches, c['global_batch'], probe, stop_update)
                    if pressure is not None:
                        from multimodal_jev.stage1_batch_probe import bring_pressure_forward
                        bring_pressure_forward(batches, c['global_batch'], pressure)
                else:
                    rows,task,_=load(c['release'],c['release_sha256'],'learning',purpose='optimize')
                    population={r['id']:r for r in rows}
                    batches,manifest=load_schedule(c['schedule'],c['schedule_sha256'],
                        release_sha256=c['release_sha256'],population=population,batch_config=c['batch_config'],
                        parent_root=c['parent_schedule'],parent_manifest_sha256=c['parent_schedule_sha256'],
                        parent_release_sha256=c['schedule_release_sha256'])
                if len(population)!=c['batch_config']['source_queries'] or len(batches)!=layout['total_updates']:raise ValueError('full learning population required')
                factory=lambda:Qwen3ASRProcessor.from_pretrained(bundle['processor'],local_files_only=True)
                processor_identity=processor_cache_identity(c['source_policy'])
                # Prebuilt inputs are read-only; only runtime output needs writable storage.
                require_training_storage(root)
                audio_mirror=None
                if c.get('audio_mirror_root') is not None:
                    from multimodal_jev.local_audio_mirror import ReadOnlyAudioMirror
                    mirror_root=Path(c['audio_mirror_root'])
                    manifest_path=mirror_root/'MANIFEST.json'
                    if c['pins'].get(str(manifest_path))!=file_hash(manifest_path):
                        raise ValueError('pinned audio mirror publication required')
                    publication=json.loads(manifest_path.read_text())
                    records=publication['records']
                    if (publication['schema']!='context-learning-audio-mirror-v1' or
                            publication['release_sha256']!=c['schedule_release_sha256'] or
                            publication['role']!='learning' or publication['recordings']!=len(population) or
                            len(records)!=len(population) or {r['id'] for r in records}!=set(population)):
                        raise ValueError('mirror learning population differs')
                    for r in records:
                        source=population[r['id']]
                        if r['source']!=source['audio'] or r['sha256']!=source['audio_sha256']:
                            raise ValueError('mirror source identity differs')
                        if Path(r['local']).name!=r['sha256']+'.audio' or not Path(r['local']).is_absolute():
                            raise ValueError('mirror physical path differs')
                    audio_mirror=ReadOnlyAudioMirror(mirror_root/'audio')
                cache=EncodedAudioCache(c['encoded_audio_cache'])
                if c.get('mixed'):
                    from multimodal_jev.stage1_mixed_cache import RoutedCache
                    cache=RoutedCache(cache,EncodedAudioCache(c['mixed']['cache']),{r['audio'] for r in extra})
                layout_reader=None
                if 'stage1_layout' in c:
                    from multimodal_jev.stage1_layout_cache import ViewCache,read_view
                    from multimodal_jev.stage1_training_layout import source_pin
                    layout_reader=Stage1TrainingLayout.open(c,list(population),task)
                    cache=ViewCache(cache,read_view(source_pin(c,c['layout_cache']),layout_reader,c))
                preparer=CachedContextPreparer(factory,cache,
                    processor_sha256=processor_identity,decision_token_id=model.decision_token_id,
                    audio_token_id=base.config.audio_token_id,max_tokens=c.get('mixed',{}).get('input_limit',4096),audio_mirror=audio_mirror,prompt_layout=c.get('prompt_layout'))
                if triplet:
                    from multimodal_jev.intent_triplet_training import TripletStream
                    stream_type = TripletStream
                    if online:
                        from multimodal_jev.class_logit_ema import ClassLogitEMA
                        from multimodal_jev.intent_triplet_training import OnlineTripletStream
                        stream_type = OnlineTripletStream
                        class_ema = ClassLogitEMA(spec['labels'], **c['class_logit_ema'])
                    stream = stream_type(batches,population,preparer,layout_reader,spec,c,rank,world,collation)
                else:
                    stream=CachedContextBatches(batches,population,task,preparer,
                        slot_request=layout_reader,
                        start_update=start_update,stop_update=stop_update,global_batch=c['global_batch'])
                signal[0]=dict(error=None,schedule=manifest)
            except Exception as exc:signal[0]=dict(error=repr(exc))
        if triplet:
            errors = [None]*world
            dist.all_gather_object(errors,signal[0]['error'])
            if any(errors):
                raise ValueError(str(errors))
        else:
            dist.broadcast_object_list(signal,src=0)
        if signal[0]['error']:raise ValueError(signal[0]['error'])
        named=[(n,p) for n,p in wrapped.named_parameters() if p.requires_grad]
        from multimodal_jev.context_lora_checkpoint_scope import validate_trainable_scope
        atomic_json(out/'checkpoint-scope.json',validate_trainable_scope(named))
        frozen={n:tensor_identity(p.detach().cpu()) for n,p in wrapped.named_parameters() if not p.requires_grad}
        if start_update:
            previous=Path(c['resume_root']);commit_path=previous/f'checkpoint-{start_update:04d}-committed.json'
            if c['pins'].get(str(commit_path))!=file_hash(commit_path):raise ValueError('pinned committed checkpoint required')
            commit=json.loads(commit_path.read_text())
            if commit['contract']!=contract or commit['updates']!=start_update or len(commit['rank_shards'])!=world:
                raise ValueError('checkpoint commit differs')
            for relative,digest in commit['rank_shards'].items():
                if file_hash(previous/relative)!=digest:raise ValueError('rank checkpoint changed')
            state=torch.load(previous/f'rank-{rank}'/f'checkpoint-{start_update:04d}.pt',map_location='cpu',weights_only=True)
            identity=dict(rank=rank,world_size=world,completed_updates=start_update,contract=contract)
            local_error=None
            try:
                validate(named,opt,state,**identity)
                if online:
                    class_ema.load_state_dict(state['auxiliary']['class_logit_ema'])
                    witness = class_ema.receipt(c['class_logit_ema'])
                    if witness != commit.get('sampler_state') or class_ema.updates != start_update:
                        raise ValueError('restored EMA differs from committed optimizer step')
            except Exception as exc:local_error=repr(exc)
            peers=[None]*world;dist.all_gather_object(peers,local_error)
            if any(peers):raise ValueError('collective checkpoint rejection: '+str(peers))
            atomic_json(out/'restored.json',restore(named,opt,state,**identity))
            if online:
                atomic_json(out/'sampler-restored.json', witness)
        records=[]
        try:
            for index in range(start_update,stop_update):
                step_begin=time.monotonic();progress('prepare_batch',update=index)
                cfg.jev_policy_id=c['source_policy_id']+':instruction-update'+str(index)
                if triplet:
                    if index == start_update or batches[index]['epoch'] != batches[index-1]['epoch']:
                        if online:
                            stream.begin_epoch(index, class_ema)
                        else:
                            previous = None
                            if batches[index]['epoch']:
                                resumed = index == start_update and bool(start_update)
                                witness = c.get('resume_confusion') if resumed else None
                                report_path = (Path(witness['path']) if witness else
                                               root/f'confusion-epoch-{batches[index]["epoch"]}.json')
                                if witness and file_hash(report_path) != witness['sha256']:
                                    raise ValueError('resumed epoch confusion changed')
                                report = json.loads(report_path.read_bytes())
                                checkpoint_root = Path(c['resume_root']) if resumed else root
                                checkpoint = checkpoint_root/f'checkpoint-{index:04d}-committed.json'
                                if report['checkpoint_sha256'] != file_hash(checkpoint):
                                    raise ValueError('hard negative map belongs to a different checkpoint')
                                previous = report['distribution']
                            else:
                                from multimodal_jev.intent_triplet_training import mine_epoch
                                progress('bootstrap_train_confusion',update=index)
                                cached = c.get('bootstrap_input')
                                if cached is None:
                                    report = mine_epoch(wrapped,stream,c,0,device,c['source_policy_id'],bootstrap=True)
                                else:
                                    if file_hash(cached['path']) != cached['sha256']:
                                        raise ValueError('bootstrap report changed')
                                    report = json.loads(Path(cached['path']).read_bytes())
                                    if report['policy_id'] != c['source_policy_id'] or report['labels'] != spec['labels']:
                                        raise ValueError('bootstrap policy or labels changed')
                                if rank == 0:
                                    atomic_json(root/'confusion-init.json',dict(report,policy_id=c['source_policy_id']))
                                previous = report['distribution']
                            stream.begin_epoch(index,previous)
                    from multimodal_jev.intent_triplet_training import update
                    actual, groups, real_queries, *versions = next(stream.queue)
                    if actual != index:
                        raise ValueError('triplet prefetch order differs')
                    preparation = time.monotonic()-step_begin
                    current_lr = set_learning_rate(opt,c['lr_schedule'],index,layout['total_updates'])
                    progress('native_update',update=index,learning_rate=current_lr)
                    torch.npu.reset_peak_memory_stats()
                    started = time.monotonic()
                    passes = c['parity_passes'] if index == start_update else 0
                    if c['batch_config'].get('batch_rule') and len(groups) != c['microbatch']:
                        raise ValueError('one local microbatch per optimizer update required')
                    metrics,parity = update(wrapped,opt,groups,real_queries,c,collation,cfg.jev_policy_id,device,world,passes,
                        **(dict(class_ema=class_ema) if online else {}))
                    if online:
                        stream.publish(class_ema)
                        metrics.update(sampler_source_update=versions[0], sampler_lag=index-versions[0])
                    torch.npu.synchronize()
                    seconds = time.monotonic()-started
                    if passes:
                        if parity['inference_passes'] != passes:
                            raise ValueError('incomplete inference parity')
                        atomic_json(out/'inference-parity.json',parity)
                    timing = dict(wait_seconds=preparation,local_groups=len(groups))
                else:
                    signal=[None];parts=None
                    if rank==0:
                        try:
                            items,metadata=stream.consume(index,cfg.jev_policy_id)
                            lengths=[(p['inputs']['input_ids'].shape[-1],p['inputs']['input_features'].shape[-1]) for p in items]
                            planned=rank_indices(batches[index],lengths,c['batch_config'])
                            ids=[i for group in planned for i in group]
                            parts=list(make_batch([items[i] for i in ids],**options).split(layout['local_batch']))
                            atomic_json(root/f'batch-{index:04d}.json',dict(policy_id=cfg.jev_policy_id,
                                rank_indices=planned,records=metadata))
                            signal[0]=dict(error=None,timing=stream.last_timing)
                        except Exception as exc:signal[0]=dict(error=repr(exc))
                    dist.broadcast_object_list(signal,src=0)
                    if signal[0]['error']:raise ValueError(signal[0]['error'])
                    incoming=[None];dist.scatter_object_list(incoming,parts,src=0);batch=incoming[0]
                    preparation=time.monotonic()-step_begin
                    actor=build_actor(cfg,wrapped,opt,collation=collation)
                    current_lr=set_learning_rate(opt,c['lr_schedule'],index,layout['total_updates'])
                    progress('native_update',update=index,learning_rate=current_lr)
                    torch.npu.reset_peak_memory_stats();torch.npu.synchronize();started=time.monotonic()
                    if c.get('parity_passes') and index == start_update:
                        from multimodal_jev.card_training_parity import observe
                        parity = observe(actor_module, wrapped, batch, cfg.jev_policy_id,
                            collation, device, c['parity_passes'], c['microbatch'], c['parity_probability_atol'])
                        actor=build_actor(cfg,wrapped,opt,collation=collation)
                    metrics=actor.update_policy(batch);torch.npu.synchronize();seconds=time.monotonic()-started
                    if c.get('parity_passes') and index == start_update:
                        if parity['compared'] != c['parity_passes']*c['microbatch']:
                            raise ValueError('incomplete train/inference parity comparison')
                        atomic_json(out/'inference-parity.json', parity)
                    timing = signal[0]['timing']
                    real_queries = actor.last_admission['global_real_requests']
                audit=audit_adam_shards(named,opt,index+1)
                save_now=probe is None and pressure is None and ((index+1)%c['checkpoint_interval']==0 or index+1==stop_update)
                if save_now:
                    if frozen!={n:tensor_identity(p.detach().cpu()) for n,p in wrapped.named_parameters() if not p.requires_grad}:
                        raise ValueError('frozen backbone changed')
                    sampler_proof = {}
                    auxiliary = None
                    if online:
                        witness = class_ema.receipt(c['class_logit_ema'])
                        peers = [None]*world
                        dist.all_gather_object(peers, witness)
                        if class_ema.updates != index+1 or any(peer != witness for peer in peers):
                            raise ValueError('checkpoint sampler differs across ranks or steps')
                        sampler_proof = dict(sampler_state=witness)
                        auxiliary = dict(class_logit_ema=class_ema.state_dict())
                    state=capture(named,opt,rank=rank,world_size=world,completed_updates=index+1,
                                  contract=contract,**(dict(auxiliary=auxiliary) if online else {}))
                    commit_file(out/f'checkpoint-{index+1:04d}.pt',lambda f:torch.save(state,f))
                    with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
                        full=canonical_execution({n:p.detach().cpu().to(torch.bfloat16).clone()
                            for n,p in model.named_parameters() if p.requires_grad})
                        hashes={n:tensor_identity(v) for n,v in full.items()};peers=[None]*world
                        dist.all_gather_object(peers,hashes)
                        if any(v!=hashes for v in peers):raise ValueError('assembled execution differs across ranks')
                        if rank==0:
                            commit_file(root/f'execution-{index+1:04d}.pt',lambda f:torch.save(full,f))
                            joint=export_checkpoint_policy(c['source_policy'],root/f'policy-{index+1:04d}',full,
                                policy_id=c['source_policy_id'],contract=c['source_contract'],base_id=c['base_id'])
                    dist.barrier()
                    if rank==0:
                        commit_json(root/f'checkpoint-{index+1:04d}-committed.json',dict(updates=index+1,
                            contract=contract,rank_shards={f'rank-{r}/checkpoint-{index+1:04d}.pt':
                                file_hash(root/f'rank-{r}/checkpoint-{index+1:04d}.pt') for r in range(world)},
                            execution_file=f'execution-{index+1:04d}.pt',
                            joint_policy=joint,joint_policy_directory=f'policy-{index+1:04d}',
                            joint_policy_manifest_sha256=file_hash(root/f'policy-{index+1:04d}/MANIFEST.json'),
                            execution_sha256=file_hash(root/f'execution-{index+1:04d}.pt'), **sampler_proof))
                    dist.barrier()
                record=dict(update=index+1,learning_rate=current_lr,preparation_seconds=preparation,actor_seconds=seconds,
                    total_seconds=time.monotonic()-step_begin,input_service=timing,metrics=metrics,
                    real_queries=real_queries,
                    peak_allocated=torch.npu.max_memory_allocated(),checkpoint=save_now,optimizer_audit=audit)
                if probe is not None or pressure is not None:
                    record['peak_reserved'] = torch.npu.max_memory_reserved()
                records.append(record);atomic_json(out/f'update-{index+1:04d}.json',record)
                if (triplet and not online and not c['engineering_prefix'] and c.get('mine_after_window', True) and
                        (index+1 == len(batches) or batches[index+1]['epoch'] != batches[index]['epoch'])):
                    from multimodal_jev.intent_triplet_training import mine_epoch
                    epoch = batches[index]['epoch']
                    progress('mine_full_train',epoch=epoch+1,update=index+1)
                    report = mine_epoch(wrapped,stream,c,epoch,device,cfg.jev_policy_id+':post-update')
                    if rank == 0:
                        report.update(checkpoint_sha256=file_hash(root/f'checkpoint-{index+1:04d}-committed.json'))
                        atomic_json(root/f'confusion-epoch-{epoch+1}.json',report)
                    dist.barrier()
            atomic_json(out/'result.json',dict(passed=True,start_update=start_update,stop_update=stop_update,
                executions=executions,scope='Native context instruction loop; task RL and efficacy separate'))
            dist.barrier()
            if rank==0:atomic_json(root/'result.json',dict(passed=True,start_update=start_update,
                stop_update=stop_update,schedule_updates=layout['total_updates'],learning_population=c['batch_config']['source_queries'],
                engineering_prefix=c['engineering_prefix']))
        finally:
            if stream is not None:stream.close()
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    run(json.loads(a.config.read_text()),a.output)
