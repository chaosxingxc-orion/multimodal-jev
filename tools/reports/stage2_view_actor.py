"""Static Stage2 actor loop for dataset releases, extracted from the pinned plugin runtime.

One loop covers prefix, replay, tail and the formal schedule. The entrypoint
owns dispatch admission. Dataset windows and plugin loading are explicit inputs.
"""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash


def configure_activation_checkpointing(model, setting):
    """Recompute the configured layers during backward, preserving module names."""
    if not setting or not setting['enabled']:
        return
    from functools import wraps
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    layers = []
    for path in setting['module_lists']:
        try:
            container = model.get_submodule(path)
        except AttributeError as exc:
            raise ValueError(f'activation checkpointing module list is missing: {path}') from exc
        if not isinstance(container, nn.ModuleList) or not len(container):
            raise ValueError(f'activation checkpointing expects a nonempty ModuleList: {path}')
        layers.extend(container)
    if len(layers) != len({id(layer) for layer in layers}):
        raise ValueError('activation checkpointing selected the same layer more than once')

    def checkpointed(forward):
        @wraps(forward)
        def call(*args, **kwargs):
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)
        return call

    for layer in layers:
        layer.forward = checkpointed(layer.forward)


def load_actor_stack(c, device, out, execution, transition):
    from transformers import Qwen3ASRProcessor
    import torch
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.context_decision_model import ContextDecisionModel
    from multimodal_jev.context_decision_head import ContextDecisionHead
    from multimodal_jev.context_runtime_profile import PROFILE, install_actor
    from multimodal_jev.context_policy_snapshot import load_policy
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    from multimodal_jev.policy_identity import checkpoint_identity
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
    lora=transition['stage2']
    adapted=initialize_adapter(base,dict(lora_rank=lora['rank'],lora_alpha=lora['alpha'],lora_execution_dtype=lora['execution_dtype'])).eval()
    head=ContextDecisionHead(**{k:c['source_contract']['branch'][k] for k in ('hidden_size','num_heads','ffn_size','residual_scale')}).to(device=device,dtype=torch.bfloat16).eval()
    model=ContextDecisionModel(adapted.get_base_model().model,head,bundle['contract']['token']['id']).eval()
    load_policy(c['source_policy'],model,expected_policy_id=c['source_policy_id'],
        expected_contract=c['source_contract'],expected_base_id=c['base_id'])
    runtime_handle,profile=install_actor(adapted,model,PROFILE)
    atomic_json(out/'runtime.json',profile)
    wrapped, opt, executions = wrap_actor(model, device, execution)
    return base,processor,model,wrapped,opt,executions,runtime_handle


def wrap_actor(model, device, execution):
    """Build the shared BF16 FSDP actor around a provider's decision model."""
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import CustomPolicy
    from multimodal_jev.context_fsdp import context_fsdp_units
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
    configure_activation_checkpointing(model, execution.get('activation_checkpointing'))
    model.decision_head.module.register_forward_pre_hook(capture_dtype('head'))
    for n,m in model.named_modules():
        if type(m) is torch.nn.Linear and ('lora_A.' in n or 'lora_B.' in n):
            m.register_forward_pre_hook(capture_dtype(n))
    parameters=[p for p in wrapped.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, **execution['optimizer'], foreach=False, fused=False)
    return wrapped, opt, executions


def run(c,root,*,view,actor_loader):
    if type(c.get('objective_trace',False)) is not bool:
        raise ValueError('explicit Boolean objective_trace required')
    if c.get('objective_trace') and (c['view_window'] not in ('prefix','recovery') or c['stop_update']-c['start_update']>2):
        raise ValueError('full objective traces limited to two-update prefix admission')
    from tools.reports.context_cached_reference_trainer import validate_objective,no_online_processor
    objective_contract=(view.editable_objective(c) if 'objective_config' in c else validate_objective(c))
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
    dual_actor=actor_loader(c['source_root'],c['objective_plugin'])
    MODE,build_actor,make_batch=dual_actor.MODE,dual_actor.build_actor,dual_actor.make_batch
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
    from tools.reports.context_replica_trainer_candidate import topology
    from multimodal_jev.context_replica_schedule import rank_indices
    layout=topology(c,world,[p.name for p in Path('/dev').glob('davinci[0-9]*')])
    if c['runtime_profile']!=PROFILE:raise ValueError('fixed actor runtime required')
    for path,digest in c['pins'].items():
        if file_hash(path)!=digest:raise ValueError('dependency changed: '+path)
    execution = c['view_execution']
    view.validate_execution(execution)
    torch.set_num_threads(execution['actor_threads'])
    torch.npu.set_device(local);device=torch.device('npu',local)
    torch.manual_seed(execution['actor_seed']);torch.npu.manual_seed_all(execution['actor_seed']);torch.use_deterministic_algorithms(True)
    dist.init_process_group('hccl',timeout=timedelta(seconds=execution['collective_timeout_seconds']))
    out=root/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    progress=lambda phase,**kw:atomic_json(out/'progress.json',dict(phase=phase,unix=time.time(),**kw))
    stream=None;client=None
    try:
        from multimodal_jev.context_training_checkpoint import capture,restore,validate
        from multimodal_jev.cached_context_batches import CachedContextBatches
        from multimodal_jev.cached_context_preparation import CachedContextPreparer,processor_cache_identity
        from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache as EncodedAudioCache
        from multimodal_jev.slurp_training import content_hash
        from multimodal_jev.local_checkpoint import commit_json
        from multimodal_jev.context_fsdp_export import canonical_execution
        from multimodal_jev.context_checkpoint_policy import export_checkpoint_policy
        from multimodal_jev.context_audio_collate import forward_context_audio_batch
        start_update=c['start_update'];stop_update=c['stop_update']
        view.indices(c)
        formal=c['view_window']=='formal'
        from tools.reports.context_replica_formal_candidate import contract as formal_contract
        contract={k:c[k] for k in ('base_id','source_policy_id','source_contract','runtime_profile',
            'release_sha256','schedule_sha256','global_batch','microbatch','stage','algorithm','behavior_bounds')}
        from tools.reports.context_dual_runtime_candidate import verify_runtime
        from tools.reports.context_dual_actor_candidate import options_check
        runtime_identity=verify_runtime(c)
        settings = view.actor_contract(c)
        options = settings['dual_objective']
        options_check(options)
        contract.update(settings,dual_runtime_identity=runtime_identity,world_size=world,
            macro_layout=layout,schedule_release_sha256=c['mirror_release_sha256'])
        if formal:contract['formal_experiment']=formal_contract(c)
        contract['training_dataset']=view.checkpoint_contract(c)
        from multimodal_jev.context_stage_lora import admit_stage2
        contract['stage_lora_transition']=admit_stage2(c['stage_lora_transition'],c['source_contract'])
        contract['reference_dataset']=dict(manifest_sha256=c['reference_dataset']['manifest_sha256'],
            identity=c['reference_dataset']['identity'],mode='readonly_no_online_fill')
        contract['stage2_objective']=objective_contract
        contract['objective_plugin']=dual_actor.objective_plugin_identity
        progress('load')
        base,processor,model,wrapped,opt,executions,runtime_handle=load_actor_stack(
            c,device,out,execution,contract['stage_lora_transition'])
        cfg=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,
            ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,
            entropy_from_logits_with_chunking=False,use_torch_compile=False,fsdp_config={'dtype':'bfloat16'},
            calculate_sum_pi_squared=False,entropy_checkpointing=False,ppo_mini_batch_size=layout['local_batch'],
            ppo_micro_batch_size_per_gpu=c['microbatch'],ppo_epochs=1,use_kl_loss=False,calculate_entropy=False,
            entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,policy_loss={'loss_mode':MODE},
            **execution['policy_clip'],global_batch_info={},grad_clip=execution['grad_clip'],
            jev_policy_id=c['source_policy_id'],jev_context_parity=c['behavior_bounds']))

        cfg.jev_context_objective=options
        collation=dict(decision_token_id=model.decision_token_id,pad_token_id=processor.tokenizer.pad_token_id,
            audio_token_id=base.config.audio_token_id,audio_chunk_size=base.config.audio_config.n_window*2,
            audio_infer_window=base.config.audio_config.n_window_infer)
        progress('prepare_full_schedule')
        signal=[None];stream=None;client=None;session=None
        if rank==0:
            try:
                rows,task,_=load(c['release'],c['release_sha256'],'learning',purpose='optimize')
                full_population={r['id']:r for r in rows}
                population,batches,manifest=view.consume(c,rows,task)
                factory=no_online_processor
                processor_identity=processor_cache_identity(c['source_policy'])
                # Input cache is read-only; runtime output storage was checked above.
                audio_mirror=None
                if c.get('audio_mirror_root') is not None:
                    mirror_root=Path(c['audio_mirror_root'])
                    manifest_path=mirror_root/'MANIFEST.json'
                    if c['pins'].get(str(manifest_path))!=file_hash(manifest_path):
                        raise ValueError('pinned audio mirror publication required')
                    publication=json.loads(manifest_path.read_text())
                    records=publication['records']
                    if (publication['schema']!='context-learning-audio-mirror-v1' or
                            publication['release_sha256']!=c['mirror_release_sha256'] or
                            publication['role']!='learning' or publication['recordings']!=len(full_population) or
                            len(records)!=len(full_population) or {r['id'] for r in records}!=set(full_population)):
                        raise ValueError('mirror learning population differs')
                    for r in records:
                        source=full_population[r['id']]
                        if r['source']!=source['audio'] or r['sha256']!=source['audio_sha256']:
                            raise ValueError('mirror source identity differs')
                        recorded_path = Path(r['local'])
                        if recorded_path.parts[-2:] != ('audio', r['sha256']+'.audio'):
                            raise ValueError('mirror physical path differs')
                    audio_mirror=view.ReadOnlyAudioMirror(mirror_root/'audio')
                preparer=CachedContextPreparer(factory,EncodedAudioCache(c['encoded_audio_cache']),
                    processor_sha256=processor_identity,decision_token_id=model.decision_token_id,
                    audio_token_id=base.config.audio_token_id,max_tokens=execution['max_input_tokens'],audio_mirror=audio_mirror,prompt_layout=c.get('prompt_layout'))
                stream=CachedContextBatches(batches,population,task,preparer,
                    start_update=start_update,stop_update=stop_update,global_batch=c['global_batch'])
                signal[0]=dict(error=None,schedule=manifest)
            except Exception as exc:signal[0]=dict(error=repr(exc))
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
            try:validate(named,opt,state,**identity)
            except Exception as exc:local_error=repr(exc)
            peers=[None]*world;dist.all_gather_object(peers,local_error)
            if any(peers):raise ValueError('collective checkpoint rejection: '+str(peers))
            atomic_json(out/'restored.json',restore(named,opt,state,**identity))
        # Bind resumed/live actor tensors to the joint policy before requesting
        # behavior. A checkpoint manifest alone does not prove that equality.
        active_path=(Path(c['resume_root'])/commit['joint_policy_directory'] if start_update
                     else Path(c['source_policy']))
        active_manifest=json.loads((active_path/'MANIFEST.json').read_text())
        active_bundle=read_execution_bundle(active_path,expected_policy_id=active_manifest['policy_id'],
            expected_contract=c['source_contract'],expected_base_id=c['base_id'])
        if start_update and file_hash(active_path/'MANIFEST.json')!=commit['joint_policy_manifest_sha256']:
            raise ValueError('resume joint policy manifest differs')
        with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
            live_weights=canonical_execution({n:p.detach().cpu().to(torch.bfloat16).clone()
                for n,p in model.named_parameters() if p.requires_grad})
            expected={k:v for k,v in active_bundle['tensors'].items() if k!='decision_token_row'}
            matches=(live_weights.keys()==expected.keys() and
                     all(torch.equal(v,expected[k]) for k,v in live_weights.items()))
            checks=[None]*world;dist.all_gather_object(checks,matches)
            if not all(checks):raise ValueError('live actor and initial engine policy differ')
        del live_weights,expected,active_bundle
        from tools.reports.context_replica_macro_client import connect_pool
        from tools.reports.context_dual_session_candidate import DualContextRolloutSession,RuntimeReferenceCache,bind_candidates
        from multimodal_jev.context_policy_artifacts import export_native_adapter
        signal=[None]
        if rank==0:
            try:
                current_path=(Path(c['resume_root'])/commit['joint_policy_directory'] if start_update
                              else Path(c['source_policy']))
                current_manifest=json.loads((current_path/'MANIFEST.json').read_text())
                current=read_execution_bundle(current_path,expected_policy_id=current_manifest['policy_id'],
                    expected_contract=c['source_contract'],expected_base_id=c['base_id'])
                if start_update and file_hash(current_path/'MANIFEST.json')!=commit['joint_policy_manifest_sha256']:
                    raise ValueError('resume joint policy manifest differs')
                from tools.reports.context_cached_reference_session import load as load_reference
                if set(c['reference_dataset']['population'])!=set(full_population):
                    raise ValueError('complete TRAIN reference coverage required')
                cache=load_reference(c['reference_dataset'],runtime_identity=runtime_identity,
                    policy_id=c['source_policy_id'],base_id=c['base_id'],contract=c['source_contract'],
                    candidate_ids=[v['id'] for v in task['schema']['candidates']])
                client=connect_pool(c['macro_client'],bundle=current)
                reference=dict(snapshot=c['source_policy'],policy_id=c['source_policy_id'],adapter=c['reference_adapter'])
                initial=dict(snapshot=str(current_path),policy_id=current['policy_id'],adapter=c['initial_adapter'])
                candidate_ids=[v['id'] for v in task['schema']['candidates']]
                session=DualContextRolloutSession(client,cache,reference=reference,initial_policy=initial,
                    runtime_identity=runtime_identity,candidate_ids=candidate_ids,arm=c['dual_arm'],
                    actor_options=options if 'objective_config' in c else None,
                    audio_token_id=base.config.audio_token_id,algorithm=c['algorithm'],start_update=start_update,total_updates=c['total_updates'])
                signal[0]=dict(error=None,policy_id=current['policy_id'])
            except Exception as exc:signal[0]=dict(error=repr(exc))
        dist.broadcast_object_list(signal,src=0)
        if signal[0]['error']:raise ValueError(signal[0]['error'])
        current_policy_id=signal[0]['policy_id']
        records=[]
        try:
            for index in range(start_update,stop_update):
                step_begin=time.monotonic();progress('prepare_batch',update=index)
                cfg.jev_policy_id=current_policy_id
                signal=[None];parts=None
                if rank==0:
                    try:
                        items,metadata=stream.consume(index,cfg.jev_policy_id)
                        items=bind_candidates(items,metadata,candidate_ids)
                        items,options,rollout_receipt=session.prepare(items,update=index)
                        if options != settings['dual_objective']:
                            raise ValueError('sealed rollout configuration differs from the actor')
                        atomic_json(root/f'rollout-{index:04d}.json',rollout_receipt)
                        lengths=[(p['inputs']['input_ids'].shape[-1],p['inputs']['input_features'].shape[-1]) for p in items]
                        planned=rank_indices(batches[index],lengths,c['batch_config'])
                        ids=[i for group in planned for i in group]
                        complete=make_batch([items[i] for i in ids],**options)
                        groups=[{k:p[k] for k in ('request_id','policy_id','labels','observed','candidate_ids','dual_credit','dual_identity','dual_options','reference_probability','reference_policy_id','request_weight','real_request','global_rows','global_real_requests')}
                            for p in complete.non_tensor_batch['multi_modal_inputs']]
                        commit_file(root/f'groups-{index:04d}.pt',lambda f:torch.save(groups,f))
                        parts=list(complete.split(layout['local_batch']))
                        atomic_json(root/f'batch-{index:04d}.json',dict(policy_id=cfg.jev_policy_id,
                            rank_indices=planned,records=metadata))
                        signal[0]=dict(error=None,timing=stream.last_timing,options=options,rollout=rollout_receipt)
                    except Exception as exc:signal[0]=dict(error=repr(exc))
                dist.broadcast_object_list(signal,src=0)
                if signal[0]['error']:raise ValueError(signal[0]['error'])
                cfg.jev_context_objective=signal[0]['options']
                incoming=[None];dist.scatter_object_list(incoming,parts,src=0);batch=incoming[0]
                preparation=time.monotonic()-step_begin
                actor=build_actor(cfg,wrapped,opt,collation=collation)
                progress('native_update',update=index)
                torch.npu.reset_peak_memory_stats();torch.npu.synchronize();started=time.monotonic()
                if c.get('objective_trace'):dual_actor.begin_objective_trace(out/'objective-traces',index)
                metrics=actor.update_policy(batch);torch.npu.synchronize();seconds=time.monotonic()-started
                if c.get('objective_trace'):dual_actor.finish_objective_trace()
                audit=audit_adam_shards(named,opt,index+1)
                save_now=not formal or (index+1) in c['evaluation_steps']
                if frozen!={n:tensor_identity(p.detach().cpu()) for n,p in wrapped.named_parameters() if not p.requires_grad}:
                    raise ValueError('frozen backbone changed')
                if save_now:
                    state=capture(named,opt,rank=rank,world_size=world,completed_updates=index+1,contract=contract)
                    commit_file(out/f'checkpoint-{index+1:04d}.pt',lambda f:torch.save(state,f))
                # Every optimizer update exports only trainable LoRA/head tensors.
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
                    published=dict(updates=index+1,contract=contract,
                        execution_file=f'execution-{index+1:04d}.pt',
                        joint_policy=joint,joint_policy_directory=f'policy-{index+1:04d}',
                        joint_policy_manifest_sha256=file_hash(root/f'policy-{index+1:04d}/MANIFEST.json'),
                        execution_sha256=file_hash(root/f'execution-{index+1:04d}.pt'))
                    if formal:commit_json(root/f'policy-{index+1:04d}-committed.json',published)
                    if save_now:
                        commit_json(root/f'checkpoint-{index+1:04d}-committed.json',dict(published,
                            rank_shards={f'rank-{r}/checkpoint-{index+1:04d}.pt':
                                file_hash(root/f'rank-{r}/checkpoint-{index+1:04d}.pt') for r in range(world)}))
                dist.barrier()
                activation=[None]
                if rank==0:
                    try:
                        target=root/f'policy-{index+1:04d}'
                        updated=read_execution_bundle(target,expected_policy_id=joint['policy_id'],
                            expected_contract=c['source_contract'],expected_base_id=c['base_id'])
                        adapter=root/f'adapter-{index+1:04d}'
                        export_native_adapter(adapter,updated)
                        receipt=session.accept_actor_update(update=index,next_policy=dict(
                            snapshot=str(target),policy_id=joint['policy_id'],adapter=str(adapter)))
                        activation[0]=dict(error=None,policy_id=joint['policy_id'],receipt=receipt)
                        atomic_json(root/f'activation-{index+1:04d}.json',activation[0])
                    except Exception as exc:activation[0]=dict(error=repr(exc))
                dist.broadcast_object_list(activation,src=0)
                if activation[0]['error']:raise ValueError(activation[0]['error'])
                current_policy_id=activation[0]['policy_id']
                record=dict(update=index+1,preparation_seconds=preparation,actor_seconds=seconds,
                    total_seconds=time.monotonic()-step_begin,input_service=signal[0]['timing'],metrics=metrics,
                    rollout=signal[0]['rollout'],next_policy_id=current_policy_id,
                    behavior_parity_history=actor.behavior_parity_history,
                    real_queries=actor.last_admission['global_real_requests'],
                    peak_allocated=torch.npu.max_memory_allocated(),checkpoint=save_now,optimizer_audit=audit)
                records.append(record);atomic_json(out/f'update-{index+1:04d}.json',record)
            atomic_json(out/'result.json',dict(passed=True,start_update=start_update,stop_update=stop_update,
                executions=executions,scope='Candidate continuous context RL; independent collection required; no efficacy claim'))
            dist.barrier()
            if rank==0:atomic_json(root/'result.json',dict(passed=True,start_update=start_update,
                stop_update=stop_update,schedule_updates=layout['total_updates'],learning_population=c['batch_config']['source_queries'],
                engineering_prefix=c['engineering_prefix']))
        finally:
            if stream is not None:
                stream.close();stream=None
            if client is not None:
                client.close();client=None
    finally:
        try:
            if stream is not None:stream.close()
        finally:
            try:
                if client is not None:client.close()
            finally:dist.destroy_process_group()
