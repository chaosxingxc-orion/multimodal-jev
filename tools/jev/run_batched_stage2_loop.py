#!/usr/bin/env python3
"""Candidate full-data DP4 RL loop using a separate persistent TP1 service.

Consumes authenticated final Stage1 initialization with fresh Adam and the full
634-update task schedule. Requires an admitted candidate RPC worker; this module
alone neither launches nor grants lifecycle acceptance.
"""
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash


def continuous(plan, **ctx):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from safetensors.torch import save_file
    from multimodal_jev.simple_rl_loop import run
    from multimodal_jev.simple_rl_schedule import SimpleRLSchedule
    from multimodal_jev.simple_resident_batch import ResidentSimpleBatch
    from multimodal_jev.simple_audio_batch import renderer_for_policy
    from multimodal_jev.span_rollout_rpc import RolloutClient
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.span_hybrid_actor import make_batch
    from multimodal_jev.span_batched_hybrid_actor import build_batched_actor
    from multimodal_jev.span_batch_training_config import batch_layout, reorder_global_items
    from multimodal_jev.span_training_session import SpanTrainingSession
    from multimodal_jev.span_adapter_export import adapter_key
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.local_checkpoint import commit_file, commit_json, seal_adapter
    from multimodal_jev.simple_resident_acceptance import compare_updated_batch
    from probe_fsdp_lora_npu import digest
    config=ctx['config'];root=ctx['root'];rank=ctx['rank']
    layout=batch_layout(config)
    rollout_devices=plan['rollout_physical_devices']
    if (not isinstance(rollout_devices,list) or len(rollout_devices)!=1
            or type(rollout_devices[0]) is not int or not 0<=rollout_devices[0]<=7
            or rollout_devices[0] in config['physical_devices']):
        raise ValueError('resident service requires a separate authorized physical device')
    model=ctx['wrapped'];optimizer=ctx['optimizer'];device=ctx['device']
    out=root/f'loop-rank-{rank}';out.mkdir(exist_ok=False)
    def gather(value):
        peers=[None]*4;dist.all_gather_object(peers,value);return peers
    def collective(name, fn):
        error=None;result=None
        try:result=fn()
        except Exception as exc:error=repr(exc)
        errors=gather(error)
        if any(errors):raise ValueError(name+': '+str(errors))
        return result
    def rank0(name, fn):
        result=collective(name,lambda:fn() if rank==0 else None)
        values=[result];dist.broadcast_object_list(values,src=0);return values[0]

    if 'initialization' not in ctx or 'fixture_recovery' in ctx:
        raise ValueError('candidate final Stage1 initialization required')
    stage2=True;recovering='recovery' in ctx
    if recovering:
        from multimodal_jev.batched_stage2_resume import continuous_plan
        initial=continuous_plan(plan,ctx['recovery'],optimizer,
                                base_id=ctx['base_id'],adapter_id=ctx['adapter_id'])
    else:
        from multimodal_jev.batched_stage2_initialization import continuous_plan
        initial=continuous_plan(plan,ctx['initialization'],optimizer,
                                base_id=ctx['base_id'],adapter_id=ctx['adapter_id'])
    fixed_config=json.loads(Path(ctx['initialization']['backend_config']).read_text())
    fixed_config['candidate_profile']=config['candidate_profile']
    policy_config=dict(fixed_config)
    if recovering:policy_config.update(adapter=ctx['recovery']['adapter'],adapter_id=ctx['adapter_id'])
    fixed=initial['reference_policy_id'];policy=initial['policy_id']
    start_update=initial['start_update'];stop_update=initial['stop_update']
    for cfg in (fixed_config,policy_config):cfg['physical_devices']=plan['rollout_physical_devices']
    schedule=collective('full schedule',lambda:SimpleRLSchedule(Path(config['epoch_file']).parent,
        manifest_sha256=config['schedule_sha256'],release_sha256=config['release_sha256']))
    population,task,_=collective('learning population',lambda:load(config['release'],config['release_sha256'],
                                                               'learning',purpose='optimize'))
    collective('population weights',lambda:schedule.validate_population(population))
    client=None;collector=None;selection=None
    def connect():
        nonlocal client,collector,selection
        # Host launch record must be fixed by this immutable training snapshot;
        # ready.json alone is not evidence of an admitted external worker.
        service_path=Path(plan['service_launch'])
        if config['pins'].get(str(service_path))!=file_hash(service_path):
            raise ValueError('pinned exact resident service launch required')
        service=json.loads(service_path.read_text())
        from multimodal_jev.resident_service_admission import validate
        observed={}
        for key in ('service_inspect','service_ready'):
            path=Path(plan[key])
            if config['pins'].get(str(path))!=file_hash(path):
                raise ValueError('pinned host service observation required')
            observed[key]=json.loads(path.read_text())
        service_config=Path(service['remote_root'])/'config.json'
        if config['pins'].get(str(service_config))!=file_hash(service_config):
            raise ValueError('pinned resident service config required')
        if (observed['service_ready'].get('candidate_profile')!=config['candidate_profile']
                or json.loads(service_config.read_text())['policy'].get('candidate_profile')!=config['candidate_profile']):
            raise ValueError('admitted service must execute the same candidate profile')
        validate(service,observed['service_inspect'],observed['service_ready'],
                 actor_devices=config['physical_devices'],config_sha256=file_hash(service_config))
        if (service['physical_devices']!=rollout_devices
                or service['socket_path']!=plan['socket_path']
                or service['exchange']!=plan['exchange']
                or service['image']!='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
                or not service['container_id']):
            raise ValueError('resident service launch scope differs')
        client=RolloutClient(plan['socket_path'],plan['exchange'])
        collector=ResidentSimpleBatch(client,{r['id']:r for r in population},task,ctx['processor'],
                                      fixed_config,seed=config['seed'])
        if stage2:
            from multimodal_jev.simple_resident_selection import ResidentSimpleSelection, SELECTION_UPDATES
            if plan.get('selection_updates')!=list(SELECTION_UPDATES):
                raise ValueError('complete preregistered selection schedule required')
            inputs=Path(plan['selection_inputs'])
            if config['pins'].get(str(inputs))!=file_hash(inputs):
                raise ValueError('pinned selection processor artifact required')
            rows,selection_task,_=load(config['release'],config['release_sha256'],'selection',purpose='evaluate')
            selection=ResidentSimpleSelection(client,ctx['processor'],rows,selection_task,
                inputs=inputs,inputs_sha256=config['pins'][str(inputs)],release_sha256=config['release_sha256'],
                output=root/'evaluation')
            if recovering:
                selection.restore_history(ctx['recovery']['selection_history'],completed_updates=start_update)
    collective('connect admitted resident worker',lambda:connect() if rank==0 else None)
    cfg=ctx['actor_config'];cfg.jev_advantage_algorithm=plan['algorithm']
    cfg.jev_kl_coefficient=.01;cfg.jev_reference_policy_id=fixed
    if cfg.ppo_micro_batch_size_per_gpu!=layout['microbatch_per_rank']:
        raise ValueError('candidate loop microbatch differs from initialized actor')
    actor=build_batched_actor(cfg,model,optimizer,collation=ctx['collation'])
    session=SpanTrainingSession(actor,policy_id=policy,completed_updates=start_update,gather=gather)
    frozen={n:digest(p) for n,p in model.named_parameters() if not p.requires_grad}
    pending={};latest_sampling={}

    def collect(slots,cursor,policy):
        chunks=None
        def prepare():
            nonlocal chunks
            if rank!=0:return
            if policy_config['base_id']+':'+policy_config['adapter_id']!=policy:
                raise ValueError('collector export policy differs')
            items,receipt=collector.collect(slots,policy_config=policy_config,completed_updates=cursor['update'])
            latest_sampling.clear();latest_sampling.update(receipt['sampling'])
            items,permutation=reorder_global_items(items,config)
            receipt['candidate_batch_layout']=layout
            receipt['candidate_rank_indices']=permutation
            batch=make_batch(items,stage='task',coefficient=cursor['coefficient'],reduction='real_request_mean',
                algorithm=plan['algorithm'],kl_coefficient=.01,reference_policy_id=fixed)
            chunks=list(batch.split(32))
            atomic_json(root/f'inputs-{cursor["update"]:04d}.json',receipt)
            atomic_json(root/f'behavior-{cursor["update"]:04d}.json',collector.last_behavior)
            atomic_json(root/f'reference-{cursor["update"]:04d}.json',collector.last_reference)
        collective('prepare resident actor batch',prepare)
        incoming=[None];dist.scatter_object_list(incoming,chunks,src=0)
        return incoming[0]

    def export(session,batch,metrics):
        def check():
            if len(optimizer.state)!=112:raise ValueError('complete LoRA Adam state required')
            for p in optimizer.param_groups[0]['params']:
                s=optimizer.state[p]
                if (p.dtype!=torch.float32 or not torch.isfinite(p).all()
                        or s['step'].item()!=session.completed_updates
                        or any(s[k].dtype!=torch.float32 or not torch.isfinite(s[k]).all()
                               for k in ('exp_avg','exp_avg_sq'))):
                    raise ValueError('continuing finite FP32 master/Adam required')
            norms=metrics['actor/grad_norm']
            if len(norms)!=1 or not 0<float(norms[0])<float('inf'):
                raise ValueError('finite nonzero continuous update gradient required')
            if frozen!={n:digest(p) for n,p in model.named_parameters() if not p.requires_grad}:
                raise ValueError('frozen base changed')
        collective('post-update master audit',check)
        with FSDP.summon_full_params(model,writeback=False,rank0_only=False):
            tensors={adapter_key(n):p.detach().cpu().to(torch.bfloat16).clone()
                for n,p in ctx['adapted'].named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
        hashes={n:digest(p) for n,p in tensors.items()}
        if any(h!=hashes for h in gather(hashes)):raise ValueError('rank export differs')
        dest=root/f'loop-adapter-{session.completed_updates:04d}'
        def save():
            dest.mkdir();save_file(tensors,str(dest/'adapter_model.safetensors'))
            ctx['adapted'].peft_config['default'].save_pretrained(dest)
            (dest/'adapter_model.safetensors').chmod(0o644);seal_adapter(dest)
            return checkpoint_identity(dest)[0]
        adapter=rank0('publish adapter',save)
        pending.update(adapter=str(dest),adapter_id=adapter)
        return ctx['base_id']+':'+adapter

    def verify(policy,updates,batch):
        def forward():
            rows=[]
            with torch.no_grad():
                from multimodal_jev.span_audio_collate import forward_span_audio_batch
                payloads=list(batch.non_tensor_batch['multi_modal_inputs'])
                for offset in range(0,len(payloads),layout['microbatch_per_rank']):
                    group=payloads[offset:offset+layout['microbatch_per_rank']]
                    outputs=forward_span_audio_batch(model,group,device=device,dtype=torch.bfloat16,**ctx['collation'])
                    for payload,probs in zip(group,outputs,strict=True):
                        rows.append(dict(request_id=payload['request_id'],candidate_logprobs=[p[0].cpu().tolist() for p in probs]))
            return rows
        local=collective('post-update actor probabilities',forward);peers=gather(local)
        collective('write resident probabilities',lambda:atomic_json(out/f'after-{updates:04d}.json',local))
        def compare():
            updated=dict(policy_config,**pending)
            rendered=renderer_for_policy(collector.last_renderer,policy)
            updated['population']={'requests':len(rendered)}
            result=client.rollout(updated,rendered)
            check=compare_updated_batch(peers,result,latest_sampling,policy_id=policy,updates=updates)
            path=root/f'rollout-after-{updates:04d}.json';atomic_json(path,result)
            return dict(**check,files={str(path):file_hash(path)})
        receipt=rank0('updated full-batch parity',compare)
        policy_config.update(pending)
        return receipt

    def checkpoint(session,cursor):
        updates=session.completed_updates
        if stage2:
            def execution_evidence():
                executions=ctx['execution_dtypes']
                if (len(executions)!=112 or any(v!=['torch.bfloat16','torch.bfloat16']
                                               for v in executions.values())):
                    raise ValueError('complete checkpoint BF16 execution evidence required')
                commit_json(out/f'execution-{updates:04d}.json',dict(rank=rank,updates=updates,
                    execution_dtypes=executions))
            collective('checkpoint execution evidence',execution_evidence)
        state=dict(rank=rank,world_size=4,updates=updates,stage='simple_stage2_rl' if stage2 else 'continuous_rl_engineering',algorithm=plan['algorithm'],
            names=[n for n,p in model.named_parameters() if p.requires_grad],
            parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
            optimizer=optimizer.state_dict(),adapter_id=pending['adapter_id'],reference_policy_id=fixed,
            schedule_sha256=schedule.sha256,next_update=updates,
            candidate_profile=config['candidate_profile'],batch_layout=layout,batch_order=config['batch_order'])
        path=out/f'checkpoint-{updates:04d}.pt'
        collective('persist shards',lambda:commit_file(path,lambda stream:torch.save(state,stream)))
        paths=[root/f'loop-rank-{r}'/path.name for r in range(4)]
        hashes=collective('checkpoint file hashes',lambda:{str(p):file_hash(p) for p in paths})
        from multimodal_jev.slurp_training import content_hash
        checkpoint_id=content_hash(hashes)
        session_cursor=session.checkpoint_cursor(schedule_sha256=schedule.sha256,checkpoint_id=checkpoint_id)
        receipt=dict(passed=True,files=hashes,cursor=session_cursor,reference_policy_id=fixed)
        rank0('commit accepted checkpoint',lambda:commit_json(root/f'checkpoint-{updates:04d}-committed.json',receipt))
        return receipt

    def selection_collective(updates,evidence):
        iterator=None
        def start():
            nonlocal iterator
            if rank==0:iterator=selection.evaluate_steps(policy_config,updates=updates,checkpoint=evidence)
        collective('selection start',start)
        try:
            while True:
                event=rank0('selection bounded batch',lambda:next(iterator))
                if event['done']:return event['receipt']
        finally:
            if iterator is not None:iterator.close()

    def evaluate(session,report):
        if not stage2:return
        from multimodal_jev.simple_resident_selection import SELECTION_UPDATES
        updates=session.completed_updates
        if updates not in SELECTION_UPDATES:return
        checkpoint=report['checkpoint']
        if checkpoint is None:raise ValueError('selection requires committed current state')
        evidence=dict(passed=checkpoint['passed'],policy_id=session.policy_id,updates=updates,
            files=dict(checkpoint['files']))
        commit_path=root/f'checkpoint-{updates:04d}-committed.json'
        evidence['files'][str(commit_path)]=file_hash(commit_path)
        for name in ('adapter_config.json','adapter_model.safetensors'):
            path=Path(pending['adapter'])/name;evidence['files'][str(path)]=file_hash(path)
        receipt=selection_collective(updates,evidence)
        atomic_json(out/f'selection-{updates:04d}.json',receipt)

    try:
        if stage2 and not recovering:
            evidence=dict(passed=True,policy_id=policy,updates=0,files=ctx['initialization']['files'])
            initial_selection=selection_collective(0,evidence)
            atomic_json(out/'selection-0000.json',initial_selection)
        result=run(session,schedule,stop_update=stop_update,collect=collect,export=export,verify=verify,
            checkpoint=checkpoint,checkpoint_interval=1,
            record=lambda r:atomic_json(out/f'update-{r["completed_updates"]:04d}.json',r),
            evaluate=evaluate if stage2 else None)
        atomic_json(out/'result.json',dict(**result,passed=True,reference_policy_id=fixed,
            scope=('Full scheduled Stage2 using final Stage1 and fresh Adam; independent terminal audit pending.' if stage2 else
                   'Scheduled audio updates using resident DP4 actor/Adam and separate TP1 service; independent terminal audit pending.')))
    finally:
        if client is not None:client.close()
