#!/usr/bin/env python3
"""Second full-audio RL update from accepted first-update state and new behavior.

Engineering only: preserves Adam, uses schedule batch1 and immutable original
reference. The new export awaits independent acceptance; no training promotion.
"""
import argparse,json,os,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def continue_update(plan, **ctx):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from safetensors.torch import save_file
    from multimodal_jev.simple_next_behavior import next_batch
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.simple_rl_schedule import SimpleRLSchedule
    from multimodal_jev.span_hybrid_actor import make_batch,build_actor
    from multimodal_jev.span_hybrid_objective import sft_coefficient
    from multimodal_jev.span_training_session import SpanTrainingSession
    from multimodal_jev.span_adapter_export import adapter_key
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.local_checkpoint import commit_file,commit_json,seal_adapter
    from multimodal_jev.span_fsdp_resume import restore_local_shard
    from probe_fsdp_lora_npu import digest
    from verify_slurp_simple_parity import verify
    config=ctx['config'];root=ctx['root'];rank=ctx['rank'];model=ctx['wrapped'];optimizer=ctx['optimizer']
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
    current=collective('accepted behavior',lambda:verify(plan['behavior_gate']))
    reference=collective('accepted reference',lambda:verify(plan['reference_gate']))
    policy=ctx['base_id']+':'+ctx['adapter_id'];fixed=ctx['base_id']+':'+reference['adapter_id']
    if (current['updates']!=1 or current['adapter_id']!=ctx['adapter_id']
            or current['sampled_resident_parity'] is None or reference['updates']!=2
            or current['requests']!=256 or reference['requests']!=256 or policy==fixed):
        raise ValueError('accepted updated behavior and original reference required')
    previous=json.loads((Path(current['training_root'])/'config.json').read_text())
    if previous['sampled_probe']['behavior_gate']!=plan['reference_gate']:
        raise ValueError('fixed reference lineage changed')
    if plan.get('total_rl_updates')!=634:raise ValueError('registered full-data CE schedule required')
    coefficient=sft_coefficient(1,634)
    chunks=None
    def prepare():
        nonlocal chunks
        if rank!=0:return
        population,task,_=load(config['release'],config['release_sha256'],'learning',purpose='optimize')
        schedule=SimpleRLSchedule(Path(config['epoch_file']).parent,
            manifest_sha256=config['schedule_sha256'],release_sha256=config['release_sha256'])
        schedule.validate_population(population)
        slots,cursor=schedule.at(1)
        if cursor['coefficient']!=coefficient:raise ValueError('CE schedule cursor differs')
        items,receipt=next_batch({r['id']:r for r in population},task,slots,ctx['processor'],
            behavior_result=json.loads((Path(plan['behavior_gate'])/'run/vllm/result.json').read_text()),
            reference_result=json.loads((Path(plan['reference_gate'])/'run/vllm/result.json').read_text()),
            policy_id=policy,reference_policy_id=fixed,completed_updates=1,seed=config['seed'])
        batch=make_batch(items,stage='task',coefficient=coefficient,reduction='real_request_mean',
            algorithm=plan['algorithm'],kl_coefficient=.01,reference_policy_id=fixed)
        chunks=list(batch.split(32));atomic_json(root/'rl-inputs.json',receipt)
    collective('next full128 audio batch',prepare)
    incoming=[None];dist.scatter_object_list(incoming,chunks,src=0);batch=incoming[0]
    def check_adam(cursor):
        if len(optimizer.state)!=112:raise ValueError('complete continuing Adam required')
        for p in optimizer.param_groups[0]['params']:
            s=optimizer.state[p]
            if (p.dtype!=torch.float32 or not torch.isfinite(p).all() or s['step'].item()!=cursor
                    or any(s[k].dtype!=torch.float32 or not torch.isfinite(s[k]).all() for k in ('exp_avg','exp_avg_sq'))):
                raise ValueError('finite continuing master/Adam cursor required')
    collective('Adam preserved before second update',lambda:check_adam(1))
    cfg=ctx['actor_config'];cfg.jev_advantage_algorithm=plan['algorithm']
    cfg.jev_kl_coefficient=.01;cfg.jev_reference_policy_id=fixed
    actor=build_actor(cfg,model,optimizer)
    session=SpanTrainingSession(actor,policy_id=policy,completed_updates=1,gather=gather)
    frozen={n:digest(p) for n,p in model.named_parameters() if not p.requires_grad}
    torch.npu.synchronize();start=time.monotonic();metrics=session.update(batch);torch.npu.synchronize()
    seconds=time.monotonic()-start
    def check():
        check_adam(2)
        if frozen!={n:digest(p) for n,p in model.named_parameters() if not p.requires_grad}:
            raise ValueError('frozen base changed')
        norms=metrics['actor/grad_norm']
        if len(norms)!=1 or not 0<float(norms[0])<float('inf'):raise ValueError('finite nonzero gradient required')
    collective('second update state',check)
    def probabilities():
        records=[]
        with torch.no_grad():
            for payload in batch.non_tensor_batch['multi_modal_inputs']:
                inputs={k:v.to(ctx['device'],dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                        for k,v in payload['inputs'].items()}
                logp=model(inputs,payload['positions'].to(ctx['device']),payload['candidates'])[1]
                records.append(dict(request_id=payload['request_id'],candidate_logprobs=[v[0].cpu().tolist() for v in logp]))
        return records
    after=collective('second policy resident probabilities',probabilities)
    atomic_json(out/'after.json',dict(records=after))
    with FSDP.summon_full_params(model,writeback=False,rank0_only=False):
        tensors={adapter_key(n):p.detach().cpu().to(torch.bfloat16).clone()
            for n,p in ctx['adapted'].named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
    hashes={n:digest(p) for n,p in tensors.items()}
    if any(h!=hashes for h in gather(hashes)):raise ValueError('cross-rank export differs')
    dest=root/'rl-adapter-step0002'
    def export():
        if rank==0:
            dest.mkdir();save_file(tensors,str(dest/'adapter_model.safetensors'))
            ctx['adapted'].peft_config['default'].save_pretrained(dest);seal_adapter(dest)
    collective('second export',export)
    adapter_id=checkpoint_identity(dest)[0];session.publish(ctx['base_id']+':'+adapter_id)
    state=dict(rank=rank,world_size=4,updates=2,stage='sampled_rl_engineering',algorithm=plan['algorithm'],
        names=[n for n,p in model.named_parameters() if p.requires_grad],
        parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
        optimizer=optimizer.state_dict(),adapter_id=adapter_id,reference_policy_id=fixed,behavior_policy_id=policy)
    collective('persist second shard',lambda:commit_file(out/'checkpoint-0002.pt',lambda stream:torch.save(state,stream)))
    def commit():
        if rank==0:commit_json(root/'rl-checkpoint-0002-committed.json',dict(adapter_id=adapter_id,world_size=4,
            files={str(p.relative_to(root)):file_hash(p) for p in [root/f'rl-rank-{r}/checkpoint-0002.pt' for r in range(4)]}))
    collective('commit second checkpoint',commit)
    optimizer.zero_grad(set_to_none=True)
    fresh=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],
        betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    restored=collective('restore second state',lambda:restore_local_shard(model,fresh,
        torch.load(out/'checkpoint-0002.pt',map_location='cpu',weights_only=True),rank=rank,world_size=4,updates=2))
    recovered=collective('restored second probabilities',probabilities)
    def compare():
        if recovered!=after:raise ValueError('second reload probabilities differ')
    collective('exact second reload',compare)
    atomic_json(out/'restored.json',dict(**restored,exact_resident_probabilities=True,requests=32,records=recovered))
    atomic_json(out/'result.json',dict(passed=True,rank=rank,updates=2,algorithm=plan['algorithm'],
        base_id=ctx['base_id'],adapter_id=adapter_id,reference_policy_id=fixed,behavior_policy_id=policy,
        metrics=metrics,coefficient=coefficient,admission=actor.last_admission,seconds=seconds,
        phase=session.phase,execution_dtypes=ctx['execution_dtypes'],
        scope='Second scheduled audio update from accepted updated behavior; original reference and continuing Adam. Awaiting independent checkpoint/probability acceptance.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text());rank=int(os.environ['RANK'])
    if (int(os.environ['WORLD_SIZE'])!=4 or int(os.environ['LOCAL_RANK'])!=rank
            or config['sampled_probe'].get('fixture_initialization')!='restore_updated_rl_then_second_update'
            or config['global_batch']!=128 or config.get('microbatch_per_rank',1)!=1):
        raise ValueError('explicit accepted next-update DP4 configuration required')
    from multimodal_jev.local_runtime_storage import require_training_storage
    from probe_simple_sampled_npu import restore_accepted_fixture
    require_training_storage(a.output);out=a.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    try:
        restore_accepted_fixture(config,a.output,rank,restore_only=True,after_restore=continue_update)
        atomic_json(out/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(out/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
