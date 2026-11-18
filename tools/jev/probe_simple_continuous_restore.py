#!/usr/bin/env python3
"""Fresh DP4 process restores continuous master/Adam and full128 probabilities.

The CLI only restores state/probabilities. An explicit after_restore callback lets
the engineering resume-update entry reuse this loader and the existing actor.
Uses independently accepted continuous source only; never promotes formal training.
"""
import argparse,json,os,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json


def run(config,root,rank,*,after_restore=None):
    import torch
    import torch_npu
    import torch.distributed as dist
    from datetime import timedelta
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision,ShardingStrategy
    from torch.distributed.fsdp.wrap import CustomPolicy
    from multimodal_jev.simple_continuous_resume import authenticate
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3,configure_actor_profile
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.span_model import NativeAudioSpanModel
    from multimodal_jev.span_fsdp import unshared_linear_modules
    from multimodal_jev.span_fsdp_resume import restore_local_shard
    from multimodal_jev.intent_resident_replay import compare_state
    from multimodal_jev.local_checkpoint import commit_file
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.simple_audio_batch import compile_batch
    from multimodal_jev.simple_rl_schedule import SimpleRLSchedule
    from multimodal_jev.simple_resident_acceptance import compare_updated_batch
    from multimodal_jev.slurp_official_training import load
    accepted=authenticate(config);updates=accepted['completed_updates']
    devices=sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))
    if devices!=['davinci'+str(i) for i in config['physical_devices']] or torch.npu.device_count()!=4:
        raise ValueError('fixed isolated physical DP4 required')
    if checkpoint_identity(config['model_path'])[0]!=config['base_id']:raise ValueError('base changed')
    torch.set_num_threads(2);torch.npu.set_device(rank);device=torch.device('npu',rank)
    torch.manual_seed(config['seed']);torch.npu.manual_seed_all(config['seed'])
    initialize_verl_preserving_qwen3();configure_actor_profile(config['runtime_profile'])
    dist.init_process_group('hccl',timeout=timedelta(seconds=900))
    out=root/f'rank-{rank}'
    try:
        processor,base=load_runtime(config['model_path'],device,'bfloat16')
        adapted=initialize_adapter(base,dict(config,initial_adapter=accepted['adapter'])).eval()
        model=NativeAudioSpanModel(adapted.get_base_model()).eval()
        for parameter in model.parameters():
            if parameter.requires_grad:parameter.data=parameter.data.float()
        executions={}
        def hook(name):
            def check(module,args):
                executions[name]=[str(module.weight.dtype),str(args[0].dtype)]
                if executions[name]!=['torch.bfloat16','torch.bfloat16']:raise ValueError('BF16 execution required')
            return check
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Linear) and ('lora_A.' in name or 'lora_B.' in name):module.register_forward_pre_hook(hook(name))
        units=unshared_linear_modules(model)
        wrapped=FSDP(model,auto_wrap_policy=CustomPolicy(lambda m:m in units),device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,sync_module_states=True,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32,
                cast_forward_inputs=True,keep_low_precision_grads=False),limit_all_gathers=True)
        optimizer=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=config['lr'],
            betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
        state=accepted['states'][rank]
        receipt=restore_local_shard(wrapped,optimizer,state,rank=rank,world_size=4,updates=updates)
        actual=dict(state,parameters={n:p.detach().cpu().clone() for n,p in wrapped.named_parameters() if p.requires_grad},optimizer=optimizer.state_dict())
        exact=compare_state(state,actual,rank=rank,updates=updates,continuous=True)
        commit_file(out/f'checkpoint-{updates:04d}.pt',lambda stream:torch.save(actual,stream))
        schedule=SimpleRLSchedule(Path(config['epoch_file']).parent,manifest_sha256=config['schedule_sha256'],release_sha256=config['release_sha256'])
        population,task,_=load(config['release'],config['release_sha256'],'learning',purpose='optimize')
        schedule.validate_population(population)
        source=Path(config['continuous_resume']['training_root'])/'run';step=updates-1
        inputs=json.loads((source/f'inputs-{step:04d}.json').read_text());sampling=inputs['sampling']
        slots,_=schedule.at(step)
        entries,_,compiled=compile_batch({r['id']:r for r in population},task,slots,processor,
            policy_id=sampling['policy_id'],global_batch=128)
        if compiled!=inputs['compiled']:raise ValueError('restored audio/input identity differs')
        records=[]
        with torch.no_grad():
            for index in range(rank*32,(rank+1)*32):
                item=entries[index]['item']
                batch={k:v.to(device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in item['inputs'].items()}
                logp=wrapped(batch,item['positions'].to(device),item['candidates'])[1]
                records.append(dict(request_id=sampling['sampling'][index]['actor_request_id'],candidate_logprobs=[p[0].cpu().tolist() for p in logp]))
        peers=[None]*4;dist.all_gather_object(peers,records)
        rollout=json.loads((source/f'rollout-after-{updates:04d}.json').read_text())
        parity=compare_updated_batch(peers,rollout,sampling,policy_id=accepted['policy_id'],updates=updates)
        if len(executions)!=112:raise ValueError('complete BF16 hooks required')
        atomic_json(out/'probabilities.json',records)
        atomic_json(out/'result.json',dict(passed=True,rank=rank,updates=updates,policy_id=accepted['policy_id'],
            reference_policy_id=accepted['reference_policy_id'],schedule_sha256=config['schedule_sha256'],
            checkpoint_id=accepted['checkpoint_id'],restore=receipt,exact_state=exact,parity=parity,execution_dtypes=executions,
            scope='Fresh-process continuous checkpoint restore and full128 recorded vLLM probability comparison; no optimizer update.'))
        if after_restore is not None:
            from probe_simple_sampled_npu import actor_config
            optimizer.zero_grad(set_to_none=True)
            after_restore(config['sampled_probe'],config=config,root=root,rank=rank,
                processor=processor,adapted=adapted,wrapped=wrapped,optimizer=optimizer,
                actor_config=actor_config(accepted['policy_id']),base_id=config['base_id'],
                adapter_id=accepted['adapter_id'],device=device,execution_dtypes=executions,
                fixture_recovery=accepted)
    finally:dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=4 or int(os.environ['LOCAL_RANK'])!=rank:raise ValueError('DP4 required')
    from multimodal_jev.local_runtime_storage import require_training_storage
    require_training_storage(a.output);out=a.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    try:
        run(json.loads(a.config.read_text()),a.output,rank)
        atomic_json(out/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(out/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
