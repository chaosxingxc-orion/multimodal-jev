#!/usr/bin/env python3
"""Independent state, sampling and probability audit for continuous DP4 fixture."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,output):
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.simple_sampled_checkpoint import reconstruct
    from multimodal_jev.simple_rl_schedule import SimpleRLSchedule
    from multimodal_jev.simple_audio_batch import compile_batch
    from multimodal_jev.simple_sampled_rollout import actor_batch
    from multimodal_jev.simple_resident_acceptance import compare_updated_batch
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.slurp_training import content_hash
    from verify_simple_sampled_restore import run as recovery
    from verify_slurp_simple_parity import verify as verify_gate
    root=Path(config['training_root'])
    declared=json.loads((root/'config.json').read_text())
    stage2='stage2_initialization' in declared
    fixture_resumed=declared.get('sampled_probe',{}).get('fixture_initialization')=='restore_continuous_then_one_update'
    if not stage2 and ('audit_mode' in config or 'completed_updates' in config):
        raise ValueError('prefix mode is only defined for formal Stage2')
    if stage2:
        from multimodal_jev.simple_stage2_audit import source
        restored=source(config,verify_parity=verify_gate)
    elif fixture_resumed:
        from verify_simple_continuous_restore import run as recover_continuous
        restored=recover_continuous(config,output)
    else:
        restored=recovery(config,output,expected_mode='restore_updated_rl_then_resident_loop')
    files=dict(restored['files']);root=Path(config['training_root'])
    def read(path):
        path=Path(path);files[str(path)]=file_hash(path);return json.loads(path.read_text())
    train=read(root/'config.json');plan=train['stage2_initialization' if stage2 else 'sampled_probe']
    if stage2:
        initial=restored['initialization']
        fixed=initial['reference_policy_id'];behavior=initial['policy_id']
        from multimodal_jev.simple_stage2_audit import audit_boundary
        terminal,prefix=audit_boundary(config)
        resumed=restored.get('recovery')
        start=resumed['completed_updates'] if resumed else 0
        if resumed:behavior=resumed['policy_id']
        names=restored['names'];update_range=range(start+1,terminal+1)
    elif fixture_resumed:
        from multimodal_jev.simple_continuous_resume import authenticate
        resumed=authenticate(train);files.update(resumed['files'])
        fixed=resumed['reference_policy_id'];behavior=resumed['policy_id']
        fixed_gate=verify_gate(plan['reference_gate']);files.update(fixed_gate['files'])
        if (resumed['completed_updates']!=3 or plan['stop_update']!=4
                or fixed!=train['base_id']+':'+fixed_gate['adapter_id']
                or plan['algorithm']!=train['continuous_resume']['algorithm']):
            raise ValueError('fixed-reference engineering step3 to step4 required')
        names=resumed['states'][0]['names'];update_range=(4,);terminal=4;prefix=False
    else:
        initial=verify_gate(plan['behavior_gate']);fixed_gate=verify_gate(plan['reference_gate'])
        files.update(initial['files']);files.update(fixed_gate['files'])
        fixed=train['base_id']+':'+fixed_gate['adapter_id'];behavior=train['base_id']+':'+initial['adapter_id']
        if plan['stop_update']!=3:raise ValueError('two-update continuous fixture required')
        initial_state=Path(initial['training_root'])/'run/rl-rank-0/checkpoint-0001.pt'
        files[str(initial_state)]=file_hash(initial_state)
        names=torch.load(initial_state,map_location='cpu',weights_only=True)['names']
        update_range=(2,3);terminal=3;prefix=False
    policies=({int(k):v for k,v in resumed['prefix_audit']['policies'].items()} if stage2 and resumed
              else {0:behavior} if stage2 else {})
    schedule=SimpleRLSchedule(Path(train['epoch_file']).parent,
        manifest_sha256=train['schedule_sha256'],release_sha256=train['release_sha256'])
    files.update(schedule.files)
    population,task,_=load(train['release'],train['release_sha256'],'learning',purpose='optimize')
    population_check=schedule.validate_population(population)
    processor=Qwen3ASRProcessor.from_pretrained(train['model_path'],local_files_only=True)
    checks=list(resumed['prefix_audit']['checks']) if stage2 and resumed else []
    for updates in update_range:
        step=updates-1;slots,cursor=schedule.at(step)
        entries,_,compiled=compile_batch({r['id']:r for r in population},task,slots,processor,
                                         policy_id=behavior,global_batch=128)
        behavior_result=read(root/'run'/f'behavior-{step:04d}.json')
        reference=read(root/'run'/f'reference-{step:04d}.json')
        _,sampling=actor_batch(entries,behavior_result,reference,policy_id=behavior,
            reference_policy_id=fixed,seed=train['seed'],update=step)
        inputs=read(root/'run'/f'inputs-{step:04d}.json')
        if (inputs['compiled']!=compiled or inputs['sampling']!=sampling
                or inputs['next_optimizer_update']!=updates):
            raise ValueError('independent continuous sampling replay differs')
        adapter=root/'run'/f'loop-adapter-{updates:04d}';adapter_id,assets=checkpoint_identity(adapter)
        files.update({str(adapter/k):v for k,v in assets.items()})
        policy=train['base_id']+':'+adapter_id
        committed=read(root/'run'/f'checkpoint-{updates:04d}-committed.json')
        paths=[root/'run'/f'loop-rank-{r}'/f'checkpoint-{updates:04d}.pt' for r in range(4)]
        hashes={str(p):file_hash(p) for p in paths};files.update(hashes)
        if (committed.get('passed') is not True or committed['files']!=hashes or committed['reference_policy_id']!=fixed
                or committed['cursor']!=dict(schema='jev-span-training-session-v1',completed_updates=updates,
                    policy_id=policy,schedule_sha256=schedule.sha256,checkpoint_id=content_hash(hashes))):
            raise ValueError('continuous durable checkpoint/cursor differs')
        states=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
        for state in states:
            if state['schedule_sha256']!=schedule.sha256 or state['next_update']!=updates:
                raise ValueError('continuous state schedule differs')
        tensors=reconstruct(states,load_file(str(adapter/'adapter_model.safetensors')),algorithm=plan['algorithm'],
            adapter_id=adapter_id,reference_policy_id=fixed,lr=train['lr'],expected_names=names,
            completed_updates=updates,expected_stage='simple_stage2_rl' if stage2 else 'continuous_rl_engineering')
        if updates==terminal:
            # A byte-consistent export is not proof that PEFT loads the same
            # parameters. Independently load the terminal adapter on CPU.
            from peft import PeftModel
            from multimodal_jev.asr import load_runtime
            from multimodal_jev.span_adapter_export import adapter_key
            _,base=load_runtime(train['model_path'],'cpu','bfloat16')
            loaded=PeftModel.from_pretrained(base,adapter,is_trainable=False,
                                            autocast_adapter_dtype=False).eval()
            actual={adapter_key(n):p.detach().cpu() for n,p in loaded.named_parameters()
                    if 'lora_A.' in n or 'lora_B.' in n}
            if (actual.keys()!=tensors.keys() or any(actual[k].dtype!=torch.bfloat16
                    or not torch.equal(actual[k],v) for k,v in tensors.items())):
                raise ValueError('terminal continuous PEFT reload differs')
            del loaded,base,actual
        resident=[read(root/'run'/f'loop-rank-{r}'/f'after-{updates:04d}.json') for r in range(4)]
        rollout=read(root/'run'/f'rollout-after-{updates:04d}.json')
        check=compare_updated_batch(resident,rollout,sampling,policy_id=policy,updates=updates)
        for rank in range(4):
            report=read(root/'run'/f'loop-rank-{rank}'/f'update-{updates:04d}.json')
            if (report['completed_updates']!=updates or report['next_update']!=updates
                    or report['schedule_sha256']!=schedule.sha256 or report['behavior_policy_id']!=behavior
                    or report['policy_id']!=policy or report['coefficient']!=cursor['coefficient']
                    or report['real_requests']!=(cursor['real_requests'] if stage2 else 128) or report['checkpoint']!=committed
                    or any(report['acceptance'].get(k)!=v for k,v in check.items())):
                raise ValueError('continuous per-rank result differs')
            norms=report['metrics']['actor/grad_norm']
            if len(norms)!=1 or not 0<float(norms[0])<float('inf'):
                raise ValueError('finite nonzero native update gradient required')
        checks.append(check);behavior=policy
        if stage2:policies[updates]=policy
    for rank in (() if prefix else range(4)):
        result=read(root/'run'/f'loop-rank-{rank}'/'result.json')
        if (result['passed'] is not True or result['completed_updates']!=terminal
                or result['phase']!='accepted' or result['policy_id']!=behavior
                or result['schedule_sha256']!=schedule.sha256 or result['reference_policy_id']!=fixed):
            raise ValueError('complete continuous rank required')
    selection=None
    if stage2:
        from multimodal_jev.simple_stage2_audit import selections
        selection=selections(root,train,policies,files,through_update=terminal,
                             **(dict(history=resumed['selection_history']) if resumed else {}))
    result=dict(passed=True,updates=terminal,new_updates=terminal if stage2 else 1 if fixture_resumed else 2,ranks=4,checks=checks,files=files,
        population_check=population_check,exact_master_export=True,exact_peft_reload=True,
        exact_pre_update_recovery=True,
        scope=('Independent stopped Stage2 prefix, initial masters/freshAdam, every accepted update and due selection through the specified cursor; suffix excluded, not a completed training run or launch authorization.' if prefix else
               'Independent full Stage2 initial masters/freshAdam, 634 scheduled updates, all checkpoint exports and actor/vLLM parity, 11 full-selection rescoring points; execution still requires actual evidence.' if stage2 else
               'Independent continuous fixture shard/export, full scheduled sampling and actor/vLLM parity; task efficacy remains separate.'))
    if fixture_resumed:
        result.update(resumed_from_update=3,exact_master_adam=True,
            recovery_probability_requests=restored['requests'],recovery_parity=restored['parity'],
            scope='Fresh-process continuous step3 exact state/full128 recovery followed by scheduled step4, export and actor/vLLM parity; engineering only.')
    if stage2:result.update(stage='simple_stage2_rl',selection=selection,
        initial_state_checks=resumed['prefix_audit']['initial_state_checks'] if resumed else restored['checks'],
        resume_state_checks=restored['checks'] if resumed else [],policies=policies,
        segment_start_update=start,segment_new_updates=terminal-start,
        audit_mode='stopped_prefix' if prefix else 'terminal',complete_training=not prefix)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'state-verified.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
