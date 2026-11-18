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
    from verify_slurp_batched_parity import verify as verify_gate
    root=Path(config['training_root'])
    declared=json.loads((root/'config.json').read_text())
    stage2='stage2_initialization' in declared
    if not stage2 or 'sampled_probe' in declared:
        raise ValueError('candidate full-data Stage2 audit required')
    from multimodal_jev.span_batch_training_config import batch_layout, reorder_global_items
    layout=batch_layout(declared)
    from multimodal_jev.batched_stage2_audit import source
    restored=source(config,verify_parity=verify_gate)
    files=dict(restored['files']);root=Path(config['training_root'])
    def read(path):
        path=Path(path);files[str(path)]=file_hash(path);return json.loads(path.read_text())
    train=read(root/'config.json');plan=train['stage2_initialization' if stage2 else 'sampled_probe']
    initial=restored['initialization']
    fixed=initial['reference_policy_id'];behavior=initial['policy_id']
    from multimodal_jev.batched_stage2_audit import audit_boundary
    terminal,prefix=audit_boundary(config)
    resumed=restored.get('recovery')
    start=resumed['completed_updates'] if resumed else 0
    if resumed:behavior=resumed['policy_id']
    names=restored['names'];update_range=range(start+1,terminal+1)
    policies=({int(k):v for k,v in resumed['prefix_audit']['policies'].items()} if stage2 and resumed
              else {0:behavior} if stage2 else {})
    schedule=SimpleRLSchedule(Path(train['epoch_file']).parent,
        manifest_sha256=train['schedule_sha256'],release_sha256=train['release_sha256'])
    files.update(schedule.files)
    population,task,_=load(train['release'],train['release_sha256'],'learning',purpose='optimize')
    population_check=schedule.validate_population(population)
    processor=Qwen3ASRProcessor.from_pretrained(train['model_path'],local_files_only=True)
    recovery_policy=None
    if resumed:
        from multimodal_jev.batched_stage2_recovery_policy import prepare, verify_reports
        replay=prepare(train,resumed,processor)
        files.update(replay['binding']['files'])
        reports=[read(root/'run'/f'rank-{r}'/'restored-policy.json') for r in range(4)]
        recovery_policy=verify_reports(reports,replay)
    checks=list(resumed['prefix_audit']['checks']) if stage2 and resumed else []
    for updates in update_range:
        step=updates-1;slots,cursor=schedule.at(step)
        entries,_,compiled=compile_batch({r['id']:r for r in population},task,slots,processor,
                                         policy_id=behavior,global_batch=128)
        behavior_result=read(root/'run'/f'behavior-{step:04d}.json')
        reference=read(root/'run'/f'reference-{step:04d}.json')
        items,sampling=actor_batch(entries,behavior_result,reference,policy_id=behavior,
            reference_policy_id=fixed,seed=train['seed'],update=step)
        _,permutation=reorder_global_items(items,train)
        inputs=read(root/'run'/f'inputs-{step:04d}.json')
        if (inputs.get('candidate_batch_layout')!=layout
                or inputs.get('candidate_rank_indices')!=permutation):
            raise ValueError('independent full-payload length permutation differs')
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
            if (state['schedule_sha256']!=schedule.sha256 or state['next_update']!=updates
                    or state.get('candidate_profile')!=train['candidate_profile']
                    or state.get('batch_layout')!=layout or state.get('batch_order')!=train['batch_order']):
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
        if any([r['request_id'] for r in resident[rank]]!=[items[i]['request_id'] for i in permutation[rank]]
               for rank in range(4)):
            raise ValueError('rank probabilities must follow authenticated candidate permutation')
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
        from multimodal_jev.batched_stage2_audit import selections
        selection=selections(root,train,policies,files,through_update=terminal,
                             **(dict(history=resumed['selection_history']) if resumed else {}))
    result=dict(passed=True,candidate_profile=train['candidate_profile'],batch_layout=layout,
        batch_order=train['batch_order'],exact_length_permutation=True,updates=terminal,new_updates=terminal,ranks=4,checks=checks,files=files,
        population_check=population_check,exact_master_export=True,exact_peft_reload=True,
        exact_pre_update_recovery=(not resumed or recovery_policy['passed']),
        restored_policy_check=recovery_policy,
        scope=('Independent stopped Stage2 prefix, initial masters/freshAdam, every accepted update and due selection through the specified cursor; suffix excluded, not a completed training run or launch authorization.' if prefix else
               'Independent full Stage2 initial masters/freshAdam, 634 scheduled updates, all checkpoint exports and actor/vLLM parity, 11 full-selection rescoring points; execution still requires actual evidence.' if stage2 else
               'Independent continuous fixture shard/export, full scheduled sampling and actor/vLLM parity; task efficacy remains separate.'))
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
