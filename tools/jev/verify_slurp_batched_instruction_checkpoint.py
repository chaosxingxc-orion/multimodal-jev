#!/usr/bin/env python3
"""Independent terminal instruction state audit and optional legacy naming repair.

Keep original checkpoints and exports immutable. Reconstruct every FP32 master
from the four single-Linear FSDP shards, verify BF16 values against the original,
then load the repaired export through actual PEFT and compare all LoRA tensors.
This is not resident-FSDP probability parity or a training efficacy measurement.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    from safetensors.torch import load_file, save_file
    from peft import PeftModel
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.span_adapter_export import adapter_key
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.slurp_official_training import load, example
    from multimodal_jev.slurp_training import content_hash
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.slurp_instruction_resume import load_full_schedule
    from multimodal_jev.span_batched_instruction_resume import authenticate
    from multimodal_jev.span_batched_instruction_audit import completed_range, input_policy_id, verify_rank_report
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only checkpoint audit required')
    torch.set_num_threads(4)
    for p, h in config['pins'].items():
        if file_hash(p) != h:
            raise ValueError('audit dependency changed: '+p)
    root = Path(config['training_root'])
    old = json.loads((root/'config.json').read_text())
    launch_path=Path(config['training_launch'])
    if config['pins'].get(str(launch_path)) != file_hash(launch_path):
        raise ValueError('pinned exact training launch required')
    launch = json.loads(launch_path.read_text())
    observed = json.loads(Path(config['training_observed']).read_text())
    if (launch['config_sha256'] != file_hash(root/'config.json')
            or observed['container_id'] != launch['container_id']
            or observed['image'] != launch['image']
            or observed['state']['Status'] != 'exited' or observed['state']['ExitCode'] != 0):
        raise ValueError('exact successful training observation required')
    for p, h in old['pins'].items():
        if file_hash(p) != h:
            raise ValueError('training dependency changed: '+p)
    if old.get('start_update',0):
        authenticate(old)
    if config.get('legacy_export_repair',True) or not config.get('require_durable_commit',False):
        raise ValueError('candidate requires unchanged export and durable commit')
    batches,epoch_updates,_=load_full_schedule(old)
    start_update, updates, (next_epoch, next_batch) = completed_range(old, epoch_updates)
    files = {str(p): file_hash(p) for p in (root/'config.json',launch_path,Path(config['training_observed']))}
    if config.get('require_durable_commit', False):
        commit_path = root/'run'/f'checkpoint-{updates:04d}-committed.json'
        commit = json.loads(commit_path.read_text())
        expected = {f'rank-{rank}/checkpoint-{updates:04d}.pt' for rank in range(4)}
        if (commit['updates'] != updates or commit['world_size'] != 4 or set(commit['files']) != expected
                or commit['adapter_id'] != checkpoint_identity(root/'run'/f'adapter-step{updates:04d}')[0]):
            raise ValueError('complete durable checkpoint commit required')
        for relative, digest in commit['files'].items():
            if file_hash(root/'run'/relative) != digest:
                raise ValueError('committed checkpoint bytes changed')
        files[str(commit_path)] = file_hash(commit_path)
    rows, task, _ = load(old['release'],old['release_sha256'],'learning',purpose='optimize')
    population = {r['id']:r for r in rows}
    identities = {}
    def identity(path):
        path = Path(path)
        if path not in identities:
            value, assets = checkpoint_identity(path)
            identities[path] = value
            files.update({str(path/k): h for k,h in assets.items()})
        return identities[path]
    processor, base=load_runtime(old['model_path'],'cpu','bfloat16')
    from multimodal_jev.span_contract import encode_span_request
    from multimodal_jev.span_length_groups import length_group_indices
    audio_receipts = {}
    input_lengths = {}
    memoize = config.get('independent_input_memoization', False)
    if type(memoize) is not bool:
        raise ValueError('explicit Boolean audit memoization mode required')
    input_processor_calls = 0
    input_memo_hits = 0
    input_audit_start = time.monotonic()
    real_records = 0
    # Reconstruct supervision and input metadata without trusting rank reports.
    for step in range(start_update, updates):
        p = root/'run'/f'batch-{step:04d}.json';batch = json.loads(p.read_text()); files[str(p)] = file_hash(p)
        epoch_index, batch_index, slots = batches[step]
        if len(batch['records']) != old['global_batch'] or len(slots) != old['global_batch']:
            raise ValueError('complete scheduled batch required')
        lengths=[]
        for index,(slot,record) in enumerate(zip(slots,batch['records'],strict=True)):
            row = population[slot['id']]
            request, target, _ = example(row,task,slot['view'],slot['variant'])
            audio = None
            if row['id'] not in audio_receipts:
                if file_hash(row['audio']) != row['audio_sha256']:
                    raise ValueError('training audio changed')
                first_audio, receipt = load_audio(row['audio'], return_receipt=True)
                audio_receipts[row['id']] = receipt
                if memoize:
                    audio = first_audio
            receipt = audio_receipts[row['id']]
            expected = dict(id=row['id'],request_id=f'{row["id"]}::epoch{epoch_index}-batch{batch_index}-slot{index}',
                view=slot['view'],variant=slot['variant'],weight=slot['weight'],
                input_sha256=content_hash(request.model_input()),audio_receipt=receipt,
                targets=target['targets'],span_weights=target['span_weights'])
            if any(record.get(k) != v for k,v in expected.items()):
                raise ValueError('batch supervision or inputs differ')
            if (type(record.get('input_tokens')) is not int
                    or not 0 < record['input_tokens'] <= old['context_limit']):
                raise ValueError('complete untruncated input length required')
            # Independent reconstruction, never a lookup in the trainer cache.
            # This audit process has one pinned processor/preprocessing version.
            # Reuse only deterministic lengths for identical audio + full input;
            # supervision, policy identity, weights and ordering stay per-slot.
            key = (str(row['audio']), row['audio_sha256'], expected['input_sha256'])
            if memoize and key in input_lengths:
                actual_length, text_length, feature_length = input_lengths[key]
                input_memo_hits += 1
            else:
                if audio is None:
                    audio = load_audio(row['audio'])
                inputs,positions=encode_span_request(processor,request,audio)
                actual_length = int(inputs['input_ids'].shape[-1])
                text_length = int(inputs['attention_mask'].sum())
                feature_length = int(inputs['input_features_mask'].sum())
                input_processor_calls += 1
                if memoize:
                    input_lengths[key] = (actual_length, text_length, feature_length)
            if actual_length!=record['input_tokens']:
                raise ValueError('actual processor token length differs')
            lengths.append((text_length, feature_length))
            real_records += not slot['padding']
        expected_ranks=length_group_indices(lengths,microbatch=old['microbatch_per_rank'],world_size=4,order=old['batch_order'])
        if batch.get('rank_indices')!=expected_ranks:
            raise ValueError('actual lengths do not reproduce complete payload permutation')
        if batch['policy_id'] != input_policy_id(old, root, step, epoch_updates, identity):
            raise ValueError('batch behavior identity differs')
        if memoize and ((step-start_update+1)%8 == 0 or step+1 == updates):
            atomic_json(output/'input-audit-progress.json',dict(
                verified_slots=(step-start_update+1)*old['global_batch'],
                processor_calls=input_processor_calls,memo_hits=input_memo_hits,
                seconds=time.monotonic()-input_audit_start))
    input_audit_seconds = time.monotonic()-input_audit_start
    original = root/'run'/f'adapter-step{updates:04d}'
    original_id, original_files = checkpoint_identity(original)
    files.update({str(original/k):h for k,h in original_files.items()})
    exported = load_file(str(original/'adapter_model.safetensors'))
    states = []; restored_reports=[]
    for rank in range(4):
        rank_root = root/'run'/f'rank-{rank}'
        report_path = rank_root/'result.json'; terminal_path = rank_root/'terminal.json'
        report = json.loads(report_path.read_text())
        if start_update:
            from multimodal_jev.span_batched_instruction_audit import verify_restored_report
            gate=json.loads(Path(old['candidate_parity_acceptance']).read_text())
            reference=json.loads(Path(gate['hf_results']).read_text())['records']
            restored_path=rank_root/'restored-policy.json'
            restored_reports.append(verify_restored_report(json.loads(restored_path.read_text()),reference,
                rank=rank,microbatch=old['microbatch_per_rank']))
            files[str(restored_path)]=file_hash(restored_path)
            restore_path=rank_root/'restored.json';restored=json.loads(restore_path.read_text())
            if (restored.get('exact_masters') is not True or restored.get('exact_adam') is not True
                    or restored.get('rank')!=rank or restored.get('previous_updates')!=start_update):
                raise ValueError('exact restored master/Adam receipt required')
            files[str(restore_path)]=file_hash(restore_path)
        verify_rank_report(report, json.loads(terminal_path.read_text()), old,
                           rank=rank, epoch_updates=epoch_updates, adapter_id=original_id)
        if report['initial_adapter_id'] != identity(Path(old['initial_adapter']) if start_update else root/'run/adapter-step0000'):
            raise ValueError('rank initial adapter differs')
        files.update({str(p): file_hash(p) for p in (report_path, terminal_path)})
        p = root/'run'/f'rank-{rank}'/f'checkpoint-{updates:04d}.pt'
        files[str(p)] = file_hash(p)
        state = torch.load(p,map_location='cpu',weights_only=True)
        if any(state.get(k) != v for k,v in dict(world_size=4,rank=rank,updates=updates,epoch=next_epoch,next_batch=next_batch,
            schedule_sha256=old['schedule_sha256'],release_sha256=old['release_sha256'],adapter_id=original_id).items()):
            raise ValueError('rank state lineage differs')
        if len(state['names']) != 112 or len(set(state['names'])) != 112 or set(state['parameters']) != set(state['names']):
            raise ValueError('complete 112 LoRA tensors required')
        optimizer = state['optimizer']; groups = optimizer['param_groups']
        if len(groups) != 1 or groups[0]['params'] != list(range(112)) or set(optimizer['state']) != set(range(112)):
            raise ValueError('complete ordered optimizer required')
        expected = dict(lr=old['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False,
            maximize=False,foreach=False,capturable=False,differentiable=False,fused=False,decoupled_weight_decay=True)
        if {k:v for k,v in groups[0].items() if k!='params'} != expected:
            raise ValueError('optimizer hyperparameters differ')
        for i,n in enumerate(state['names']):
            master=state['parameters'][n]; adam=optimizer['state'][i]
            if master.ndim!=1 or master.dtype!=torch.float32 or not torch.isfinite(master).all():
                raise ValueError('finite flat FP32 master shard required')
            if set(adam) != {'step','exp_avg','exp_avg_sq'} or adam['step'].dtype!=torch.float32 or adam['step'].numel()!=1 or adam['step'].item()!=updates:
                raise ValueError('exact Adam step required')
            for key in ('exp_avg','exp_avg_sq'):
                if adam[key].shape!=master.shape or adam[key].dtype!=torch.float32 or not torch.isfinite(adam[key]).all():
                    raise ValueError('finite shape-matched FP32 moments required')
            if (adam['exp_avg_sq']<0).any():
                raise ValueError('negative second moment')
        states.append(state)
    if any(s['names'] != states[0]['names'] for s in states):
        raise ValueError('cross-rank name order differs')
    repaired = {}; sub_bf16 = 0
    for name in states[0]['names']:
        from multimodal_jev.span_adapter_export import checkpoint_export_key
        key=adapter_key(name,native=True)
        source_key=checkpoint_export_key(key,legacy_repair=config.get('legacy_export_repair', True))
        if source_key not in exported or key in repaired:
            raise ValueError('expected one-to-one declared export layout')
        master=torch.cat([s['parameters'][name] for s in states])
        old_tensor=exported[source_key]
        if master.numel()!=old_tensor.numel() or old_tensor.dtype!=torch.bfloat16:
            raise ValueError('single-Linear shard reconstruction size/dtype differs')
        value=master.reshape(old_tensor.shape).to(torch.bfloat16)
        if not torch.equal(value,old_tensor):
            raise ValueError('reconstructed master export differs')
        sub_bf16 += int((master!=master.bfloat16().float()).sum())
        repaired[key]=value.contiguous()
    if len(exported)!=len(repaired):
        raise ValueError('unexpected export tensor')
    adapter=output/f'adapter-step{updates:04d}';adapter.mkdir()
    save_file(repaired,str(adapter/'adapter_model.safetensors'))
    (adapter/'adapter_model.safetensors').chmod(0o644)
    shutil.copyfile(original/'adapter_config.json',adapter/'adapter_config.json')
    adapter_id, adapter_files=checkpoint_identity(adapter)
    # Actual PEFT reload: name validity alone does not establish effective loading.
    loaded=PeftModel.from_pretrained(base,adapter,is_trainable=False,autocast_adapter_dtype=False).eval()
    actual={adapter_key(n):p.detach().cpu() for n,p in loaded.named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
    if actual.keys()!=repaired.keys() or any(actual[k].dtype!=torch.bfloat16 or not torch.equal(actual[k],v) for k,v in repaired.items()):
        raise ValueError('actual PEFT reload lost or changed LoRA tensors')
    files.update({str(adapter/k):h for k,h in adapter_files.items()})
    return dict(passed=True,training_root=str(root),updates=updates,executed_updates=updates-start_update,
        start_update=start_update,epoch=next_epoch,next_batch=next_batch,exact_master_export=True,
        exact_peft_reload=True,adapter=str(adapter),adapter_id=adapter_id,original_adapter_id=original_id,
        tensors=112,ranks=4,sub_bf16_master_elements=sub_bf16,
        independent_input_memoization=memoize,input_processor_calls=input_processor_calls,
        input_memo_hits=input_memo_hits,input_audit_seconds=input_audit_seconds,
        supervision_records_verified=(updates-start_update)*old['global_batch'],real_supervision_records_verified=real_records,files=files,
        exact_length_permutation=True,candidate_profile=old['candidate_profile'],restored_policy_reports=restored_reports,
        legacy_export_repair=config.get('legacy_export_repair', True),
        durable_commit_verified=config.get('require_durable_commit', False),
        scope='Exact FP32 master reconstruction, Adam structure/cursor, declared BF16 export layout and actual CPU PEFT tensor reload. No resident output parity, cross-backend gate, or efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'state-verified.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
