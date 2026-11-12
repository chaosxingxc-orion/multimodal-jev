#!/usr/bin/env python3
"""Independent prefix-v1 state audit and lossless PEFT naming repair, CPU only.

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
    from multimodal_jev.span_adapter_export import adapter_key, legacy_broken_key
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.slurp_official_training import load, example
    from multimodal_jev.slurp_training import content_hash
    from multimodal_jev.audio_policy import load_audio
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only checkpoint audit required')
    torch.set_num_threads(4)
    for p, h in config['pins'].items():
        if file_hash(p) != h:
            raise ValueError('audit dependency changed: '+p)
    root = Path(config['training_root'])
    old = json.loads((root/'config.json').read_text())
    launch = json.loads((root/'launch.json').read_text())
    observed = json.loads(Path(config['training_observed']).read_text())
    if (launch['config_sha256'] != file_hash(root/'config.json')
            or observed['container_id'] != launch['container_id']
            or observed['image'] != launch['image']
            or observed['state']['Status'] != 'exited' or observed['state']['ExitCode'] != 0):
        raise ValueError('exact successful training observation required')
    for p, h in old['pins'].items():
        if file_hash(p) != h:
            raise ValueError('training dependency changed: '+p)
    if old['updates'] != 2 or old['global_batch'] != 128:
        raise ValueError('specific two-update prefix repair required')
    files = {str(p): file_hash(p) for p in (root/'config.json',root/'launch.json',Path(config['training_observed']))}
    rows, task, _ = load(old['release'],old['release_sha256'],'learning',purpose='optimize')
    population = {r['id']:r for r in rows}
    epoch = json.loads(Path(old['epoch_file']).read_text())
    if content_hash(epoch['batches']) != epoch['report']['batches_sha256']:
        raise ValueError('epoch changed')
    # Reconstruct supervision and input metadata without trusting rank reports.
    for step in range(2):
        p = root/'run'/f'batch-{step:04d}.json';batch = json.loads(p.read_text()); files[str(p)] = file_hash(p)
        slots = epoch['batches'][step]
        if len(batch['records']) != 128 or len(slots) != 128:
            raise ValueError('complete scheduled batch required')
        for index,(slot,record) in enumerate(zip(slots,batch['records'],strict=True)):
            row = population[slot['id']]
            request, target, _ = example(row,task,slot['view'],slot['variant'])
            if file_hash(row['audio']) != row['audio_sha256']:
                raise ValueError('training audio changed')
            _, receipt = load_audio(row['audio'], return_receipt=True)
            expected = dict(id=row['id'],request_id=f'{row["id"]}::epoch0-batch{step}-slot{index}',
                view=slot['view'],variant=slot['variant'],weight=slot['weight'],
                input_sha256=content_hash(request.model_input()),audio_receipt=receipt,
                targets=target['targets'],span_weights=target['span_weights'])
            if any(record.get(k) != v for k,v in expected.items()):
                raise ValueError('batch supervision or inputs differ')
        original = root/'run'/f'adapter-step{step:04d}'
        if batch['policy_id'] != old['base_id']+':'+checkpoint_identity(original)[0]:
            raise ValueError('batch behavior identity differs')
    updates = 2
    original = root/'run'/f'adapter-step{updates:04d}'
    original_id, original_files = checkpoint_identity(original)
    files.update({str(original/k):h for k,h in original_files.items()})
    exported = load_file(str(original/'adapter_model.safetensors'))
    states = []
    for rank in range(4):
        p = root/'run'/f'rank-{rank}'/f'checkpoint-{updates:04d}.pt'
        files[str(p)] = file_hash(p)
        state = torch.load(p,map_location='cpu',weights_only=True)
        if any(state.get(k) != v for k,v in dict(world_size=4,rank=rank,updates=2,epoch=0,next_batch=2,
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
            if set(adam) != {'step','exp_avg','exp_avg_sq'} or adam['step'].dtype!=torch.float32 or adam['step'].numel()!=1 or adam['step'].item()!=2:
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
        key=adapter_key(name,native=True); broken=legacy_broken_key(key)
        if broken not in exported or key in repaired:
            raise ValueError('expected one-to-one known export defect')
        master=torch.cat([s['parameters'][name] for s in states])
        old_tensor=exported[broken]
        if master.numel()!=old_tensor.numel() or old_tensor.dtype!=torch.bfloat16:
            raise ValueError('single-Linear shard reconstruction size/dtype differs')
        value=master.reshape(old_tensor.shape).to(torch.bfloat16)
        if not torch.equal(value,old_tensor):
            raise ValueError('reconstructed master export differs')
        sub_bf16 += int((master!=master.bfloat16().float()).sum())
        repaired[key]=value.contiguous()
    if len(exported)!=len(repaired):
        raise ValueError('unexpected export tensor')
    adapter=output/'adapter-step0002';adapter.mkdir()
    save_file(repaired,str(adapter/'adapter_model.safetensors'))
    (adapter/'adapter_model.safetensors').chmod(0o644)
    shutil.copyfile(original/'adapter_config.json',adapter/'adapter_config.json')
    adapter_id, adapter_files=checkpoint_identity(adapter)
    # Actual PEFT reload: name validity alone does not establish effective loading.
    _, base=load_runtime(old['model_path'],'cpu','bfloat16')
    loaded=PeftModel.from_pretrained(base,adapter,is_trainable=False,autocast_adapter_dtype=False).eval()
    actual={adapter_key(n):p.detach().cpu() for n,p in loaded.named_parameters() if 'lora_A.' in n or 'lora_B.' in n}
    if actual.keys()!=repaired.keys() or any(actual[k].dtype!=torch.bfloat16 or not torch.equal(actual[k],v) for k,v in repaired.items()):
        raise ValueError('actual PEFT reload lost or changed LoRA tensors')
    files.update({str(adapter/k):h for k,h in adapter_files.items()})
    return dict(passed=True,training_root=str(root),updates=2,exact_master_export=True,
        exact_peft_reload=True,adapter=str(adapter),adapter_id=adapter_id,original_adapter_id=original_id,
        tensors=112,ranks=4,sub_bf16_master_elements=sub_bf16,supervision_records_verified=256,files=files,
        scope='Exact FP32 master reconstruction, Adam structure/cursor, BF16 export naming repair and actual CPU PEFT tensor reload. No resident output parity, cross-backend gate, or efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'state-verified.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
