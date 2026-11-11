#!/usr/bin/env python3
"""Independent persisted-artifact checks for the four-rank synthetic probe."""
import argparse
import json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json, file_hash


def verify(root):
    root=Path(root)
    read=lambda p:json.loads(p.read_text())
    launch=read(root/'launch.json')
    if launch['physical_devices']!=[4,5,6,7] or launch['launch_exit_code']!=0:
        raise ValueError('authorized successful launch required')
    if file_hash(root/'source-manifest.json')!=launch['source_manifest_sha256']:
        raise ValueError('source manifest changed')
    manifest=read(root/'source-manifest.json')
    for name,digest in manifest.items():
        if file_hash(root/'source'/name)!=digest:raise ValueError('source snapshot changed')
    rows=[];file_pins={}
    for rank in range(4):
        out=root/'run'/f'rank-{rank}'
        r=read(out/'result.json')
        if (read(out/'terminal.json')['exit_code']!=0 or r['status']!='passed'
                or r['rank']!=rank or r['world_size']!=4 or r['backend']!='hccl'
                or r['updates']!=3 or r['restored_at_update']!=2
                or r['execution_dtypes']!=[['torch.bfloat16','torch.bfloat16']]):
            raise ValueError('rank runtime contract failed')
        for key in ('exact_resume_output','exact_resume_masters','exact_resume_adam',
                    'frozen_base_unchanged','cross_rank_export_exact'):
            if r[key] is not True:raise ValueError('rank check failed: '+key)
        if r['global_changed_shards']!=8:raise ValueError('both adapters must change on all ranks')
        checkpoint=out/'local-checkpoint.pt'
        if file_hash(checkpoint)!=r['checkpoint_sha256']:raise ValueError('checkpoint changed')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['rank']!=rank or state['world_size']!=4 or state['updates']!=2:
            raise ValueError('checkpoint topology/update mismatch')
        if set(state['parameters'])!=set(r['local_parameter_counts']):
            raise ValueError('shard parameter set differs')
        for name,p in state['parameters'].items():
            is_lora='lora_A.' in name or 'lora_B.' in name
            if p.numel()!=r['local_parameter_counts'][name] or p.dtype!=(torch.float32 if is_lora else torch.bfloat16):
                raise ValueError('local master/base dtype or sharding differs')
        optimizer=state['optimizer']['state']
        if len(optimizer)!=2:raise ValueError('both LoRA optimizer states required')
        for item in optimizer.values():
            if (item['step'].item()!=2 or item['exp_avg'].dtype!=torch.float32
                    or item['exp_avg_sq'].dtype!=torch.float32
                    or any(not torch.isfinite(v).all() for v in item.values())):
                raise ValueError('FP32 finite Adam state at step two required')
        rows.append(r)
        file_pins.update({str(p):file_hash(p) for p in (out/'result.json',out/'terminal.json',checkpoint)})
    if any(r['adapter_export_hashes']!=rows[0]['adapter_export_hashes'] for r in rows):
        raise ValueError('export identities differ between ranks')
    export_path=root/'run/rank-0/bf16-adapter.pt'
    export=torch.load(export_path,map_location='cpu',weights_only=True)
    import hashlib
    if len(export)!=2 or any(p.dtype!=torch.bfloat16 for p in export.values()):
        raise ValueError('two BF16 exported LoRA tensors required')
    hashes={n:hashlib.sha256(p.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for n,p in export.items()}
    if hashes!=rows[0]['adapter_export_hashes']:raise ValueError('actual export differs from recorded identities')
    file_pins[str(export_path)]=file_hash(export_path)
    return dict(root=str(root),passed=True,image=launch['image'],physical_devices=launch['physical_devices'],
        source_manifest_sha256=launch['source_manifest_sha256'],ranks=rows,files=file_pins,
        scope='Persisted source, four-rank results, real local optimizer shards and BF16 tensor export audited. '
        'Exact resumed-versus-uninterrupted checks executed within the reviewed NPU probe; '
        'not a speech/Omni/vLLM/distributed-verl-trainer acceptance.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=verify(args.root);atomic_json(args.output,result)
    print(json.dumps(dict(passed=result['passed'],ranks=len(result['ranks']))))
