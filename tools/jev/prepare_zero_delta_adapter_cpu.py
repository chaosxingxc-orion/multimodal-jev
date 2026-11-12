#!/usr/bin/env python3
"""Create a base-function inference control: retain A, zero every LoRA B.

No task training. Base equivalence is local linear-operator evidence until
real audio/backend comparison; this is not a new RL initialization decision.
"""
import argparse
import json
from pathlib import Path
import shutil
import torch
from safetensors.torch import load_file,save_file
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity


def build(source,destination,expected_id):
    source,destination=map(Path,(source,destination))
    if destination.exists():raise FileExistsError('immutable control adapter exists')
    original_id,files=checkpoint_identity(source)
    if original_id!=expected_id:raise ValueError('pinned original adapter required')
    cfg=json.loads((source/'adapter_config.json').read_text())
    if (cfg['peft_type']!='LORA' or cfg['bias']!='none' or cfg['lora_dropout']!=0
            or cfg.get('modules_to_save') or cfg.get('use_dora') or cfg.get('use_rslora')
            or cfg.get('lora_bias') or cfg.get('rank_pattern') or cfg.get('alpha_pattern')
            or cfg['r']!=4 or cfg['lora_alpha']!=8):
        raise ValueError('plain pinned q/v LoRA configuration required')
    tensors=load_file(str(source/'adapter_model.safetensors'));targets=cfg['target_modules']
    expected={f'base_model.model.{t}.lora_{side}.weight' for t in targets for side in ('A','B')}
    if set(tensors)!=expected or len(tensors)!=112 or len(set(targets))!=56:
        raise ValueError('complete exact112 q/v LoRA tensors required')
    if any(not t.startswith('model.language_model.layers.') or t.rsplit('.',1)[-1] not in ('q_proj','v_proj') for t in targets):
        raise ValueError('only language q/v targets permitted')
    if any(p.dtype!=torch.bfloat16 or not torch.isfinite(p).all() for p in tensors.values()):
        raise ValueError('finite BF16 source tensors required')
    zero={n:(torch.zeros_like(p) if '.lora_B.' in n else p.clone()) for n,p in tensors.items()}
    checks=[];torch.manual_seed(317)
    for target in sorted(targets):
        prefix='base_model.model.'+target
        a,b=(zero[prefix+'.lora_'+side+'.weight'] for side in ('A','B'))
        if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=4 or b.shape[1]!=4:raise ValueError('rank4 factors required')
        x=torch.randn(3,a.shape[1]).bfloat16()
        delta=torch.nn.functional.linear(torch.nn.functional.linear(x,a),b)*2
        base=torch.randn_like(delta)
        if torch.count_nonzero(delta) or not torch.equal(base+delta,base):raise AssertionError('zero delta operator differs')
        checks.append(dict(target=target,input_features=a.shape[1],output_features=b.shape[0],zero_delta=True))
    destination.mkdir(parents=True);shutil.copyfile(source/'adapter_config.json',destination/'adapter_config.json')
    save_file(zero,str(destination/'adapter_model.safetensors'))
    (destination/'adapter_model.safetensors').chmod(0o644)
    loaded=load_file(str(destination/'adapter_model.safetensors'))
    if loaded.keys()!=zero.keys() or any(not torch.equal(loaded[n],p) for n,p in zero.items()):raise AssertionError('control serialization differs')
    identity,output_files=checkpoint_identity(destination)
    return dict(passed=True,source=str(source),source_id=original_id,source_files=files,
                adapter=str(destination),adapter_id=identity,adapter_files=output_files,checks=checks,
                scope='All56 actual BF16 LoRA factor pairs give exact zero delta on fixed synthetic features; A retained/Bzero. Full real audio/model/vLLM equivalence not yet executed. Inference control only, not RL initialization or task improvement.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--source-id',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        r=build(a.source,a.output/'adapter',a.source_id);atomic_json(a.output/'result.json',r);atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
