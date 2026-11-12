#!/usr/bin/env python3
"""Verify native actor against matched explicit-loss FSDP and HF/vLLM.

Engineering equivalence of optimizer implementations, not RL/SFT efficacy.
"""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_native_actor import ACTOR_SHA256
from verify_span_fsdp_audio import verify


def verify_native(root, suite, reference):
    root,suite,reference=map(Path,(root,suite,reference))
    result=verify(root,suite)
    read=lambda p:json.loads(p.read_text())
    actual_config=read(root/'config.json');expected_config=read(reference/'config.json')
    for key in ('model_path','initial_adapter','runtime_acceptance','release','manifest_sha256',
                'question_order','seed','batch_index','lr','lora_rank','lora_alpha',
                'audio_preprocessing','runtime_profile','samples_per_request','reward_mode'):
        if actual_config.get(key)!=expected_config.get(key):
            raise ValueError('explicit/native experiment configuration differs: '+key)
    pins=actual_config['pins'];errors=[];compared=0
    for rank in range(4):
        actual_dir=root/'run'/f'rank-{rank}';expected_dir=reference/'run'/f'rank-{rank}'
        for name in ('result.json','selection.json','optimizer-local.pt','terminal.json'):
            path=expected_dir/name
            if pins.get(str(path))!=file_hash(path):
                raise ValueError('matched reference evidence is not pinned: '+str(path))
        receipt=read(actual_dir/'native-actor.json')
        if (receipt['actor_sha256']!=ACTOR_SHA256
                or receipt['update_loop']!='DataParallelPPOActor.update_policy'
                or receipt['execution_dtype']!='bfloat16'
                or receipt['old_probabilities_unchanged'] is not True):
            raise ValueError('native actor provenance contract differs')
        partition=receipt['partition']
        if partition!=dict(world_size=4,global_requests=8,global_actions=256,local_requests=2,passed=True):
            raise ValueError('native actor complete global groups differ')
        if read(actual_dir/'selection.json')!=read(expected_dir/'selection.json'):
            raise ValueError('matched input schedule differs')
        if read(expected_dir/'terminal.json')['exit_code']!=0:
            raise ValueError('matched reference did not finish successfully')
        a=torch.load(actual_dir/'optimizer-local.pt',map_location='cpu',weights_only=True)
        b=torch.load(expected_dir/'optimizer-local.pt',map_location='cpu',weights_only=True)
        for key in ('rank','world_size','updates','names'):
            if a[key]!=b[key]:raise ValueError('matched optimizer layout differs')
        if a['optimizer']['param_groups']!=b['optimizer']['param_groups']:
            raise ValueError('matched optimizer hyperparameters differ')
        pairs=[(a['parameters'][k],v) for k,v in b['parameters'].items()]
        for key,values in b['optimizer']['state'].items():
            for field,value in values.items():
                pairs.append((a['optimizer']['state'][key][field],value))
        for observed,expected in pairs:
            torch.testing.assert_close(observed,expected,atol=2e-7,rtol=2e-5)
            if observed.numel():errors.append(float((observed-expected).abs().max()))
            compared+=1
        result['files'][str(actual_dir/'native-actor.json')]=file_hash(actual_dir/'native-actor.json')
    left=load_file(str(root/'run/adapter/adapter_model.safetensors'))
    right_path=reference/'run/adapter/adapter_model.safetensors'
    if pins.get(str(right_path))!=file_hash(right_path):raise ValueError('matched adapter not pinned')
    right=load_file(str(right_path))
    if left.keys()!=right.keys() or any(not torch.equal(v,right[k]) for k,v in left.items()):
        raise ValueError('native and matched explicit-loss exported BF16 adapters differ')
    result.update(native_actor_verified=True,actor_sha256=ACTOR_SHA256,
        matched_explicit_loss_reference=str(reference),compared_master_and_adam_tensors=compared,
        maximum_master_and_adam_error=max(errors),matched_bf16_export_exact=True,
        scope='One native verl actor/FSDP BF16 audio update, matched explicit-loss optimizer equivalence and independent HF/TP1 acceptance. Not continuous Ray trainer or task improvement.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--suite',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();atomic_json(a.output,verify_native(a.root,a.suite,a.reference))
