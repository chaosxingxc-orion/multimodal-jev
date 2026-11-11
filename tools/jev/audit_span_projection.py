#!/usr/bin/env python3
"""NPU replay of captured first-layer inputs, for numerical diagnosis only.

Captured activations never feed production evaluation or training. Compare
identical inputs/weights across projection shapes and NPU storage settings.
"""
import argparse
import json
from pathlib import Path
import time
import torch
import torch_npu
from safetensors import safe_open
from multimodal_jev.baseline import atomic_json, file_hash


def difference(a, b, mask=None):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    if mask is not None:
        a, b = a[mask], b[mask]
    delta = (a-b).abs()
    return dict(maximum=float(delta.max()), rms=float(delta.square().mean().sqrt()),
                unequal=int(delta.count_nonzero()), elements=delta.numel())


def run(config, output):
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4']:
        raise ValueError('physical NPU4 isolation required')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:
            raise ValueError('pinned source/input changed: '+path)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    actor=torch.load(config['hf_trace'],map_location='cpu',weights_only=True)
    behavior=torch.load(config['vllm_trace'],map_location='cpu',weights_only=True)
    x=actor['layer.0.input_norm'].to('npu:0')
    same=(actor['layer.0.input_norm']==behavior['layer.0.input_norm']).all(-1)
    if not same.any():
        raise ValueError('no identical input rows')
    prefix='model.language_model.layers.0.self_attn.'
    weights=[];adapters={}
    with safe_open(config['model_weights'],framework='pt',device='cpu') as f:
        for key in ('q_proj','k_proj','v_proj'):
            weights.append(f.get_tensor(prefix+key+'.weight').to('npu:0'))
    with safe_open(config['adapter_weights'],framework='pt',device='cpu') as f:
        for key in ('q_proj','v_proj'):
            adapters[key]=tuple(f.get_tensor('base_model.model.'+prefix+key+'.lora_'+ab+'.weight').to('npu:0')
                                for ab in ('A','B'))
    widths=[w.shape[0] for w in weights]
    behavior_parts=behavior['layer.0.qkv_proj'].split(widths,-1)
    outputs={};comparisons={};conv_comparisons={}
    original=torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
    try:
        with torch.no_grad():
            for internal in (False,True):
                torch.npu.config.allow_internal_format=internal
                for layout in ('nd','nz'):
                    fmt=2 if layout=='nd' else 29
                    w=[torch_npu.npu_format_cast(v,fmt) for v in weights]
                    packed=torch_npu.npu_format_cast(torch.cat(weights),fmt)
                    for shape in ('separate_2d','separate_3d','packed_2d'):
                        key=f'internal-{internal}/{layout}/{shape}'
                        def compute():
                            if shape=='packed_2d':
                                values=torch.nn.functional.linear(x,packed).split(widths,-1)
                            else:
                                xx=x.unsqueeze(0) if shape=='separate_3d' else x
                                values=[torch.nn.functional.linear(xx,v).reshape(len(x),-1) for v in w]
                            result=[]
                            for name,value in zip(('q_proj','k_proj','v_proj'),values):
                                if name in adapters:
                                    a,b=adapters[name]
                                    value=value+torch.nn.functional.linear(torch.nn.functional.linear(x,a),b)*config['lora_scale']
                                result.append(value)
                            return tuple(result)
                        first=compute();second=compute()
                        outputs[key]=tuple(v.cpu() for v in first)
                        comparisons[key]={name:dict(
                            versus_hf=difference(value,actor['layer.0.'+name]),
                            versus_vllm_identical_inputs=difference(value,target,same),
                            repeat=difference(value,repeated))
                            for name,value,target,repeated in zip(('q_proj','k_proj','v_proj'),first,behavior_parts,second)}
            baseline=outputs['internal-False/nd/separate_3d']
            for key,values in outputs.items():
                comparisons[key]['versus_replayed_hf_shapes']={name:difference(value,ref)
                    for name,value,ref in zip(('q_proj','k_proj','v_proj'),values,baseline)}
            # Both traces already contain authenticated convolution inputs,
            # weights and bias. Replay only; do not substitute tower outputs.
            for name in ('input','weight','bias'):
                if not torch.equal(actor['audio.conv2d1.'+name],behavior['audio.conv2d1.'+name]):
                    raise ValueError('conv identity differs: '+name)
            cx=actor['audio.conv2d1.input'].to('npu:0')
            cw=actor['audio.conv2d1.weight'].to('npu:0')
            cb=actor['audio.conv2d1.bias'].to('npu:0')
            for internal in (False,True):
                torch.npu.config.allow_internal_format=internal
                for layout in ('hf','vllm'):
                    strides=tuple(actor['audio.conv2d1.input_stride'].tolist() if layout=='hf'
                                  else behavior['audio.conv2d1.input_stride'].tolist())
                    value=torch.empty_strided(cx.shape,strides,device=cx.device,dtype=cx.dtype)
                    value.copy_(cx)
                    for bias_mode in ('fused','separate'):
                        def conv():
                            out=torch.nn.functional.conv2d(value,cw,cb if bias_mode=='fused' else None,stride=2,padding=1)
                            return out if bias_mode=='fused' else out+cb[None,:,None,None]
                        first=conv();second=conv()
                        conv_comparisons[f'internal-{internal}/{layout}/{bias_mode}']=dict(
                            versus_hf=difference(first,actor['audio.conv2d1']),
                            versus_vllm=difference(first,behavior['audio.conv2d1']),repeat=difference(first,second))
    finally:
        if original in (b'enable',b'disable'):
            torch.npu.config.allow_internal_format=original==b'enable'
    result=dict(status='completed',identical_input_rows=int(same.sum()),total_rows=len(same),
                original_allow_internal_format=None if original is None else original.decode(),comparisons=comparisons,
                conv_comparisons=conv_comparisons,
                scope='Captured activation projection replay only; not model parity or RL admission.')
    atomic_json(output/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        run(json.loads(args.config.read_text()),args.output)
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
