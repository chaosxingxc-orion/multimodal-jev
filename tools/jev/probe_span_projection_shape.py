#!/usr/bin/env python3
"""Offline K-projection diagnosis on a saved real input; no rollout substitution."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    import torch_npu
    from safetensors import safe_open
    if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci4']:
        raise ValueError('physical NPU4 isolation required')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest: raise ValueError('pinned input changed: '+path)
    torch.npu.set_device(0); torch.npu.config.allow_internal_format=False
    if torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')!=b'disable':
        raise ValueError('internal format disable failed')
    actor=torch.load(config['hf_trace'],weights_only=True,map_location='cpu')
    ranks=[torch.load(p,weights_only=True,map_location='cpu') for p in config['tp_traces']]
    x_cpu=actor['layer.0.input_norm']; target=actor['layer.0.k_proj']
    if any(not torch.equal(r['layer.0.input_norm'],x_cpu) for r in ranks):
        raise ValueError('projection inputs do not match')
    with safe_open(config['weights'],framework='pt',device='cpu') as saved:
        keys=[k for k in saved.keys() if k=='model.language_model.layers.0.self_attn.k_proj.weight']
        if len(keys)!=1: raise ValueError('exact first language K weight required')
        w_cpu=saved.get_tensor(keys[0])
    if x_cpu.dtype!=torch.bfloat16 or w_cpu.dtype!=torch.bfloat16:
        raise ValueError('BF16 real inputs and weights required')
    q_width=actor['layer.0.q_proj'].shape[-1]//4
    kv_width=target.shape[-1]//4
    recorded=torch.cat([r['layer.0.qkv_proj'][:,q_width:q_width+kv_width] for r in ranks],-1)
    x=x_cpu.to('npu:0'); w=w_cpu.to('npu:0')
    tensors={'hf_recorded':target,'tp4_recorded':recorded}
    with torch.inference_mode():
        for repeat in range(3):
            tensors[f'full_{repeat}']=torch.nn.functional.linear(x,w).cpu()
            tensors[f'sliced_{repeat}']=torch.cat([torch.nn.functional.linear(x,s.contiguous().clone())
                for s in w.chunk(4,0)],-1).cpu()
        # Same output shape as HF; zero rows affect only unused output channels.
        # This is an offline discriminator, not a proposed production kernel.
        padded=[]
        for rank,s in enumerate(w.chunk(4,0)):
            full_size=torch.zeros_like(w); start=rank*kv_width
            full_size[start:start+kv_width].copy_(s)
            padded.append(torch.nn.functional.linear(x,full_size)[:,start:start+kv_width])
        tensors['padded_full_width']=torch.cat(padded,-1).cpu()
    def delta(a,b):
        diff=(a.float()-b.float()).abs()
        return dict(exact=bool(torch.equal(a,b)),maximum=float(diff.max()),
                    unequal_elements=int((a!=b).sum()),elements=a.numel())
    report={name:delta(t,target) for name,t in tensors.items()}
    pairs={name:delta(tensors[a],tensors[b]) for name,a,b in (
        ('full_repeat','full_0','full_2'),('slice_repeat','sliced_0','sliced_2'),
        ('sliced_matches_tp4','sliced_0','tp4_recorded'),
        ('padding_matches_full','padded_full_width','full_0'))}
    path=output/'tensors.pt'
    with path.open('xb') as handle: torch.save(tensors,handle)
    return dict(input_shape=list(x_cpu.shape),weight_shape=list(w_cpu.shape),weight_key=keys[0],
                dtype='bfloat16',against_hf=report,pairs=pairs,tensors_sha256=file_hash(path),
                scope='Offline first-layer K projection only; same real input, unadapted K weight; no full-model fix or RL admission')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    try:
        atomic_json(args.output/'result.json',run(json.loads(args.config.read_text()),args.output))
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time())); raise
