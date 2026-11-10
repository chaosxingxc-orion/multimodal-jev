#!/usr/bin/env python3
"""Isolate batch-shape effects using original saved layer-zero activations.

Loads four published tensors, not a model. Padding values are synthetic zeros;
this is an operator experiment, not a replay/localization of the failed actor.
"""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(args):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(16)
    start=time.monotonic()
    audit=json.loads(args.audit.read_text())
    if file_hash(args.audit)!='3c08bfd664f6b445054778c281e219eeb362f7b051059f59cc988453c5530930':
        raise ValueError('original model audit required')
    shard=args.model/'model-00001-of-00015.safetensors'
    pin=next(r['sha256'] for r in audit['files'] if r['file']==shard.name)
    if file_hash(shard)!=pin:
        raise ValueError('original projection weight shard required')
    keys=['thinker.model.layers.0.input_layernorm.weight']+[
        f'thinker.model.layers.0.self_attn.{name}_proj.weight' for name in ('q','k','v')]
    with safe_open(shard,framework='pt',device='cpu') as handle:
        weights={k:handle.get_tensor(k) for k in keys}
    order=[2,0,5,1,3,4];inputs=[];pins=[]
    for row in order:
        report=json.loads((args.trace/f'row-{row:02d}.json').read_text())
        path=args.trace/f'row-{row:02d}-trace/layer-00.input.pt'
        pin=report['reference']['layer-00.input']['sha256']
        if file_hash(path)!=pin:raise ValueError('original saved activation changed')
        value=torch.load(path,map_location='cpu',weights_only=True)
        if value.ndim!=2 or value.dtype!=torch.bfloat16 or value.shape[1]!=2048:
            raise ValueError('original BF16 layer-zero valid tokens required')
        inputs.append(value);pins.append(dict(row=row,sha256=pin,shape=list(value.shape)))
    def norm(x):
        v=x.float();v=v*torch.rsqrt(v.pow(2).mean(-1,keepdim=True)+1e-6)
        return weights[keys[0]]*v.to(x.dtype)
    def delta(x,y):
        d=(x.float()-y.float()).abs()
        return dict(equal=bool(torch.equal(x,y)),max_abs_error=float(d.max()),
                    differing_elements=int((x!=y).sum()))
    reports=[]
    with torch.inference_mode():
        for offset in (0,2,4):
            pair=inputs[offset:offset+2]
            padded=[torch.nn.functional.pad(v,(0,0,1230-len(v),0)).unsqueeze(0) for v in pair]
            batch=torch.cat(padded)
            single_norm=[norm(v) for v in padded];batch_norm=norm(batch)
            for j,v in enumerate(pair):
                reports.append(dict(pair=offset//2,row=order[offset+j],operation='rmsnorm',
                    **delta(single_norm[j][0,-len(v):],batch_norm[j,-len(v):])))
            for name in ('q','k','v'):
                weight=weights[f'thinker.model.layers.0.self_attn.{name}_proj.weight']
                single=[torch.nn.functional.linear(v,weight) for v in single_norm]
                combined=torch.nn.functional.linear(batch_norm,weight)
                for j,v in enumerate(pair):
                    reports.append(dict(pair=offset//2,row=order[offset+j],operation=name+'_proj',
                        **delta(single[j][0,-len(v):],combined[j,-len(v):])))
    atomic_json(args.output/'result.json',dict(status='diagnostic_complete',torch_version=torch.__version__,
        threads=torch.get_num_threads(),order=order,input_pins=pins,weight_shard_sha256=file_hash(shard),
        reports=reports,elapsed_seconds=time.monotonic()-start,weight_updates=0,
        scope='Original saved layer-zero valid inputs and published projection weights; compare batch1 and batch2 with zero left padding. Isolated CPU operators, no audio encoder, full attention, full model, actual actor tracing, generation or training.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('model','audit','trace','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    try:run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0))
