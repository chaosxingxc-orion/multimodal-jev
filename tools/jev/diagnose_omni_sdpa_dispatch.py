#!/usr/bin/env python3
"""Same-shape SDPA controls on frozen captured Q/K/V; no model or task replay."""
import argparse
import contextlib
import json
from pathlib import Path
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.audio2tool_inference import write_once


def run(args):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import profile, ProfilerActivity
    torch.set_num_threads(16)
    start=time.monotonic()
    parent=json.loads((args.parent/'result.json').read_text())
    if file_hash(args.parent/'result.json')!='389652411c69533f07a151d7f89acde2c8a3d90e8900f4772327e93032ad329b':
        raise ValueError('accepted first-attention result required')
    tensors={}
    for stage in ('q_rope','k_rope','v_proj','attention_core'):
        item=next(s for s in parent['stages'] if s['stage']==stage)
        for condition in ('audio_padding','text_padding'):
            p=args.parent/(condition+'-'+stage+'.pt')
            if file_hash(p)!=item['tensors'][condition]:raise ValueError('captured tensor changed')
            x=torch.load(p,map_location='cpu',weights_only=True)
            if list(x.shape)!=item['shape'] or str(x.dtype)!=item['dtype'] or not torch.isfinite(x).all():
                raise ValueError('invalid captured tensor')
            tensors[condition,stage]=x
        if stage!='attention_core' and not torch.equal(tensors['audio_padding',stage],tensors['text_padding',stage]):
            raise ValueError('identical effective Q/K/V required')
    length=1076
    q=tensors['audio_padding','q_rope'].unsqueeze(0)
    k=tensors['audio_padding','k_rope'].unsqueeze(0)
    v=tensors['audio_padding','v_proj'].reshape(length,4,128).transpose(0,1).unsqueeze(0)
    if q.shape!=(1,32,length,128) or k.shape!=v.shape or k.shape!=(1,4,length,128):
        raise ValueError('original valid shape required')
    causal=torch.ones((length,length),dtype=torch.bool).tril().view(1,1,length,length)
    def repeat(x):
        return x[:,:,None,:,:].expand(1,4,8,length,128).reshape(1,32,length,128)
    def compare(a,b):
        d=(a.float()-b.float()).abs()
        return dict(equal=bool(torch.equal(a,b)),differing_elements=int((a!=b).sum()),
                    max_abs_error=float(d.max()),mean_abs_error=float(d.mean()))
    results=[];outputs={}
    for backend in ('default','math'):
        for mask in ('implicit','explicit'):
            for grouping in ('gqa','repeat'):
                name=f'{backend}-{mask}-{grouping}'
                key,val=(k,v) if grouping=='gqa' else (repeat(k),repeat(v))
                ctx=sdpa_kernel(SDPBackend.MATH) if backend=='math' else contextlib.nullcontext()
                with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16),ctx:
                    with profile(activities=[ProfilerActivity.CPU]) as prof:
                        y=torch.nn.functional.scaled_dot_product_attention(q,key,val,
                            attn_mask=causal if mask=='explicit' else None,is_causal=mask=='implicit',
                            enable_gqa=grouping=='gqa',dropout_p=0.,scale=128**-0.5)
                y=y.transpose(1,2).contiguous().reshape(length,4096)
                if not torch.isfinite(y).all():raise ValueError('nonfinite output')
                operators=sorted({e.key for e in prof.key_averages() if 'scaled_dot_product' in e.key})
                torch.save(y,args.output/(name+'.pt'));outputs[name]=y
                results.append(dict(case=name,operators=operators,sha256=file_hash(args.output/(name+'.pt')),
                    reference_unpadded=compare(y,tensors['audio_padding','attention_core']),
                    reference_padded=compare(y,tensors['text_padding','attention_core'])))
    reproduced=torch.equal(outputs['default-implicit-gqa'],tensors['audio_padding','attention_core'])
    write_once(args.output/'result.json',dict(status='diagnostic_complete',reference_unpadded_exact=reproduced,
        cases=results,comparisons=[dict(left=a,right=b,**compare(outputs[a],outputs[b])) for a,b in (
            ('default-implicit-gqa','default-implicit-repeat'),
            ('default-implicit-repeat','default-explicit-repeat'),
            ('default-implicit-gqa','default-explicit-gqa'),
            ('math-implicit-gqa','math-implicit-repeat'),
            ('math-implicit-gqa','math-explicit-gqa'),
            ('math-implicit-repeat','math-explicit-repeat'))],
        source_sha256=file_hash(__file__),torch_version=torch.__version__,elapsed_seconds=time.monotonic()-start,
        weight_updates=0,model_loaded=False,targets_read=False,
        scope='Eight same-shape CPU BF16 SDPA controls with identical captured valid Q/K/V. No padded positions, model inference or task replay. Interpret only if original unpadded endpoint is exact; no NPU or full model fix claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    try:run(a)
    except BaseException as exc:
        write_once(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    write_once(a.output/'terminal.json',dict(exit_code=0))
