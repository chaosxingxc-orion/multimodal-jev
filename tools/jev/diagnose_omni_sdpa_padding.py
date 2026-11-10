#!/usr/bin/env python3
"""Vary excluded Q/K/V positions around authenticated valid TRAIN activations."""
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
    if file_hash(args.parent/'result.json')!='389652411c69533f07a151d7f89acde2c8a3d90e8900f4772327e93032ad329b':
        raise ValueError('accepted first-attention result required')
    parent=json.loads((args.parent/'result.json').read_text())
    captured={}
    for stage in ('q_rope','k_rope','v_proj','attention_core'):
        item=next(s for s in parent['stages'] if s['stage']==stage)
        for condition in ('audio_padding','text_padding'):
            p=args.parent/(condition+'-'+stage+'.pt')
            if file_hash(p)!=item['tensors'][condition]:raise ValueError('captured tensor changed')
            x=torch.load(p,map_location='cpu',weights_only=True)
            if list(x.shape)!=item['shape'] or str(x.dtype)!=item['dtype'] or not torch.isfinite(x).all():
                raise ValueError('invalid captured tensor')
            captured[condition,stage]=x
        if stage!='attention_core' and not torch.equal(captured['audio_padding',stage],captured['text_padding',stage]):
            raise ValueError('identical effective Q/K/V required')
    length=1076;padding=154
    q=captured['audio_padding','q_rope'].unsqueeze(0)
    k=captured['audio_padding','k_rope'].unsqueeze(0)
    v=captured['audio_padding','v_proj'].reshape(length,4,128).transpose(0,1).unsqueeze(0)
    if q.shape!=(1,32,length,128) or k.shape!=v.shape or k.shape!=(1,4,length,128):
        raise ValueError('original valid shape required')
    def extend(x,left,right,fill,seed):
        shape=(x.shape[0],x.shape[1],left+length+right,x.shape[-1])
        if fill=='random':
            g=torch.Generator().manual_seed(seed)
            out=torch.randn(shape,generator=g,dtype=torch.float32).to(x.dtype)
        else:out=torch.zeros(shape,dtype=x.dtype)
        out[:,:,left:left+length,:]=x
        if not torch.equal(out[:,:,left:left+length,:],x):raise ValueError('valid tensor changed')
        return out
    def repeat(x):
        return x[:,:,None,:,:].expand(1,4,8,x.shape[2],128).reshape(1,32,x.shape[2],128)
    def compare(a,b):
        d=(a.float()-b.float()).abs()
        return dict(equal=bool(torch.equal(a,b)),differing_elements=int((a!=b).sum()),
                    max_abs_error=float(d.max()),mean_abs_error=float(d.mean()))
    layouts=[('none',0,0,0,0,'zero'),
             ('left_zero',padding,0,padding,0,'zero'),
             ('left_random',padding,0,padding,0,'random'),
             ('right_zero',0,padding,0,padding,'zero'),
             ('right_random',0,padding,0,padding,'random'),
             ('keys_left_zero',0,0,padding,0,'zero'),
             ('queries_left_zero',padding,0,0,0,'zero')]
    results=[];outputs={}
    tri=torch.ones((length,length),dtype=torch.bool).tril()
    for backend in ('default','math'):
        for layout,ql,qr,kl,kr,fill in layouts:
            name=f'{backend}-{layout}'
            query=extend(q,ql,qr,fill,17);key=repeat(extend(k,kl,kr,fill,18));value=repeat(extend(v,kl,kr,fill,19))
            # Every valid query sees exactly its original causal keys. Other
            # query outputs are excluded and their mask rows remain all false.
            mask=torch.zeros((1,1,query.shape[2],key.shape[2]),dtype=torch.bool)
            mask[0,0,ql:ql+length,kl:kl+length]=tri
            if int(mask.sum())!=length*(length+1)//2 or not torch.equal(mask[0,0,ql:ql+length,kl:kl+length],tri):
                raise ValueError('causal graph changed')
            ctx=sdpa_kernel(SDPBackend.MATH) if backend=='math' else contextlib.nullcontext()
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16),ctx:
                with profile(activities=[ProfilerActivity.CPU]) as prof:
                    y=torch.nn.functional.scaled_dot_product_attention(query,key,value,
                        attn_mask=mask,is_causal=False,dropout_p=0.,scale=128**-0.5)
            y=y[:,:,ql:ql+length,:].transpose(1,2).contiguous().reshape(length,4096)
            if not torch.isfinite(y).all():raise ValueError('nonfinite valid output')
            operators=sorted({e.key for e in prof.key_averages() if 'scaled_dot_product' in e.key})
            torch.save(y,args.output/(name+'.pt'));outputs[name]=y
            results.append(dict(case=name,query_length=query.shape[2],key_length=key.shape[2],query_offset=ql,key_offset=kl,
                excluded_values=fill,valid_causal_edges=int(mask.sum()),operators=operators,
                sha256=file_hash(args.output/(name+'.pt')),
                reference_unpadded=compare(y,captured['audio_padding','attention_core']),
                reference_padded=compare(y,captured['text_padding','attention_core'])))
    pairs=[]
    for backend in ('default','math'):
        pairs.extend((backend+'-none',backend+'-'+layout[0]) for layout in layouts[1:])
        pairs.extend((backend+'-'+side+'_zero',backend+'-'+side+'_random') for side in ('left','right'))
    write_once(args.output/'result.json',dict(status='diagnostic_complete',
        reference_unpadded_exact=bool(torch.equal(outputs['default-none'],captured['audio_padding','attention_core'])),
        reference_padded_exact=bool(torch.equal(outputs['default-left_zero'],captured['text_padding','attention_core'])),
        cases=results,comparisons=[dict(left=a,right=b,**compare(outputs[a],outputs[b])) for a,b in pairs],
        source_sha256=file_hash(__file__),torch_version=torch.__version__,elapsed_seconds=time.monotonic()-start,
        weight_updates=0,model_loaded=False,targets_read=False,
        scope='Fourteen frozen CPU BF16 SDPA shape controls. Identical valid Q/K/V and valid causal graph; excluded positions use synthetic finite zero/random values. Original endpoints must reproduce before interpreting shape mechanism. No model replay, NPU validation or full-model fix.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    try:run(a)
    except BaseException as exc:
        write_once(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    write_once(a.output/'terminal.json',dict(exit_code=0))
