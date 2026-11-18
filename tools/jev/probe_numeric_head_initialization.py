#!/usr/bin/env python3
"""Read two real frozen weight rows; synthetic hidden-state algebra only."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile

import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_rows(root):
    config=json.loads((root/'config.json').read_text())
    tokenizer=json.loads((root/'tokenizer.json').read_text())
    vocab=tokenizer['model']['vocab']
    tokens=[vocab['A'],vocab['B']]
    hidden=config['text_config']['hidden_size']; size=config['text_config']['vocab_size']
    if config.get('tie_word_embeddings') is not True or len(set(tokens))!=2:
        raise ValueError('explicit tied head and distinct A/B tokens required')
    # Narrow read-only parser for this local BF16 tensor; not a general loader.
    path=root/'model.safetensors'
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if not 0<n<16*1024*1024:raise ValueError('unexpected header size')
        header=json.loads(f.read(n))
        key='model.language_model.embed_tokens.weight'
        if 'lm_head.weight' in header:raise ValueError('separate head needs separate verification')
        entry=header[key]; start,end=entry['data_offsets']
        if (entry['dtype']!='BF16' or entry['shape']!=[size,hidden]
                or end-start!=size*hidden*2 or start<0 or 8+n+end>path.stat().st_size):
            raise ValueError('unexpected embedding layout')
        rows=[];digests=[]
        for token in tokens:
            if not 0<=token<size:raise ValueError('token outside vocabulary')
            f.seek(8+n+start+token*hidden*2);raw=f.read(hidden*2)
            if len(raw)!=hidden*2:raise ValueError('truncated row')
            digests.append(hashlib.sha256(raw).hexdigest())
            rows.append(torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).clone())
    return torch.stack(rows),dict(tensor=key,tokens=tokens,hidden_size=hidden,
        row_sha256=digests,model_weight_sha256=sha(path),config_sha256=sha(root/'config.json'),
        tokenizer_sha256=sha(root/'tokenizer.json'))


def run(root):
    torch.set_num_threads(2)
    rows,identity=read_rows(root)
    generator=torch.Generator().manual_seed(317)
    h=torch.randn(128,rows.shape[1],generator=generator,dtype=torch.float64)
    h=h/(h.square().mean(-1,keepdim=True).sqrt())
    y=(torch.arange(len(h))%2).double()
    cases=[]
    for dtype in (torch.float64,torch.float32,torch.bfloat16):
        x=h.to(dtype).requires_grad_();w=rows.to(dtype)
        # Initialize in FP32 then round once for execution/export.
        delta=(rows[1].float()-rows[0].float()).to(dtype)
        z=torch.nn.functional.linear(x,w)
        scalar=torch.nn.functional.linear(x,delta[None,:]).squeeze(-1)
        norm_dtype=torch.float64 if dtype==torch.float64 else torch.float32
        logits=z.to(norm_dtype);score=scalar.to(norm_dtype)
        p=logits.softmax(-1)[:,1];q=score.sigmoid()
        ce=torch.nn.functional.cross_entropy(logits,y.long())
        bce=torch.nn.functional.binary_cross_entropy_with_logits(score,y.to(norm_dtype))
        grad_a=torch.autograd.grad(ce,x,retain_graph=True)[0]
        grad_b=torch.autograd.grad(bce,x)[0]
        with tempfile.TemporaryDirectory(prefix='jev-numeric-head-') as td:
            path=Path(td)/'head.pt'
            torch.save({'weight':delta[None,:].detach(),'bias':torch.zeros(1,dtype=dtype)},path)
            restored=torch.load(path,weights_only=True)
            restored_score=torch.nn.functional.linear(x.detach(),restored['weight'],restored['bias']).squeeze(-1)
            exact_restore=torch.equal(restored_score,scalar.detach())
        cases.append(dict(dtype=str(dtype),max_probability_error=float((p-q).abs().max().detach()),
            max_hidden_gradient_error=float((grad_a.double()-grad_b.double()).abs().max()),
            candidate_nll=float(ce.detach()),scalar_nll=float(bce.detach()),
            different_threshold_decisions=int(((p>=.5)!=(q>=.5)).sum()),
            serialized_head_exact_restore=exact_restore))
    if cases[0]['max_probability_error']>1e-12 or cases[0]['max_hidden_gradient_error']>1e-12:
        raise ValueError('FP64 algebra mismatch')
    if not all(c['serialized_head_exact_restore'] for c in cases):raise ValueError('restore differs')
    return dict(schema='numeric-head-real-row-initialization-v1',identity=identity,
        hidden_states=dict(source='synthetic_normalized_gaussian_not_model_activations',seed=317,count=128),
        cases=cases,scalar_parameters=rows.shape[1]+1,training_dispatch_allowed=False,
        scope='Actual frozen A/B weight rows with synthetic hidden states; same-position linear readout and hidden-gradient diagnostic. No audio forward, special token, LoRA update, vLLM/NPU parity, calibration or efficacy acceptance.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=run(a.model)
    result['probe_sha256']=sha(__file__)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result['cases'],indent=2))
