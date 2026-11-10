#!/usr/bin/env python3
"""Replay only first text attention from accepted actual actor prefix tensors."""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.omni_compact_attention import compact_text_attention


def run(args):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
    torch.set_num_threads(16);start=time.monotonic()
    if file_hash(inspect.getfile(hf))!='a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340':
        raise ValueError('original HF source required')
    if file_hash(args.prefix/'result.json')!='fb10bfc904f23e4ebb408ce3a685c9eac66a55f3bfd48b50dfae95443c5e08e7':
        raise ValueError('accepted actual actor prefix captures required')
    prefix=json.loads((args.prefix/'result.json').read_text())
    if prefix['identity_sha256']!=file_hash(args.prefix/'identity.json'):raise ValueError('prefix identity changed')
    identity=json.loads((args.prefix/'identity.json').read_text())
    if file_hash(args.trace/'result.json')!='23ee087e133334428c0e0fcc85b728a4f7459350954b32ef7a22b1b6a284a161':
        raise ValueError('original complete reference trace required')
    trace_reports=json.loads((args.trace/'result.json').read_text())['reports']
    if file_hash(inspect.getfile(inspect.unwrap(compact_text_attention)))!=identity['source_files']['src/multimodal_jev/omni_compact_attention.py']:
        raise ValueError('original actor compact attention adapter required')
    weights={r['file']:r['sha256'] for r in identity['model_files']}
    for name in ('config.json','model-00001-of-00015.safetensors'):
        if file_hash(args.model/name)!=weights[name]:raise ValueError('original model assets required')
    config=Qwen3OmniMoeTextConfig(**json.loads((args.model/'config.json').read_text())['thinker_config']['text_config'])
    config._attn_implementation='sdpa'
    attn=hf.Qwen3OmniMoeThinkerTextAttention(config,0).to(torch.bfloat16).eval().requires_grad_(False)
    norm=hf.Qwen3OmniMoeThinkerTextRMSNorm(config.hidden_size,eps=config.rms_norm_eps).to(torch.bfloat16).eval().requires_grad_(False)
    with safe_open(args.model/'model-00001-of-00015.safetensors',framework='pt',device='cpu') as f:
        attn.load_state_dict({k:f.get_tensor('thinker.model.layers.0.self_attn.'+k) for k in attn.state_dict()},strict=True)
        norm.load_state_dict({'weight':f.get_tensor('thinker.model.layers.0.input_layernorm.weight')},strict=True)
    reports=[];all_values={}
    for record in prefix['records']:
        size,offset=record['micro_batch_size'],record['offset']
        atomic_json(args.output/'progress.json',dict(phase='first_attention',size=size,offset=offset,heartbeat_unix=time.time()))
        path=args.prefix/record['tensor_file']
        if file_hash(path)!=record['tensor_sha256']:raise ValueError('prefix tensor changed')
        inp=torch.load(path,map_location='cpu',weights_only=True);values={};hooks=[]
        def save(name,t):values[name]=t.detach().clone()
        def after(name):
            def hook(module,a,out):save(name,out)
            return hook
        for name in ('q_proj','k_proj','v_proj','q_norm','k_norm'):
            hooks.append(getattr(attn,name).register_forward_hook(after(name)))
        def before_output(module,a):save('before_o_proj',a[0])
        hooks.append(attn.o_proj.register_forward_pre_hook(before_output))
        try:
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16),compact_text_attention(hf) as events:
                registry=hf.ALL_ATTENTION_FUNCTIONS
                compact=registry.get_interface('sdpa',hf.eager_attention_forward)
                def capture(module,q,k,v,mask,**kwargs):
                    for name,t in [('rotated_q',q),('rotated_k',k),('sdpa_v',v)]:
                        save(name,t.transpose(1,2))
                    out,w=compact(module,q,k,v,mask,**kwargs);save('sdpa_output',out);return out,w
                registry.register('sdpa',capture)
                try:
                    normalized=norm(inp['hidden_states']);save('input_norm',normalized)
                    out,_=attn(normalized,position_embeddings=inp['position_embeddings'],attention_mask=inp['attention_mask'])
                    save('attention_output',out)
                finally:registry.register('sdpa',compact)
        finally:
            for hook in hooks:hook.remove()
        saved=[]
        for j in range(size):
            original_row=identity['order'][offset+j];mask=inp['token_mask'][j].bool()
            row_values={k:v[j][mask].contiguous() for k,v in values.items()}
            tensor=args.output/f'micro-{size}-row-{original_row}.pt';torch.save(row_values,tensor)
            all_values[size,original_row]=row_values
            old=json.loads((args.trace/f'row-{original_row:02d}.json').read_text())
            if old!=trace_reports[original_row]:raise ValueError('reference trace row changed')
            ref=args.trace/f'row-{original_row:02d}-trace/layer-00.attention.pt'
            if file_hash(ref)!=old['reference']['layer-00.attention']['sha256']:raise ValueError('reference attention changed')
            original=torch.load(ref,map_location='cpu',weights_only=True)
            delta=(row_values['attention_output'].float()-original.float()).abs()
            saved.append(dict(row=original_row,tensor_file=tensor.name,tensor_sha256=file_hash(tensor),
                reference_sha256=file_hash(ref),reference_exact=bool(torch.equal(row_values['attention_output'],original)),
                reference_max_error=float(delta.max())))
        reports.append(dict(size=size,offset=offset,attention_records=events,rows=saved))
        atomic_json(args.output/f'micro-{size}-offset-{offset}.json',reports[-1])
    differences=[]
    for row in identity['order']:
        events=[]
        for name,v in all_values[1,row].items():
            other=all_values[2,row][name]
            if v.shape!=other.shape or v.dtype!=other.dtype:raise ValueError('stage shapes differ')
            delta=(v.float()-other.float()).abs()
            events.append(dict(stage=name,equal=bool(torch.equal(v,other)),max_abs_error=float(delta.max()),differing_elements=int((v!=other).sum())))
        differences.append(dict(row=row,events=events,first_difference=next((e['stage'] for e in events if not e['equal']),None)))
    atomic_json(args.output/'result.json',dict(status='diagnostic_complete',reports=reports,differences=differences,
        elapsed_seconds=time.monotonic()-start,weight_updates=0,source_sha256=file_hash(__file__),
        scope='First text attention and input norm only, real saved prefix states/positions/masks and published weights; no audio re-encoding, MoE, later layers, full actor probabilities or training. Per-stage valid tensors persisted.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('model','prefix','trace','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    try:run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0))
