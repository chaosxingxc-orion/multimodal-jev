#!/usr/bin/env python3
"""Shape/precision controls for the observed expert36 up-projection difference."""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash


def run(args):
    import torch
    from safetensors import safe_open
    from transformers.integrations import moe

    torch.set_num_threads(16)
    start=time.monotonic()
    read=lambda p:json.loads(p.read_text())
    if file_hash(args.experts/'result.json')!='32f8afe6d9d2fd39bc84c51fc03a956d331a15bd36227499956a8ea2aee3176d':
        raise ValueError('accepted expert stage diagnostic required')
    previous=read(args.experts/'result.json')
    original_identity=read(args.first/'identity.json')
    if file_hash(args.first/'identity.json')!=read(args.first/'result.json')['identity_sha256']:
        raise ValueError('original runtime identity changed')
    if file_hash(inspect.getfile(moe))!=original_identity['source_sha256'][inspect.getfile(moe)]:
        raise ValueError('original grouped implementation required')
    for name in ('model.safetensors.index.json','model-00001-of-00015.safetensors','model-00002-of-00015.safetensors'):
        if file_hash(args.model/name)!=previous['model_files_verified'][name]:
            raise ValueError('original expert weights required')
    index=read(args.model/'model.safetensors.index.json')['weight_map']
    names=[f'thinker.model.layers.1.mlp.experts.36.{part}_proj.weight' for part in ('gate','up')]
    tensors=[]
    for name in names:
        with safe_open(args.model/index[name],framework='pt',device='cpu') as f:
            tensors.append(f.get_tensor(name))
    weight=torch.cat(tensors).contiguous()
    if weight.shape!=(1536,2048) or weight.dtype!=torch.bfloat16:
        raise ValueError('original gate/up shape required')
    reports=[]
    outputs={}
    order=[2,0,5,1,3,4]

    def diff(a,b):
        if a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('incompatible/nonfinite projection comparison')
        return dict(equal=bool(torch.equal(a,b)),max_abs_error=float((a.float()-b.float()).abs().max()),differing_elements=int((a!=b).sum()))

    for record in previous['reports']:
        size,offset=record['size'],record['offset']
        atomic_json(args.output/'progress.json',dict(phase='projection_controls',size=size,offset=offset,heartbeat_unix=time.time()))
        p=args.experts/record['tensor_file']
        if file_hash(p)!=record['tensor_sha256']:
            raise ValueError('original expert capture changed')
        capture=torch.load(p,map_location='cpu',weights_only=True)
        stages=capture['stages']
        routes=stages['router.indices'].flatten()
        sorted_ids,perm=torch.sort(routes)
        selected=perm[sorted_ids==36]
        tokens=selected//8
        x=stages['input'].reshape(-1,2048)[tokens].contiguous()
        reference=stages['gate_up'].reshape(-1,1536)[selected].contiguous()
        offs=torch.tensor([len(x)],dtype=torch.int32)
        aligned=moe._can_use_grouped_mm(x,weight.unsqueeze(0).transpose(-2,-1),offs)
        with torch.inference_mode(),torch.autocast('cpu',enabled=False):
            grouped=moe._grouped_linear(x,weight.unsqueeze(0),offs)
            bf16=torch.mm(x,weight.t())
            fp32_raw=torch.mm(x.float(),weight.float().t())
            fp32=fp32_raw.to(torch.bfloat16)
        variants=dict(original=reference,grouped_single_expert=grouped,mm_bf16=bf16,mm_fp32=fp32)
        checks={k:diff(v,reference) for k,v in variants.items() if k!='original'}
        precise=[]
        for j in range(size):
            row=order[offset+j]
            valid=capture['token_mask'][j].bool()
            # Original valid token 66, up channel 763; fixed before this run.
            padded_token=int(valid.nonzero()[66])
            candidates=(tokens==j*1230+padded_token).nonzero().flatten()
            for idx in candidates.tolist():
                if int(selected[idx]%8)!=3:
                    continue
                exact=float((x[idx].double()*weight[1531].double()).sum())
                precise.append(dict(row=row,valid_token=66,top_k=3,channel=1531,
                    dot_fp64=exact,fp64_rounded_bf16=float(torch.tensor(exact,dtype=torch.float64).to(torch.bfloat16)),
                    fp32_raw=float(fp32_raw[idx,1531]),values={k:float(v[idx,1531]) for k,v in variants.items()}))
            positions=(tokens//1230==j).nonzero().flatten()
            local=selected[positions]-j*1230*8
            sorted_local,permutation=torch.sort(local)
            rows=positions[permutation]
            outputs[size,row]=dict(local_indices=sorted_local,inputs=x[rows],
                                    values={k:v[rows] for k,v in variants.items()})
        p=args.output/f'micro-{size}-offset-{offset}.pt'
        torch.save(dict(selected_token_topk_indices=selected,inputs=x,weight=weight,
                        values=variants,fp32_raw=fp32_raw,token_mask=capture['token_mask']),p)
        r=dict(size=size,offset=offset,expert_tokens=len(x),single_group_can_use_native=aligned,
               pointers_mod16=dict(input=x.data_ptr()%16,weight=weight.data_ptr()%16),
               original_comparisons=checks,precise_points=precise,tensor_file=p.name,tensor_sha256=file_hash(p))
        atomic_json(args.output/f'micro-{size}-offset-{offset}.json',r)
        reports.append(r)
        del capture,stages,x,reference,grouped,bf16,fp32,fp32_raw,variants
    between=[]
    for row in order:
        a,b=outputs[1,row],outputs[2,row]
        if not torch.equal(a['local_indices'],b['local_indices']) or not torch.equal(a['inputs'],b['inputs']):
            raise ValueError('same original expert inputs/routes required across shapes')
        between.append(dict(row=row,expert_tokens=len(a['inputs']),comparisons={k:diff(v,b['values'][k]) for k,v in a['values'].items()}))
    atomic_json(args.output/'result.json',dict(status='diagnostic_complete',reports=reports,between_microbatches=between,
        elapsed_seconds=time.monotonic()-start,weight_updates=0,source_sha256=file_hash(__file__),
        source_result_sha256=file_hash(args.experts/'result.json'),
        scope='Only expert36 gate/up projection on all original saved routed inputs, including padding, with original sort order. Native single-expert grouped, BF16 mm, FP32 mm controls; FP64 point checks at predeclared token66/topk3/channel1531. No full MoE, model, actor, training or NPU parity claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('model','experts','first','output'):
        p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    try:run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0))
