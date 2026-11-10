#!/usr/bin/env python3
"""All original TRAIN6 frozen forwards with unreachable text K/V omitted."""
import argparse
import importlib.metadata
import inspect
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.omni_actor_batch import prepare,evidence
from multimodal_jev.omni_compact_attention import compact_text_attention
from diagnose_omni_pretrained_layers import (
    IDENTITY_HASH,HF_HASH,complete_dependency,pair_inputs)
from probe_omni_pretrained_verl import read
from probe_omni_pretrained_forward import loading_report
from sample_sfc_omni_learning import load_inputs
from infer_qwen3_omni_probe import verify_assets

SDPA_HASH='b72c5c96ea070d594f36e774cfc799160a0ecb157353502a61eb3c5ea1e35b46'


def run(args):
    import torch
    from transformers import Qwen3OmniMoeProcessor,Qwen3OmniMoeThinkerForConditionalGeneration
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    torch.set_num_threads(16);torch.manual_seed(17);begin=time.monotonic()
    progress=lambda phase,**kw:atomic_json(args.output/'progress.json',dict(phase=phase,heartbeat_unix=time.time(),**kw))
    root=args.precision_root
    if file_hash(root/'run/identity.json')!=IDENTITY_HASH:raise ValueError('original precision identity required')
    original=read(root/'run/identity.json');base,rows,_,checks=load_inputs(original['config'])
    if {k:importlib.metadata.version(k) for k in base['versions']}!=base['versions']:
        raise ValueError('fixed runtime required')
    original_sdpa=hf.ALL_ATTENTION_FUNCTIONS.get_interface('sdpa',hf.eager_attention_forward)
    if file_hash(inspect.getfile(hf))!=HF_HASH or file_hash(inspect.getfile(original_sdpa))!=SDPA_HASH:
        raise ValueError('original HF modeling and SDPA implementation required')
    dependency=complete_dependency(root,args.inspection,args.inspection_hash)
    progress('verify_assets');assets=verify_assets(base)
    if assets['files']!=original['model_files']:raise ValueError('original model identity required')
    processor=Qwen3OmniMoeProcessor.from_pretrained(base['audio_model'],local_files_only=True)
    response=processor.tokenizer.encode('false',add_special_tokens=False)+[processor.tokenizer.eos_token_id]
    progress('prepare_audio');prepared=prepare(processor,rows,waveform_boundary='raw')
    for key,values in evidence(prepared).items():
        if any(checks[key][name]!=value for name,value in values.items()):raise ValueError('prompt changed')
    keys=[(r['id'],r['input']['decisions'][0]['id']) for r in rows]
    if (len(keys)!=6 or len(set(keys))!=6 or [list(k) for k in keys]!=original['keys']
            or response!=original['response_ids']):raise ValueError('complete original TRAIN6 required')
    items=[prepared[k] for k in keys]
    pw=max(len(i['input_ids']) for i in items);aw=max(i['input_features'].shape[-1] for i in items)
    if (pw!=original['prompt_width'] or aw!=original['audio_width'] or pw+len(response)>4096
            or processor.tokenizer.pad_token_id is None):raise ValueError('original full context required')
    pairs=[pair_inputs(i,response,pw,aw,processor.tokenizer.pad_token_id) for i in items]
    with compact_text_attention(hf) as probe_records:
        if hf.ALL_ATTENTION_FUNCTIONS.get_interface('sdpa',hf.eager_attention_forward) is original_sdpa:
            raise ValueError('actual HF registry did not install diagnostic')
    if probe_records or hf.ALL_ATTENTION_FUNCTIONS.get_interface('sdpa',hf.eager_attention_forward) is not original_sdpa:
        raise ValueError('actual HF registry restoration failed')
    atomic_json(args.output/'identity.json',dict(precision_identity_sha256=IDENTITY_HASH,keys=keys,
        response_ids=response,model_files=assets['files'],dependency=dependency,preflight_only=args.preflight_only,
        runner_sha256=file_hash(__file__),adapter_sha256=file_hash(inspect.getfile(inspect.unwrap(compact_text_attention))),
        hf_sha256=HF_HASH,sdpa_sha256=SDPA_HASH,waveform_boundary='raw',
        padding_counts=[int(y['attention_mask'].numel()-y['attention_mask'].sum()) for x,y in pairs],
        response_logp_max_error_threshold=.1,targets_read=False,weight_updates=0))
    if args.preflight_only:
        atomic_json(args.output/'result.json',dict(status='preflight_passed',rows=6,model_loaded=False,
            registry_restored=True,weight_updates=0));return
    progress('load_thinker')
    model,loading=Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(base['audio_model'],
        local_files_only=True,dtype=torch.bfloat16,device_map='cpu',attn_implementation='sdpa',output_loading_info=True)
    loading=loading_report(loading);atomic_json(args.output/'loading-info.json',loading)
    if (any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs'))
            or any(not k.startswith(('talker.','code2wav.')) for k in loading.get('unexpected_keys',[]))):
        raise ValueError('complete original Thinker required')
    model.eval().requires_grad_(False)
    if any(p.is_meta or p.device.type!='cpu' for p in model.parameters()):raise ValueError('materialized CPU weights required')
    reports=[];labels=torch.tensor(response).unsqueeze(-1)
    for row,(unpadded,inputs) in enumerate(pairs):
        progress('forward',row=row);start=time.monotonic()
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
            with compact_text_attention(hf) as records:
                output=model(**inputs)
                logits=output.logits[0,-len(response)-1:-1].detach().clone();del output
        if hf.ALL_ATTENTION_FUNCTIONS.get_interface('sdpa',hf.eager_attention_forward) is not original_sdpa:
            raise ValueError('SDPA patch leaked after forward')
        valid=int(inputs['attention_mask'].sum());total=inputs['input_ids'].shape[1]
        if (len(records)!=48 or [r['layer'] for r in records]!=list(range(48))
                or any(r['query_length']!=total or r['key_lengths']!=[valid] for r in records)
                or any(r['mode']!=('unchanged_no_mask' if valid==total else 'compact_masked_keys') for r in records)):
            raise ValueError('all original text attention layers and valid key counts required')
        probability=logits.float().log_softmax(-1).gather(-1,labels).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(probability).all():raise ValueError('nonfinite candidate')
        comparisons={}
        for cell in ('audio_padding','text_padding'):
            prior_path=root/'run'/f'row-{row:02d}-{cell}-logits.pt'
            pin=read(root/'run'/f'row-{row:02d}-{cell}.json')['logits_sha256']
            if file_hash(prior_path)!=pin:raise ValueError('reference logits changed')
            previous=torch.load(prior_path,map_location='cpu',weights_only=True)
            if previous.shape!=logits.shape or previous.dtype!=logits.dtype:raise ValueError('logit identity differs')
            old=previous.float().log_softmax(-1).gather(-1,labels).squeeze(-1)
            comparisons[cell]=dict(exact_logits=bool(torch.equal(logits,previous)),
                max_logits_error=float((logits.float()-previous.float()).abs().max()),
                max_response_logp_error=float((probability-old).abs().max()),reference_sha256=pin)
        path=args.output/f'row-{row:02d}-logits.pt';torch.save(logits,path)
        report=dict(row=row,key=keys[row],valid_tokens=valid,padding=total-valid,attention_records=records,
            comparisons=comparisons,fp32_log_probs=probability.tolist(),logits_sha256=file_hash(path),
            response_gate_passed=comparisons['audio_padding']['max_response_logp_error']<=.1,
            elapsed_seconds=time.monotonic()-start)
        atomic_json(args.output/f'row-{row:02d}.json',report);reports.append(report)
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('unexpected gradients')
    progress('verify_weights_after');verify_assets(base)
    atomic_json(args.output/'result.json',dict(status='diagnostic_complete',reports=reports,
        all_six_response_gates_passed=all(r['response_gate_passed'] for r in reports),threshold=.1,
        elapsed_seconds=time.monotonic()-begin,weight_updates=0,
        scope='All six original TRAIN rows, frozen CPU BF16 SDPA candidate with unreachable text keys removed; unchanged input/response layout. No verl actor, NPU parity, optimizer, training or task improvement claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('precision-root','inspection','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--inspection-hash',required=True);p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))
