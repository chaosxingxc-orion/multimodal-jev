#!/usr/bin/env python3
"""Frozen full pretrained Thinker CPU forward on registered TRAIN audio.

One first decision per existing six-turn engineering roster; fixed synthetic
response tokens, no labels, sampling, gradients or claim of vLLM parity.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.omni_actor_batch import prepare,evidence
from sample_sfc_omni_learning import load_inputs
from infer_qwen3_omni_probe import verify_assets


def run(config,out):
    import torch
    from transformers import Qwen3OmniMoeProcessor,Qwen3OmniMoeThinkerForConditionalGeneration
    torch.set_num_threads(16);torch.manual_seed(17)
    start=time.monotonic()
    def progress(phase,**kw):atomic_json(out/'progress.json',dict(phase=phase,heartbeat_unix=time.time(),**kw))
    base,rows,preflight,checks=load_inputs(config)
    versions={k:importlib.metadata.version(k) for k in base['versions']}
    if versions!=base['versions']:raise ValueError('fixed runtime required')
    progress('verify_model')
    assets=verify_assets(base)
    processor=Qwen3OmniMoeProcessor.from_pretrained(base['audio_model'],local_files_only=True)
    progress('prepare_audio')
    prepared=prepare(processor,rows)
    for key,pin in evidence(prepared).items():
        if any(checks[key][name]!=v for name,v in pin.items()):raise ValueError('registered prompt differs')
    selected=[(r['id'],r['input']['decisions'][0]['id']) for r in rows]
    atomic_json(out/'identity.json',dict(config=config,versions=versions,model_files=assets['files'],
        selected_keys=selected,selection='first declared decision per registered TRAIN6 turn',
        prompt_evidence=[dict(example_id=k[0],decision_id=k[1],**evidence(prepared)[k]) for k in selected],
        response_text='false',source_files={str(p.relative_to(Path(__file__).parents[2])):file_hash(p)
            for folder in ('src/multimodal_jev','tools/jev') for p in (Path(__file__).parents[2]/folder).glob('*.py')},
        model_arguments=dict(dtype='bfloat16',device_map='cpu',attn_implementation='sdpa'),targets_read=False))
    progress('load_thinker')
    model,loading=Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        base['audio_model'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map='cpu',attn_implementation='sdpa',output_loading_info=True)
    atomic_json(out/'loading-info.json',loading)
    if any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):
        raise ValueError('incomplete or mismatched pretrained Thinker load')
    if any(not k.startswith(('talker.','code2wav.')) for k in loading.get('unexpected_keys',[])):
        raise ValueError('unexpected non-output-branch checkpoint keys')
    model.eval().requires_grad_(False)
    if any(p.is_meta or p.device.type!='cpu' or p.requires_grad for p in model.parameters()):
        raise ValueError('fully materialized frozen CPU Thinker required')
    response=processor.tokenizer.encode('false',add_special_tokens=False)+[processor.tokenizer.eos_token_id]
    if any(type(t) is not int or t<0 for t in response):raise ValueError('valid fixed response tokens required')
    answers=[]
    for i,key in enumerate(selected):
        item=prepared[key];prompt=item['input_ids'];boundary=len(prompt)
        ids=torch.cat((prompt,torch.tensor(response))).unsqueeze(0)
        if ids.shape[1]>4096:raise ValueError('untruncated context budget exceeded')
        progress('forward',done=i,total=len(selected),example_id=key[0],input_tokens=ids.shape[1])
        begin=time.monotonic()
        with torch.inference_mode():
            output=model(input_ids=ids,attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
                input_features=item['input_features'].to(torch.bfloat16),
                feature_attention_mask=item['feature_attention_mask'],use_cache=False,return_dict=True)
            logits=output.logits[0,boundary-1:boundary+len(response)-1].float()
            logp=logits.log_softmax(-1).gather(-1,torch.tensor(response).unsqueeze(-1)).squeeze(-1)
            if not torch.isfinite(logp).all() or (logp>0).any():raise ValueError('invalid response probabilities')
            record=dict(example_id=key[0],decision_id=key[1],audio_count=item['input_features'].shape[0],
                prompt_tokens=boundary,response_token_ids=response,response_log_probs=logp.tolist(),
                seconds=time.monotonic()-begin,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            del output,logits,logp
        answers.append(record);atomic_json(out/f'forward-{i:02d}.json',record)
    progress('verify_model_after_forward',done=len(answers),total=len(selected))
    verify_assets(base)
    atomic_json(out/'result.json',dict(status='passed',responses=answers,model_loaded=True,weight_updates=0,
        parameters=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.monotonic()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,targets_read=False,
        identity_sha256=file_hash(out/'identity.json'),loading_info_sha256=file_hash(out/'loading-info.json'),
        scope='Full pretrained BF16 CPU Thinker, real registered TRAIN single/double audio, fixed synthetic '
              'response tokens. Finite forward only; no generation quality, vLLM parity, verl actor, NPU or training.'))
    progress('complete',done=len(answers),total=len(selected))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--config-hash',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if file_hash(a.config)!=a.config_hash:raise ValueError('registered configuration changed')
    a.output.mkdir(parents=True,exist_ok=False)
    try:run(json.loads(a.config.read_text()),a.output)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
