#!/usr/bin/env python3
"""Real Omni processor extremes selected using audio headers only; no labels."""
import argparse
import inspect
import json
import math
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.sfc_audio import load_sfc_audio


def run(args):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths
    torch.set_num_threads(2)
    for key in ('inventory','headers','prompt'):
        if file_hash(getattr(args,key))!=getattr(args,key+'_hash'):
            raise ValueError('registered input changed: '+key)
    inventory=[json.loads(l) for l in args.inventory.open()]
    headers=[json.loads(l) for l in args.headers.open()];index={r['path']:r for r in headers}
    if len(index)!=len(headers) or any('error' in r for r in headers):
        raise ValueError('complete unique headers required')
    if set(index)!={a['path'] for r in inventory for a in r['audios']}:
        raise ValueError('header inventory coverage differs')
    longest=max(headers,key=lambda r:(r['seconds'],r['path']))
    dialogue=max((r for r in inventory if r['tier']==7),
                 key=lambda r:(sum(index[a['path']]['seconds'] for a in r['audios']),r['query_id']))
    clip=next(a for r in inventory for a in r['audios'] if a['path']==longest['path'])
    selections=[('longest_clip',[clip]),('longest_dialogue',dialogue['audios'])]
    processor=Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    # Reuse exactly the complete-definition system text from the registered
    # text-only probe; strip its user turn using fixed ChatML delimiters.
    saved=args.prompt.read_text();start='<|im_start|>system\n';end='<|im_end|>\n'
    if not saved.startswith(start) or saved.count(start)!=1:
        raise ValueError('registered system prompt layout differs')
    system=saved[len(start):].split(end,1)[0]
    atomic_json(args.output/'identity.json',dict(selections=selections,
        input_hashes={k:file_hash(getattr(args,k)) for k in ('inventory','headers','prompt')},
        processor_source_sha256=file_hash(inspect.getfile(type(processor))),
        feature_extractor_source_sha256=file_hash(inspect.getfile(type(processor.feature_extractor))),
        processor_assets={p.name:file_hash(p) for p in args.model.iterdir()
                          if p.is_file() and p.suffix in ('.json','.txt')},
        n_samples=processor.feature_extractor.n_samples,model_loaded=False,answers_read=False))
    results=[]
    for name,audios in selections:
        begin=time.monotonic();waves=[];conversions=[];content=[]
        atomic_json(args.output/'progress.json',dict(phase='waveforms',selection=name,heartbeat_unix=time.time()))
        for a in audios:
            path=args.root/a['path']
            if file_hash(path)!=a['sha256']:raise ValueError('audio differs from manifest')
            wave,conversion=load_sfc_audio(path);h=index[a['path']]
            if len(wave)!=math.ceil(h['frames']*16000/h['sample_rate']):
                raise ValueError('resampling frame count differs')
            waves.append(wave);conversions.append(conversion)
            if 'turn' in a:
                content.append(dict(type='text',text=f"Conversation turn {a['turn']}, speaker role {a['role']}:"))
            content.append(dict(type='audio',audio=wave))
        messages=[dict(role='system',content=system),dict(role='user',content=content)]
        prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        atomic_json(args.output/'progress.json',dict(phase='processor',selection=name,heartbeat_unix=time.time()))
        batch=processor(text=[prompt],audio=waves,sampling_rate=16000,return_tensors='pt',
                        padding=True,truncation=False)
        features=batch['input_features'];mask=batch['feature_attention_mask']
        hop=processor.feature_extractor.hop_length;max_frames=max(map(len,waves))//hop
        expected=[min((len(w)+hop-1)//hop,max_frames) for w in waves]
        observed=mask.sum(-1).tolist()
        if (observed!=expected or features.shape[0]!=len(waves)
                or features.shape[-1]!=max_frames or not torch.isfinite(features).all()):
            raise ValueError('untruncated complete feature coverage differs')
        # Verify expansion per ordered audio span against the actual processor helper.
        lengths=_get_feat_extract_output_lengths(torch.tensor(observed))
        if isinstance(lengths,tuple):lengths=lengths[-1]
        audio_id=processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
        positions=(batch['input_ids'][0]==audio_id).nonzero().flatten().tolist();spans=[]
        for p in positions:
            if not spans or p!=spans[-1][-1]+1:spans.append([])
            spans[-1].append(p)
        if list(map(len,spans))!=lengths.tolist():raise ValueError('audio token spans differ')
        total=int(batch['attention_mask'].sum())
        record=dict(selection=name,audio_files=len(waves),conversions=conversions,
            valid_frames=observed,expected_frames=expected,audio_token_lengths=lengths.tolist(),
            input_tokens=total,feature_shape=list(features.shape),all_finite=True,
            exceeds_configured_n_samples=[len(w)>processor.feature_extractor.n_samples for w in waves],
            remaining_with_1024_output={str(n):n-total-1024 for n in (8192,16384,32768,65536)},
            seconds=time.monotonic()-begin)
        atomic_json(args.output/(name+'.json'),record);results.append(record)
        del batch,features,mask,waves
    atomic_json(args.output/'result.json',dict(status='passed',results=results,
        model_loaded=False,answers_read=False,runner_sha256=file_hash(__file__),
        scope='Actual HF processor on header-selected longest clip and dialogue with full registry. Complete waveforms, explicit polyphase resampling and no truncation. Provisional role-labeled dialogue layout; not official Tier7 semantics acceptance, full dataset fit, vLLM parity or model inference.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','model','inventory','headers','prompt','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('inventory','headers','prompt'):p.add_argument('--'+key+'-hash',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))
