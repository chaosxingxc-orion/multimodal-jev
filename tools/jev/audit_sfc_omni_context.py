#!/usr/bin/env python3
"""Actual CPU processor context accounting for every official SFC input."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_omni import messages


def run(model,release,output):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    torch.set_num_threads(4);start=time.monotonic()
    manifest=json.loads((release/'MANIFEST.json').read_text())
    identity=dict(model=str(model),release=str(release),manifest_sha256=file_hash(release/'MANIFEST.json'),
        versions={k:importlib.metadata.version(k) for k in ('transformers','torch','numpy','scipy','soundfile')},
        source_files={n:file_hash(Path(__file__).parents[2]/n) for n in
            ('tools/jev/audit_sfc_omni_context.py','src/multimodal_jev/sfc_omni.py','src/multimodal_jev/sfc_audio.py')},
        processor_assets={p.name:file_hash(p) for p in model.iterdir() if p.is_file() and p.suffix in ('.json','.txt')},
        context_limit=4096,output_budget=1024)
    atomic_json(output/'identity.json',identity)
    processor=Qwen3OmniMoeProcessor.from_pretrained(model,local_files_only=True)
    reports={};seen_audio={}
    for split,pin in manifest['splits'].items():
        path=release/(split+'.jsonl')
        if file_hash(path)!=pin['sha256']:raise ValueError('input manifest changed')
        counts=dict(records=0,source_groups=0,min_input_tokens=None,max_input_tokens=0,
            audio_over_limit=0,context_overflow=0,max_audio_seconds=0.)
        ids=set();groups=set()
        record_path=output/(split+'.context.jsonl')
        with path.open() as source,record_path.open('xb') as handle:
            for line in source:
                row=json.loads(line);inputs=row['input'];metadata=row['metadata'];uid=row['example_id']
                if uid in ids or metadata['official_split']!=split:raise ValueError('duplicate or wrong-split input')
                ids.add(uid);groups.add(metadata['source_group']);waves=[]
                for audio,digest in zip(inputs['audios'],metadata['audio_sha256'],strict=True):
                    if audio not in seen_audio:seen_audio[audio]=file_hash(audio)
                    if seen_audio[audio]!=digest:raise ValueError('audio changed')
                    wave,_=load_sfc_audio(audio);waves.append(wave)
                durations=[len(w)/16000 for w in waves]
                counts['max_audio_seconds']=max(counts['max_audio_seconds'],*durations)
                too_long=any(len(w)>processor.feature_extractor.n_samples for w in waves)
                tokens=None
                if not too_long:
                    prompt=processor.apply_chat_template(messages(inputs,waves),tokenize=False,add_generation_prompt=True)
                    batch=processor(text=[prompt],audio=waves,sampling_rate=16000,return_tensors='pt',padding=True)
                    if not torch.isfinite(batch['input_features']).all():raise ValueError('nonfinite features')
                    if batch['input_features'].shape[0]!=len(waves):raise ValueError('audio count differs')
                    tokens=int(batch['attention_mask'][0].sum())
                    counts['max_input_tokens']=max(counts['max_input_tokens'],tokens)
                    counts['min_input_tokens']=tokens if counts['min_input_tokens'] is None else min(counts['min_input_tokens'],tokens)
                    counts['context_overflow']+=tokens+1024>4096
                counts['audio_over_limit']+=too_long;counts['records']+=1
                handle.write(canonical(dict(example_id=uid,audio_seconds=durations,input_tokens=tokens,
                    audio_over_limit=too_long,context_overflow=None if tokens is None else tokens+1024>4096)))
                handle.flush()
                atomic_json(output/'progress.json',dict(phase='processor',split=split,done=counts['records'],
                    total=pin['records'],heartbeat_unix=time.time()))
        counts['source_groups']=len(groups)
        if counts['records']!=pin['records'] or len(groups)!=pin['source_groups']:raise ValueError('incomplete split coverage')
        counts['records_sha256']=file_hash(record_path);reports[split]=counts
        atomic_json(output/'completed-splits.json',reports)
    report=dict(status='passed',identity_sha256=file_hash(output/'identity.json'),splits=reports,
        all_inputs_fit=all(r['audio_over_limit']==r['context_overflow']==0 for r in reports.values()),
        audio_files_hashed=len(seen_audio),elapsed_seconds=time.monotonic()-start,
        scope='Complete CPU HF processor input audit; oversized audio recorded without truncation. No weights, generation, gold scoring, vLLM equivalence or task efficacy claim.')
    atomic_json(output/'result.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('model','release','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a.model,a.release,a.output)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
