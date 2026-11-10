#!/usr/bin/env python3
"""TRAIN-only real-waveform Omni processor validation without model weights."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_omni import messages


def select(rows):
    if not rows or any(r['metadata']['official_split']!='train_id' for r in rows):
        raise ValueError('TRAIN-only engineering inputs required')
    groups={}
    for r in rows:
        groups.setdefault(r['metadata']['level'],set()).add(r['metadata']['source_group'])
    if set(groups)!={'level_1','level_2','level_3_1','level_3_2'}:
        raise ValueError('all four SFC levels required')
    selected={min(v) for v in groups.values()}
    result=[r for r in rows if r['metadata']['source_group'] in selected]
    if len(result)!=6 or sorted(len(r['input']['audios']) for r in result)!=[1,1,1,1,2,2]:
        raise ValueError('four complete conversations and six turns required')
    return result


def run(model,release,output):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    torch.set_num_threads(4);start=time.monotonic()
    manifest=json.loads((release/'MANIFEST.json').read_text());path=release/'train_id.jsonl'
    if file_hash(path)!=manifest['splits']['train_id']['sha256']:raise ValueError('TRAIN manifest changed')
    rows=select([json.loads(line) for line in path.open()])
    # Drop supervision immediately after deterministic input-only selection.
    rows=[{k:r[k] for k in ('example_id','input','metadata')} for r in rows]
    identity=dict(model=str(model),release=str(release),train_sha256=file_hash(path),
        selected_ids=[r['example_id'] for r in rows],selection='lexicographically first complete source group per level',
        versions={k:importlib.metadata.version(k) for k in ('transformers','torch','numpy','scipy','soundfile')},
        source_files={n:file_hash(Path(__file__).parents[2]/n) for n in
            ('tools/jev/preflight_sfc_omni.py','src/multimodal_jev/sfc_omni.py','src/multimodal_jev/sfc_audio.py')},
        processor_assets={p.name:file_hash(p) for p in model.iterdir() if p.is_file() and p.suffix in ('.json','.txt')})
    atomic_json(output/'identity.json',identity)
    processor=Qwen3OmniMoeProcessor.from_pretrained(model,local_files_only=True)
    records=[]
    for r in rows:
        atomic_json(output/'progress.json',dict(phase='processor',done=len(records),total=len(rows),heartbeat_unix=time.time()))
        waves=[];conversions=[]
        for audio,digest in zip(r['input']['audios'],r['metadata']['audio_sha256'],strict=True):
            if file_hash(audio)!=digest:raise ValueError('audio changed')
            wave,conversion=load_sfc_audio(audio)
            if len(wave)>processor.feature_extractor.n_samples:raise ValueError('audio truncation prohibited')
            waves.append(wave);conversions.append(conversion)
        prompt=processor.apply_chat_template(messages(r['input'],waves),tokenize=False,add_generation_prompt=True)
        batch=processor(text=[prompt],audio=waves,sampling_rate=16000,return_tensors='pt',padding=True)
        ids=batch['input_ids'][0];audio_id=processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
        positions=(ids==audio_id).nonzero().flatten().tolist();spans=[]
        for pos in positions:
            if not spans or pos!=spans[-1][-1]+1:spans.append([])
            spans[-1].append(pos)
        if len(spans)!=len(waves):raise ValueError('expanded audio segment count differs')
        if not torch.isfinite(batch['input_features']).all():raise ValueError('nonfinite audio features')
        order=None
        if len(waves)==2:
            swapped=processor(text=[prompt],audio=waves[::-1],sampling_rate=16000,return_tensors='pt',padding=True)
            torch.testing.assert_close(batch['input_features'],swapped['input_features'].flip(0),atol=1e-5,rtol=1e-5)
            mask_key='feature_attention_mask'
            if not torch.equal(batch[mask_key],swapped[mask_key].flip(0)):raise ValueError('waveform mask order differs')
            if torch.allclose(batch['input_features'],swapped['input_features'],atol=1e-5,rtol=1e-5):
                raise ValueError('swapped distinct audio must change features')
            order=dict(reversal_matches=True,unreversed_features_differ=True,
                maximum_error=float((batch['input_features']-swapped['input_features'].flip(0)).abs().max()))
        records.append(dict(example_id=r['example_id'],level=r['metadata']['level'],audio_count=len(waves),
            conversions=conversions,input_tokens=len(ids),audio_token_lengths=list(map(len,spans)),
            shapes={k:list(v.shape) for k,v in batch.items()},swap_control=order,
            fits_4096_with_1024_output=len(ids)+1024<=4096))
    report=dict(status='passed',records=records,elapsed_seconds=time.monotonic()-start,
        identity_sha256=file_hash(output/'identity.json'),
        scope='Six TRAIN turns in four complete conversations; actual CPU HF processor and waveform-order control. No model weights, vLLM engine, inference quality or full-population context-fit claim.')
    atomic_json(output/'result.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('model','release','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:print(json.dumps(run(a.model,a.release,a.output),indent=2))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
