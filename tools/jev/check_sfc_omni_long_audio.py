#!/usr/bin/env python3
"""Resolve conservative 30-second audit exclusions using actual untruncated features."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_omni import messages


def run(prior,out):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    torch.set_num_threads(4)
    identity=json.loads((prior/'run/identity.json').read_text())
    result=json.loads((prior/'run/result.json').read_text())
    if json.loads((prior/'run/terminal.json').read_text())['exit_code']!=0:raise ValueError('completed audit required')
    if file_hash(prior/'run/identity.json')!=result['identity_sha256']:raise ValueError('audit identity changed')
    processor=Qwen3OmniMoeProcessor.from_pretrained(identity['model'],local_files_only=True)
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessorKwargs
    if Qwen3OmniMoeProcessorKwargs._defaults['audio_kwargs']['truncation'] is not False:
        raise ValueError('untruncated official defaults required')
    records=[];release=Path(identity['release']);manifest=json.loads((release/'MANIFEST.json').read_text())
    for split,summary in result['splits'].items():
        context=prior/'run'/(split+'.context.jsonl')
        if file_hash(context)!=summary['records_sha256']:raise ValueError('context audit changed')
        wanted={r['example_id'] for r in map(json.loads,context.read_text().splitlines()) if r['audio_over_limit']}
        if not wanted:continue
        if split!='train_id':raise ValueError('registered supplement is TRAIN only')
        path=release/(split+'.jsonl')
        if file_hash(path)!=manifest['splits'][split]['sha256']:raise ValueError('release changed')
        found=set()
        for line in path.open():
            row=json.loads(line)
            if row['example_id'] not in wanted:continue
            found.add(row['example_id']);waves=[]
            for audio,digest in zip(row['input']['audios'],row['metadata']['audio_sha256'],strict=True):
                if file_hash(audio)!=digest:raise ValueError('audio changed')
                waves.append(load_sfc_audio(audio)[0])
            prompt=processor.apply_chat_template(messages(row['input'],waves),tokenize=False,add_generation_prompt=True)
            batch=processor(text=[prompt],audio=waves,sampling_rate=16000,return_tensors='pt',padding=True)
            # Feature mask is sampled every hop_length from the full waveform mask.
            expected=[(len(w)+processor.feature_extractor.hop_length-1)//processor.feature_extractor.hop_length for w in waves]
            observed=batch['feature_attention_mask'].sum(-1).tolist()
            if expected!=observed:raise ValueError('audio frame coverage differs; possible truncation')
            if not torch.isfinite(batch['input_features']).all():raise ValueError('nonfinite features')
            tokens=int(batch['attention_mask'][0].sum())
            records.append(dict(example_id=row['example_id'],audio_seconds=[len(w)/16000 for w in waves],
                expected_frames=expected,observed_frames=observed,input_tokens=tokens,fits_4096_with_1024_output=tokens+1024<=4096))
            atomic_json(out/'progress.json',dict(done=len(records),heartbeat_unix=time.time()))
        if found!=wanted:raise ValueError('incomplete supplement')
    if len(records)!=8:raise ValueError('all eight conservatively excluded TRAIN requests required')
    atomic_json(out/'result.json',dict(status='passed',prior_result_sha256=file_hash(prior/'run/result.json'),
        records=records,all_fit=all(r['fits_4096_with_1024_output'] for r in records),
        interpretation='feature_extractor.n_samples is not an Omni model audio limit; official processor disables truncation',
        scope='All eight original audit exclusions, actual HF features and full frame masks; no vLLM or generation evidence. Original audit retained.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a.prior,a.output)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
