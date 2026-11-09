#!/usr/bin/env python3
"""Resumable real-vLLM SFC inference using the explicit chronological ASR port.

No HF fallback. Run only on allocated devices. This script is inference-only;
official and strict scores are computed separately from persisted raw outputs.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.sfc_audio import load_sfc_audio, vllm_request


def run(config, output):
    from vllm import LLM, SamplingParams
    from transformers import Qwen3ASRProcessor
    checkpoint=Path(config['checkpoint']);manifest=Path(config['manifest'])
    files={str(p.relative_to(checkpoint)):file_hash(p) for p in sorted(checkpoint.rglob('*'))
           if p.is_file() and p.suffix in {'.json','.jinja','.safetensors','.model','.txt'}}
    if not any(name.endswith('.safetensors') for name in files):raise ValueError('weights missing')
    identity={'config':config,'checkpoint_files':files,'manifest_sha256':file_hash(manifest),
              'adapter_sha256':file_hash(Path(__file__).parents[2]/'src/multimodal_jev/sfc_audio.py'),
              'runner_sha256':file_hash(__file__)}
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text())!=identity:
        raise ValueError('resume identity mismatch')
    atomic_json(output/'identity.json',identity)
    selected=config.get('example_ids')
    if selected is not None and (not selected or len(set(selected))!=len(selected)):
        raise ValueError('nonempty unique selection required')
    # Strip supervision before making any generation requests or retaining rows.
    rows=[]
    with manifest.open() as handle:
        for line in handle:
            row=json.loads(line)
            if selected is not None and row['example_id'] not in selected:continue
            if row['metadata']['official_split']!=config['official_split']:
                raise ValueError('split mismatch')
            rows.append({key:row[key] for key in ('example_id','input','metadata')})
    ids=[r['example_id'] for r in rows]
    if not rows or len(ids)!=len(set(ids)):raise ValueError('nonempty unique inputs required')
    if selected is not None and set(ids)!=set(selected):raise ValueError('selection IDs missing')
    processor=Qwen3ASRProcessor.from_pretrained(checkpoint,local_files_only=True)
    atomic_json(output/'progress.json',{'phase':'engine_load','heartbeat_unix':time.time()})
    engine=LLM(model=str(checkpoint),tokenizer=str(checkpoint),trust_remote_code=False,
               limit_mm_per_prompt={'audio':2},**config['engine'])
    sampling=SamplingParams(temperature=0,max_tokens=config['max_tokens'],n=1)
    cache=output/'records';cache.mkdir(exist_ok=True);records=[]
    for index,row in enumerate(rows):
        key=row['example_id'];path=cache/(hashlib.sha256(key.encode()).hexdigest()+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            if record['example_id']!=key:raise ValueError('cached ID mismatch')
        else:
            waves=[];conversions=[]
            for audio,digest in zip(row['input']['audios'],row['metadata']['audio_sha256']):
                if file_hash(audio)!=digest:raise ValueError('audio changed')
                wave,conversion=load_sfc_audio(audio);waves.append(wave);conversions.append(conversion)
            request=vllm_request(row['input'],waves)
            start=time.monotonic()
            results=engine.generate([request],sampling,use_tqdm=False)
            if len(results)!=1 or not results[0].finished or len(results[0].outputs)!=1:
                raise ValueError('incomplete engine result')
            response=results[0].outputs[0]
            if response.finish_reason not in {'stop','length'}:raise ValueError('abnormal engine termination')
            tokens=list(response.token_ids)
            record={'example_id':key,'raw_text':processor.tokenizer.decode(tokens,skip_special_tokens=True),
                    'token_ids':tokens,'finish_reason':response.finish_reason,
                    'audio_conversion':conversions,'generation_seconds':time.monotonic()-start}
            atomic_json(path,record)
        records.append(record)
        atomic_json(output/'progress.json',{'phase':'generation','done':index+1,'total':len(rows),'heartbeat_unix':time.time()})
    predictions=output/'predictions.jsonl'
    predictions.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    atomic_json(output/'result.json',{'status':'passed','records':len(records),'official_split':config['official_split'],
                'predictions_sha256':file_hash(predictions),
                'scope':'Qwen3-ASR chronological prompt port; offline supplied-history SFC; not official Omni model reproduction'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'result.json').exists():raise FileExistsError('completed inference')
        try:
            run(json.loads(args.config.read_text()),args.output)
        except BaseException as exc:
            atomic_json(args.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
            raise
        atomic_json(args.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})
