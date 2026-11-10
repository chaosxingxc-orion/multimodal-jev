#!/usr/bin/env python3
"""Frozen Qwen3-Omni SFC TRAIN probe, preserving complete supplied history."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_audio import load_sfc_audio
from multimodal_jev.sfc_omni import vllm_request
from infer_qwen3_omni_probe import verify_assets
from preflight_sfc_omni import select


def select_inputs(cfg):
    expected=dict(split='train_id',parameters_frozen=True,route='audio-direct',
        dtype='bfloat16',seed=17,max_context_tokens=4096,max_new_tokens=1024,
        sampling={'temperature':0.,'n':1})
    if any(cfg.get(k)!=v for k,v in expected.items()) or cfg.get('adapter') is not None:
        raise ValueError('registered frozen TRAIN audio-direct runtime required')
    root=Path(cfg['processor_preflight']);result=json.loads((root/'run/result.json').read_text())
    identity=json.loads((root/'run/identity.json').read_text())
    if (file_hash(root/'run/result.json')!=cfg['processor_preflight_sha256']
            or file_hash(root/'run/identity.json')!=result['identity_sha256']
            or json.loads((root/'run/terminal.json').read_text())['exit_code']!=0
            or result['status']!='passed' or identity['model']!=cfg['audio_model']
            or any(not r['fits_4096_with_1024_output'] for r in result['records'])):
        raise ValueError('successful pinned processor preflight required')
    for name,digest in identity['source_files'].items():
        if file_hash(Path(__file__).parents[2]/name)!=digest:raise ValueError('processor input implementation changed')
    path=Path(identity['release'])/'train_id.jsonl'
    if file_hash(path)!=identity['train_sha256']:raise ValueError('TRAIN release changed')
    rows=select([json.loads(line) for line in path.open()])
    if [r['example_id'] for r in rows]!=identity['selected_ids']:
        raise ValueError('original six processor-validated turns required')
    return [{k:r[k] for k in ('example_id','input','metadata')} for r in rows],identity


def run(cfg,out):
    rows,preflight=select_inputs(cfg);start=time.monotonic()
    atomic_json(out/'progress.json',dict(phase='verifying_model',heartbeat_unix=time.time()))
    assets=verify_assets(cfg)
    versions={k:importlib.metadata.version(k) for k in ('torch','torch-npu','transformers','vllm','vllm-ascend')}
    if versions!=cfg['versions']:raise ValueError('pinned inference versions required')
    identity=dict(config=cfg,model_files=assets['files'],versions=versions,
        processor_preflight_sha256=cfg['processor_preflight_sha256'],
        inputs_sha256=hashlib.sha256(canonical(rows)).hexdigest(),
        source_files={n:file_hash(Path(__file__).parents[2]/n) for n in
            ('tools/jev/infer_sfc_omni_probe.py','tools/jev/preflight_sfc_omni.py',
             'tools/jev/infer_qwen3_omni_probe.py','src/multimodal_jev/sfc_omni.py','src/multimodal_jev/sfc_audio.py')})
    atomic_json(out/'identity.json',identity)
    atomic_json(out/'selection.json',dict(ids=[r['example_id'] for r in rows]))
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM,SamplingParams
    processor=Qwen3OmniMoeProcessor.from_pretrained(cfg['audio_model'],local_files_only=True)
    atomic_json(out/'progress.json',dict(phase='engine_load',heartbeat_unix=time.time()))
    engine=LLM(model=cfg['audio_model'],dtype='bfloat16',tensor_parallel_size=2,seed=17,
        max_model_len=4096,max_num_seqs=1,max_num_batched_tokens=4096,
        gpu_memory_utilization=.85,enforce_eager=True,trust_remote_code=False,
        limit_mm_per_prompt={'audio':2,'image':0,'video':0},enable_prefix_caching=False)
    params=SamplingParams(temperature=0.,max_tokens=1024,n=1,seed=17)
    records=[];cache=out/'records';cache.mkdir()
    try:
        for row in rows:
            waves=[];conversions=[]
            for path,digest in zip(row['input']['audios'],row['metadata']['audio_sha256'],strict=True):
                if file_hash(path)!=digest:raise ValueError('audio changed')
                wave,conversion=load_sfc_audio(path)
                if len(wave)>processor.feature_extractor.n_samples:raise ValueError('audio truncation prohibited')
                waves.append(wave);conversions.append(conversion)
            request=vllm_request(processor,row['input'],waves)
            # Compare the actual engine prompt IDs against the same HF processor,
            # not just a duration-derived estimate or equal token count.
            batch=processor(text=[request['prompt']],audio=waves,sampling_rate=16000,return_tensors='pt',padding=True)
            expected_ids=batch['input_ids'][0].tolist()
            if len(expected_ids)+1024>4096:raise ValueError('complete prompt exceeds context')
            before=time.monotonic();responses=engine.generate([request],params,use_tqdm=False)
            if len(responses)!=1 or not responses[0].finished or len(responses[0].outputs)!=1:
                raise ValueError('one complete generation required')
            response=responses[0];answer=response.outputs[0]
            if list(response.prompt_token_ids)!=expected_ids:raise ValueError('HF and vLLM expanded prompt IDs differ')
            if answer.finish_reason not in ('stop','length') or len(answer.token_ids)>1024:
                raise ValueError('unexpected generation termination')
            record=dict(example_id=row['example_id'],source_group=row['metadata']['source_group'],
                split='train_id',raw_text=answer.text,token_ids=list(answer.token_ids),
                input_tokens=len(expected_ids),hf_vllm_prompt_ids_equal=True,audio_count=len(waves),
                input_sha256=hashlib.sha256(canonical(row['input'])).hexdigest(),
                audio_sha256=row['metadata']['audio_sha256'],audio_conversion=conversions,
                prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest(),
                finish_reason=answer.finish_reason,length_limited=answer.finish_reason=='length',
                seconds=time.monotonic()-before)
            atomic_json(cache/(hashlib.sha256(row['example_id'].encode()).hexdigest()+'.json'),record)
            records.append(record)
            atomic_json(out/'progress.json',dict(phase='generation',done=len(records),total=len(rows),heartbeat_unix=time.time()))
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=30)
    atomic_json(out/'progress.json',dict(phase='final_model_verification',done=len(records),total=len(rows),heartbeat_unix=time.time()))
    verify_assets(cfg)
    path=out/'predictions.jsonl';path.write_bytes(b''.join(map(canonical,records)))
    atomic_json(out/'result.json',dict(status='passed',requests=len(rows),predictions=len(records),
        predictions_sha256=file_hash(path),weight_updates=0,model_files_unchanged=True,
        length_limited=sum(r['length_limited'] for r in records),elapsed_seconds=time.monotonic()-start,
        scope='Six complete TRAIN turns; actual vLLM multi-audio generation. No quality threshold, TEST claim, or proof of causal use of both audio recordings.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('config','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(json.loads(a.config.read_text()),a.output)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
