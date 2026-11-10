#!/usr/bin/env python3
"""All release records, frozen audio-to-JSON; no target access or replay."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_audio import prepare_request
from multimodal_jev.audio2tool_engine import ASSET_AUDIT_SHA,ENGINE,VERSIONS,prior_records
from multimodal_jev.audio2tool_inference import (INPUT_SHA,BUDGET_SHA,Ledger,accept_generation,
    read_population,require_capacity,write_once)
from multimodal_jev.baseline import atomic_json,file_hash


def run(args):
    rows,budgets=read_population(args.inputs,args.budgets)
    if args.preflight_only:
        # Full roster check only; real waveform/processor acceptance is separate.
        write_once(args.output/'result.json',dict(status='population_preflight_passed',records=len(rows),
            unique_inputs=30731,inputs_sha256=INPUT_SHA,budgets_sha256=BUDGET_SHA,
            maximum_input_tokens=max(b['input_tokens'] for b in budgets),model_loaded=False,
            scope='Complete pinned release roster and budgets only. No waveform decoding, capacity acceptance or inference.'))
        return
    capacity=require_capacity(args.capacity_root,args.capacity_inspection,args.capacity_inspection_hash,
                              args.capacity_container_id,args.processor_run)
    versions={k:importlib.metadata.version(k) for k in VERSIONS}
    if versions!=VERSIONS or file_hash(args.asset_audit)!=ASSET_AUDIT_SHA:
        raise ValueError('fixed engine versions and original model audit required')
    audit=json.loads(args.asset_audit.read_text())
    if audit['model']!=str(args.model):raise ValueError('registered original model path required')
    def verify_assets():
        for entry in audit['files']:
            if file_hash(args.model/entry['file'])!=entry['sha256']:raise ValueError('model asset changed')
    start=time.monotonic();atomic_json(args.output/'progress.json',dict(phase='model_verification',heartbeat_unix=time.time()))
    verify_assets()
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM,SamplingParams
    import torch
    processor=Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    _,processor_identity=prior_records(args.processor_run)
    import inspect
    if file_hash(inspect.getfile(type(processor)))!=processor_identity['hf_processor_source_sha256']:
        raise ValueError('accepted HF processor source required')
    registry=args.registry.read_bytes()
    repo=Path(__file__).resolve().parents[2]
    source_files={str(p.relative_to(repo)):file_hash(p) for p in sorted((repo/'src/multimodal_jev').glob('*.py'))}
    source_files['tools/jev/infer_audio2tool_native.py']=file_hash(__file__)
    write_once(args.output/'identity.json',dict(inputs_sha256=INPUT_SHA,budgets_sha256=BUDGET_SHA,
        versions=versions,engine=ENGINE,model=str(args.model),model_files=audit['files'],
        asset_audit_sha256=ASSET_AUDIT_SHA,capacity=capacity,source_files=source_files,
        sampling=dict(temperature=0.,max_tokens=1024,n=1,seed=17),
        duplicate_policy='execute_each_release_record_independently',parameters_frozen=True,targets_read=False))
    ledger=Ledger(args.output/'ledger');engine=None
    try:
        atomic_json(args.output/'progress.json',dict(phase='engine_load',heartbeat_unix=time.time()))
        engine=LLM(model=str(args.model),**ENGINE)
        params=SamplingParams(temperature=0.,max_tokens=1024,n=1,seed=17)
        for index,(row,budget) in enumerate(zip(rows,budgets,strict=True)):
            atomic_json(args.output/'progress.json',dict(phase='input',done=index,total=len(rows),id=row['id'],heartbeat_unix=time.time()))
            request,receipts=prepare_request(processor,row['input'],args.root,registry)
            waves=[w for w,rate in request['multi_modal_data']['audio']]
            batch=processor(text=[request['prompt']],audio=waves,sampling_rate=16000,
                            return_tensors='pt',padding=True,truncation=False)
            expected_ids=batch['input_ids'][0].tolist();frames=batch['feature_attention_mask'].sum(-1).tolist()
            if (frames!=budget['frames'] or frames!=[r['expected_frames'] for r in receipts]
                    or len(expected_ids)!=budget['input_tokens'] or not torch.isfinite(batch['input_features']).all()):
                raise ValueError('actual full audio processing differs from registered budget')
            del batch
            attempt=ledger.begin(row,request,receipts,expected_ids);begin=time.monotonic()
            atomic_json(args.output/'progress.json',dict(phase='generation',done=index,total=len(rows),id=row['id'],heartbeat_unix=time.time()))
            try:
                result=accept_generation(engine.generate([request],params,use_tqdm=False),expected_ids)
            except Exception as exc:
                ledger.failure(attempt,exc,time.monotonic()-begin);raise
            ledger.finish(attempt,result,time.monotonic()-begin)
            del request,waves
    finally:
        if engine is not None:engine.llm_engine.engine_core.shutdown(timeout=30)
    atomic_json(args.output/'progress.json',dict(phase='final_model_verification',done=len(rows),total=len(rows),heartbeat_unix=time.time()))
    verify_assets();summary=ledger.complete(rows)
    write_once(args.output/'result.json',dict(status='passed',**summary,identity_sha256=file_hash(args.output/'identity.json'),
        weight_updates=0,targets_read=False,model_files_unchanged=True,elapsed_seconds=time.monotonic()-start,
        scope='All 30733 release records independently generated by frozen audio-to-task actor. Local native protocol; scoring and external container acceptance separate.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','budgets','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('root','model','registry','asset-audit','processor-run','capacity-root','capacity-inspection'):
        p.add_argument('--'+k,type=Path)
    for k in ('capacity-inspection-hash','capacity-container-id'):p.add_argument('--'+k)
    p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    if not args.preflight_only and any(getattr(args,k.replace('-','_')) is None for k in (
            'root','model','registry','asset-audit','processor-run','capacity-root','capacity-inspection',
            'capacity-inspection-hash','capacity-container-id')):
        p.error('actual inference requires all model and externally inspected capacity arguments')
    args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except BaseException as exc:
        write_once(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    write_once(args.output/'terminal.json',dict(exit_code=0))
