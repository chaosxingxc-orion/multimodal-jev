#!/usr/bin/env python3
"""Frozen Omni NPU prefill capacity; CPU preflight never loads weights."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_audio import prepare_request
from multimodal_jev.audio2tool_engine import (ASSET_AUDIT_SHA, ENGINE, VERSIONS,
    accept_response, prior_records, validate_request)
from multimodal_jev.baseline import atomic_json,file_hash
from check_audio2tool_audio_loader import INPUT_SHA,BUDGET_SHA
from probe_audio2tool_native_processor import selections


def run(args):
    from transformers import Qwen3OmniMoeProcessor
    start=time.monotonic()
    if (file_hash(args.inputs)!=INPUT_SHA or file_hash(args.budgets)!=BUDGET_SHA
            or file_hash(args.asset_audit)!=ASSET_AUDIT_SHA):
        raise ValueError('registered inputs, budgets and original model audit required')
    versions={k:importlib.metadata.version(k) for k in VERSIONS}
    if versions!=VERSIONS:raise ValueError('fixed inference stack required')
    prior,previous=prior_records(args.processor_run)
    rows=[json.loads(l) for l in args.inputs.open()];budgets=[json.loads(l) for l in args.budgets.open()]
    chosen=selections(rows,budgets)
    if {k:r['id'] for k,(r,b) in chosen.items()}!={k:r['id'] for k,r in prior.items()}:
        raise ValueError('boundary inputs differ')
    audit=json.loads(args.asset_audit.read_text())
    if (audit['model']!=str(args.model) or audit['status']!='verified_inference_assets'
            or audit['repository']!='Qwen/Qwen3-Omni-30B-A3B-Instruct'
            or audit['revision']!='26291f793822fb6be9555850f06dfe95f2d7e695'
            or not audit['all_publisher_files_present']):
        raise ValueError('original complete local Omni assets required')
    def verify_weights():
        for entry in audit['files']:
            if file_hash(args.model/entry['file'])!=entry['sha256']:raise ValueError('model asset changed')
    # Processor assets always verified; full weights before/after actual engine.
    for name,pin in previous['processor_assets'].items():
        if file_hash(args.model/name)!=pin:raise ValueError('processor asset changed')
    if not args.preflight_only:verify_weights()
    processor=Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    registry=args.registry.read_bytes()
    source=Path(__file__).resolve().parents[2]
    source_names=['tools/jev/probe_audio2tool_engine.py','tools/jev/probe_audio2tool_native_processor.py',
        'tools/jev/check_audio2tool_audio_loader.py','src/multimodal_jev/audio2tool_engine.py',
        'src/multimodal_jev/audio2tool_audio.py','src/multimodal_jev/audio2tool_input.py',
        'src/multimodal_jev/audio2tool_dialogue_input.py','src/multimodal_jev/audio2tool_inventory.py',
        'src/multimodal_jev/sfc_audio.py','src/multimodal_jev/omni_waveform_boundary.py',
        'src/multimodal_jev/baseline.py']
    atomic_json(args.output/'identity.json',dict(versions=versions,engine=ENGINE,model=str(args.model),
        asset_audit_sha256=ASSET_AUDIT_SHA,inputs_sha256=INPUT_SHA,budgets_sha256=BUDGET_SHA,
        processor_result_sha256=file_hash(args.processor_run/'result.json'),
        source_files={name:file_hash(source/name) for name in source_names},
        sampling=dict(temperature=0.,max_tokens=1,n=1,seed=17),preflight_only=args.preflight_only,
        annotations_read=False,weight_updates=0))
    engine=None;records=[]
    try:
        if not args.preflight_only:
            from vllm import LLM,SamplingParams
            atomic_json(args.output/'progress.json',dict(phase='engine_load',heartbeat_unix=time.time()))
            engine=LLM(model=str(args.model),**ENGINE)
            params=SamplingParams(temperature=0.,max_tokens=1,n=1,seed=17)
        for name,(row,budget) in sorted(chosen.items()):
            atomic_json(args.output/'progress.json',dict(phase='input',selection=name,heartbeat_unix=time.time()))
            request,receipts=prepare_request(processor,row['input'],args.root,registry)
            validate_request(request,receipts,prior[name])
            record=dict(selection=name,id=row['id'],input_tokens=budget['input_tokens'],
                audio_count=len(receipts),input_matches_processor=True)
            if engine is not None:
                begin=time.monotonic()
                atomic_json(args.output/'progress.json',dict(phase='engine_prefill',selection=name,heartbeat_unix=time.time()))
                record.update(accept_response(engine.generate([request],params,use_tqdm=False),prior[name]))
                record['engine_seconds']=time.monotonic()-begin
            atomic_json(args.output/(name+'.json'),record);records.append(record)
            del request
    finally:
        if engine is not None:engine.llm_engine.engine_core.shutdown(timeout=30)
    if not args.preflight_only:verify_weights()
    atomic_json(args.output/'result.json',dict(status='passed',records=records,
        identity_sha256=file_hash(args.output/'identity.json'),elapsed_seconds=time.monotonic()-start,
        model_loaded=not args.preflight_only,weight_updates=0,annotations_read=False,
        scope=('CPU input preflight only: all four requests exactly match prior accepted processor inputs; no weights loaded.'
               if args.preflight_only else 'Four real NPU prefills plus one generated token each. Not full-population execution, 1024-token decoding, task-quality evaluation or paper reproduction.')))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','model','inputs','budgets','registry','asset-audit','processor-run','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0))
