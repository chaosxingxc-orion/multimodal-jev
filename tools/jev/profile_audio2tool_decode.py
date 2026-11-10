#!/usr/bin/env python3
"""Pinned 4 native + 12 Jev natural-stop decode timing, never task scoring."""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.audio2tool_audio import prepare_request as prepare_native
from multimodal_jev.audio2tool_decode_profile import profile_case,summarize
from multimodal_jev.audio2tool_engine import ASSET_AUDIT_SHA,ENGINE,VERSIONS,prior_records,validate_request
from multimodal_jev.audio2tool_inference import read_population,write_once,require_capacity as admit_native
from multimodal_jev.audio2tool_jev_capacity import MODEL,require_capacity as admit_jev
from multimodal_jev.audio2tool_jev_engine import boundaries,prepare_request as prepare_jev,validate_boundary
from multimodal_jev.baseline import atomic_json,file_hash
from probe_audio2tool_native_processor import selections


def capacity(args,registry):
    if not args.capacity_evidence or not args.capacity_evidence_hash or file_hash(args.capacity_evidence)!=args.capacity_evidence_hash:
        raise ValueError('coordinator-pinned actual native and Jev capacity evidence required')
    proof=json.loads(args.capacity_evidence.read_text());n=proof['native'];j=proof['jev']
    native=admit_native(n['root'],Path(n['inspection']),n['inspection_sha256'],n['container_id'],args.native_processor_run)
    jev=admit_jev(j['root'],j['intent_sha256'],j['inspection'],j['inspection_sha256'],j['container_id'],
                  args.context_run,args.release,registry)
    return dict(native=native,jev=jev)


def run(args):
    start=time.monotonic()
    progress=lambda phase,**kw:atomic_json(args.output/'progress.json',dict(phase=phase,heartbeat_unix=time.time(),**kw))
    registry=args.registry.read_bytes()
    rows,budgets=read_population(args.inputs,args.budgets)
    selected=selections(rows,budgets);prior,processor_identity=prior_records(args.native_processor_run)
    _,bank,jev=boundaries(args.context_run,args.release,registry)
    cases=[]
    for name,(row,budget) in sorted(selected.items()):
        if row['id']!=prior[name]['id']:raise ValueError('original native boundaries required')
        cases.append(dict(name='native-'+name,family='native',id=row['id'],row=row,prior=prior[name]))
    for row,case in jev:
        cases.append(dict(name='jev-'+case['case'].replace(':','-'),family='jev',id=row['id'],row=row,
                          decision_id=case['decision_id'],prior=case))
    names=[c['name'] for c in cases]
    if len(names)!=16 or len(set(names))!=16:raise ValueError('complete distinct sixteen boundary cases required')
    admission=None if args.preflight_only else capacity(args,registry)
    versions={k:importlib.metadata.version(k) for k in VERSIONS}
    if versions!=VERSIONS or str(args.model)!=MODEL or file_hash(args.asset_audit)!=ASSET_AUDIT_SHA:
        raise ValueError('fixed inference stack and original model required')
    audit=json.loads(args.asset_audit.read_text())
    if audit['model']!=MODEL:raise ValueError('original full model manifest required')
    def weights():
        for entry in audit['files']:
            if file_hash(args.model/entry['file'])!=entry['sha256']:raise ValueError('model asset changed')
    for name,pin in processor_identity['processor_assets'].items():
        if file_hash(args.model/name)!=pin:raise ValueError('processor asset changed')
    if not args.preflight_only:progress('verify_weights');weights()
    import torch
    from transformers import Qwen3OmniMoeProcessor
    processor=Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    if file_hash(inspect.getfile(type(processor)))!=processor_identity['hf_processor_source_sha256']:
        raise ValueError('actual accepted HF processor required')
    source=Path(__file__).resolve().parents[2]
    sources={str(p.relative_to(source)):file_hash(p) for p in sorted((source/'src/multimodal_jev').glob('*.py'))}
    for name in ('profile_audio2tool_decode.py','probe_audio2tool_native_processor.py'):
        sources['tools/jev/'+name]=file_hash(source/'tools/jev'/name)
    write_once(args.output/'identity.json',dict(versions=versions,engine=ENGINE,model=MODEL,
        asset_audit_sha256=ASSET_AUDIT_SHA,model_files=audit['files'],sources=sources,cases=names,
        sampling=dict(temperature=0.,max_tokens=1024,n=1,seed=17),capacity=admission,
        input_pins={str(p):file_hash(p) for p in (args.inputs,args.budgets,args.registry,args.release/'MANIFEST.json',args.context_run/'result.json')},
        preflight_only=args.preflight_only,targets_read=False,weight_updates=0,
        protocol='One ordered pass: four native then twelve Jev boundary requests, no warmup or automatic replay; raw natural-stop outputs, no label-based selection.'))
    engine=None;records=[];load_seconds=None
    try:
        if not args.preflight_only:
            from vllm import LLM,SamplingParams
            progress('engine_load');begin=time.monotonic();engine=LLM(model=MODEL,**ENGINE);load_seconds=time.monotonic()-begin
            sampling=SamplingParams(temperature=0.,max_tokens=1024,n=1,seed=17)
            config=getattr(getattr(engine.llm_engine,'vllm_config',None),'model_config',None)
            mm=getattr(config,'multimodal_config',None)
            fields=('mm_processor_cache_gb','mm_processor_cache_type','mm_shm_cache_max_object_size_mb')
            write_once(args.output/'effective-cache.json',dict(
                fields={k:getattr(mm,k,None) for k in fields},configuration_exposed=mm is not None,
                note='Report effective defaults without asserting that multimodal feature/processor caching is disabled. No prefix cache is requested.'))
        else:sampling=None
        for index,case in enumerate(cases):
            progress('boundary',done=index,total=16,case=case['name'])
            def prepare():
                if case['family']=='native':
                    request,receipts=prepare_native(processor,case['row']['input'],args.root,registry)
                    validate_request(request,receipts,case['prior'])
                else:
                    request,receipts=prepare_jev(processor,case['row'],case['decision_id'],args.root,bank,registry)
                    validate_boundary(request,receipts,case['prior'])
                return request,receipts
            records.append(profile_case(processor,engine,sampling,case,prepare,args.output,lambda t:bool(torch.isfinite(t).all())))
    finally:
        if engine is not None:engine.llm_engine.engine_core.shutdown(timeout=30)
    if not args.preflight_only:progress('verify_weights_after');weights()
    summary=None if args.preflight_only else summarize(records,names)
    write_once(args.output/'result.json',dict(status='preflight_passed' if args.preflight_only else 'profile_complete',
        records=records,summary=summary,engine_load_seconds=load_seconds,elapsed_seconds=time.monotonic()-start,
        identity_sha256=file_hash(args.output/'identity.json'),model_loaded=not args.preflight_only,targets_read=False,weight_updates=0,
        scope='Sixteen registered boundary inputs with actual full HF processing; CPU preflight has no model. Actual profile uses max_tokens1024 with natural EOS, not a population benchmark or guaranteed 1024-token stress test. No task scoring or training.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','model','inputs','budgets','registry','asset-audit','release','context-run','native-processor-run','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--capacity-evidence',type=Path);p.add_argument('--capacity-evidence-hash')
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    try:run(a)
    except BaseException as exc:
        write_once(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    write_once(a.output/'terminal.json',dict(exit_code=0))
