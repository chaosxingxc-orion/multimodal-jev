#!/usr/bin/env python3
"""Reconstruct fixed rule inputs and verify saved HF outputs in native vLLM."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
# Import outside __main__: multiprocessing spawn re-executes this module, and
# every vLLM engine process must register the custom Span architecture too.
from check_span_runtime_suite import run


def prepare(config,output):
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import SpanRequest,SpanQuestion,encode_span_request,span_labels
    from multimodal_jev.task_schema import DecisionSchema
    from multimodal_jev.span_request_identity import bind_span_request
    from multimodal_jev.policy_identity import checkpoint_identity
    from multimodal_jev.span_rule_population import configured_rule_rows
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('pinned source/reference changed')
    prior=Path(config['hf_run']);read=lambda p:json.loads(p.read_text())
    prior_config=read(prior/'config.json');result=read(prior/'run/result.json')
    if read(prior/'run/terminal.json')['exit_code']!=0:raise ValueError('HF run failed')
    if file_hash(prior/'run/records.jsonl')!=result['records_sha256']:raise ValueError('HF records changed')
    base_id=checkpoint_identity(config['model'])[0];adapter_id=checkpoint_identity(config['adapter'])[0]
    if result['base_id']!=base_id or result['adapter']!=adapter_id:raise ValueError('HF policy differs')
    rows=configured_rule_rows(config);records=[json.loads(s) for s in (prior/'run/records.jsonl').read_text().splitlines()]
    if [r['id'] for r in rows]!=prior_config['request_ids']:raise ValueError('fixed HF selection differs')
    processor=Qwen3ASRProcessor.from_pretrained(config['model'],local_files_only=True)
    bundles=[]
    for row,record in zip(rows,records,strict=True):
        raw=row['request'];request=SpanRequest(raw['audio'],tuple(SpanQuestion(q['id'],DecisionSchema.from_dict(q['schema']))
            for q in raw['questions']),raw['final_id'],tuple(raw['requirements']),raw['version'])
        if file_hash(request.audio)!=row['provenance']['audio_sha256']:raise ValueError('audio changed')
        audio,receipt=load_audio(request.audio,return_receipt=True)
        batch,positions=encode_span_request(processor,request,audio)
        mapping=span_labels(processor.tokenizer,request.questions)
        uid,payload=bind_span_request(request,receipt,batch,positions,[m['token_ids'] for m in mapping],base_id+':'+adapter_id)
        if row['id']!=record['id'] or uid!=record['request_id'] or payload!=record['request_identity']:
            raise ValueError('reconstructed HF input differs')
        bundles.append(dict(id=row['id'],request=raw,audio_sha256=row['provenance']['audio_sha256'],audio_processing=receipt,
            input_ids=batch['input_ids'].tolist(),positions=positions.tolist(),targets=row['targets'],supervised=row['supervised']))
    identity=output/'identity.json';reference=output/'hf-reference.json'
    atomic_json(identity,dict(bundles=bundles))
    atomic_json(reference,dict(base_id=base_id,adapter_id=adapter_id,records=records))
    suite={**config,'identity':str(identity),'hf_results':str(reference),'pins':dict(config['pins'])}
    suite['pins'].update({str(p):file_hash(p) for p in (identity,reference)})
    atomic_json(output/'suite-config.json',suite)
    return suite


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        config=json.loads(args.config.read_text())
        if config.get('physical_devices')!=[5]:raise ValueError('rule diagnostic reserved to physical NPU5')
        suite=prepare(config,args.output)
        result=run(suite,args.output)
        atomic_json(args.output/'result.json',result)
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
