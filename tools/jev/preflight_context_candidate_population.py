#!/usr/bin/env python3
"""Actual audio/processor preflight for the complete authenticated population.

Workers receive input-only metadata. No labels, model weights or optimization;
all hot ledgers stay on admitted local storage. Any failed row fails the run.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def initialize(config, candidates, requirements):
    import torch
    from transformers import Qwen3ASRProcessor
    global PROCESSOR, CANDIDATES, REQUIREMENTS, CONFIG
    torch.set_num_threads(1)
    CONFIG, CANDIDATES, REQUIREMENTS = config, candidates, requirements
    PROCESSOR = Qwen3ASRProcessor.from_pretrained(config['processor'], local_files_only=True)


def inspect(job):
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.context_candidate_task import compile_candidates
    from multimodal_jev.context_decision_contract import encode_context_request
    from multimodal_jev.span_request_identity import tensor_identity
    from multimodal_jev.slurp_training import content_hash
    if file_hash(job['audio']) != job['audio_sha256']:
        raise ValueError('audio identity changed: '+job['id'])
    request=compile_candidates(job['audio'],CANDIDATES,REQUIREMENTS,semantics='exclusive_event')
    wave,receipt=load_audio(job['audio'],return_receipt=True)
    inputs,positions=encode_context_request(PROCESSOR,request,wave,
        decision_token_id=CONFIG['decision_token_id'],audio_token_id=CONFIG['audio_token_id'],max_tokens=32768)
    length=inputs['input_ids'].shape[-1]
    if positions.shape!=(1,60):raise ValueError('full 60 markers required')
    return dict(**job,request_sha256=content_hash(request.model_input()),tokens=length,
        positions=positions.tolist(),input_ids=tensor_identity(inputs['input_ids']),
        input_features=tensor_identity(inputs['input_features']),audio_receipt_sha256=content_hash(receipt),
        fits_context=length<=CONFIG['context_limit'])


def run(config,output):
    from multimodal_jev.slurp_official_training import load
    from multimodal_jev.local_runtime_storage import check_paths
    from multimodal_jev.slurp_training import content_hash
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    check_paths({'preflight_output':output})
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('changed dependency '+p)
    jobs=[];population={};common_task=None
    for role,purpose in [('learning','optimize'),('selection','evaluate')]:
        rows,task,_=load(config['release'],config['release_sha256'],role,purpose=purpose)
        if common_task is not None and task!=common_task:raise ValueError('task differs across roles')
        common_task=task
        population[role]=dict(recordings=len(rows),source_groups=len({r['source_group'] for r in rows}))
        jobs.extend({k:row[k] for k in ('id','audio','audio_sha256','role','source_group')} for row in rows)
    if [population[r]['recordings'] for r in ('learning','selection')]!=[40513,9226]:
        raise ValueError('complete frozen population required')
    candidates=[dict(id=c['id'],definition=c['description']) for c in common_task['schema']['candidates']]
    if len(candidates)!=60:raise ValueError('full ontology required')
    summary={role:dict(count=0,min_tokens=None,max_tokens=0,overflow=0,total_tokens=0) for role in population}
    start=time.monotonic();ledger=output/'inputs.jsonl'
    with ledger.open('x') as stream, ProcessPoolExecutor(max_workers=4,
            mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
            initargs=(config,candidates,common_task['thinking']['requirements'])) as pool:
        # Executor map submits small input metadata, never waveform tensors.
        for index,row in enumerate(pool.map(inspect,jobs,chunksize=32),1):
            stream.write(json.dumps(row,separators=(',',':'))+'\n')
            r=summary[row['role']];r['count']+=1;r['total_tokens']+=row['tokens']
            r['min_tokens']=row['tokens'] if r['min_tokens'] is None else min(r['min_tokens'],row['tokens'])
            r['max_tokens']=max(r['max_tokens'],row['tokens']);r['overflow']+=not row['fits_context']
            if index%256==0:
                stream.flush()
                atomic_json(output/'progress.json',dict(completed=index,total=len(jobs),seconds=time.monotonic()-start,summary=summary))
        stream.flush();os.fsync(stream.fileno())
    if any(summary[role]['count']!=population[role]['recordings'] for role in population):
        raise ValueError('incomplete processed population')
    return dict(population=population,summary=summary,ledger_sha256=file_hash(ledger),
        input_population_sha256=content_hash(jobs),seconds=time.monotonic()-start,
        all_fit=all(r['overflow']==0 for r in summary.values()),training_dispatch_allowed=False,
        scope='Full actual audio CPU processor/marker preflight for frozen test-only tokenizer. '
              'No model forward, labels in input, NPU capacity, policy learning or test optimization.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
