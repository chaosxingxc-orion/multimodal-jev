#!/usr/bin/env python3
"""Real processor budget for all 60 SLURP event definitions and numeric markers."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.context_candidate_task import compile_candidates
    from multimodal_jev.context_decision_contract import encode_context_request
    from multimodal_jev.span_request_identity import tensor_identity
    from multimodal_jev.slurp_training import content_hash
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('changed dependency '+p)
    torch.set_num_threads(2)
    processor=Qwen3ASRProcessor.from_pretrained(config['processor'],local_files_only=True)
    task=json.loads(Path(config['tasks']).read_text())['intent']
    candidates=[dict(id=c['id'],definition=c['description']) for c in task['schema']['candidates']]
    if len(candidates)!=60:raise ValueError('full original ontology required')
    model_config=json.loads(Path(config['model']+'/config.json').read_text())
    records=[]
    for asset in config['audio']:
        if file_hash(asset['audio'])!=asset['sha256']:raise ValueError('audio changed')
        wave,receipt=load_audio(asset['audio'],return_receipt=True)
        request=compile_candidates(asset['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        # Diagnostic ceiling only; record overflow of the existing 4096 runtime
        # instead of silently truncating or treating this as capacity admission.
        batch,positions=encode_context_request(processor,request,wave,decision_token_id=config['decision_token_id'],
            audio_token_id=model_config['audio_token_id'],max_tokens=32768)
        records.append(dict(audio_id=asset['id'],audio_sha256=asset['sha256'],request=request.model_input(),
            request_sha256=content_hash(request.model_input()),tokens=batch['input_ids'].shape[-1],
            marker_count=positions.shape[-1],positions=positions.tolist(),
            input_ids=tensor_identity(batch['input_ids']),audio_receipt=receipt,
            fits_existing_4096=batch['input_ids'].shape[-1]<=4096))
        atomic_json(output/'progress.json',dict(completed=len(records)))
    return dict(records=records,training_dispatch_allowed=False,
        scope='Actual CPU processor for full 60-event definition input and sigmoid marker protocol on two audio lengths; '
              'no gold in input, no model/task predictions, full-population context or NPU capacity claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
