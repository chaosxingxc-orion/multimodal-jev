#!/usr/bin/env python3
"""Real FSC TRAIN audio processor probe; no target labels or model scoring."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(config):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.fsc_span_transfer import compile_request
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import encode_span_request,span_labels
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    for path,h in config['pins'].items():
        if file_hash(path)!=h:raise ValueError('probe input changed')
    torch.set_num_threads(2)
    processor=Qwen3ASRProcessor.from_pretrained(config['model'],local_files_only=True)
    task=json.loads(Path(config['task']).read_text());roster=json.loads(Path(config['roster']).read_text())
    if roster['role']!='engineering_train_inputs' or len(roster['records'])!=12:
        raise ValueError('fixed twelve-source TRAIN engineering roster required')
    records=[]
    for item in roster['records']:
        if file_hash(item['audio'])!=item['audio_sha256']:raise ValueError('audio identity changed')
        audio,receipt=load_audio(item['audio'],return_receipt=True);features=None
        for variant in ('native','permutation','rebind'):
            for template in ('direct','structured'):
                request,binding=compile_request(item['audio'],task,template,variant=variant,seed=317)
                batch,positions=encode_span_request(processor,request,audio)
                mapping=span_labels(processor.tokenizer,request.questions)
                length=batch['input_ids'].shape[-1]
                if length>4096:raise ValueError('full untruncated template exceeds context')
                current={k:tensor_identity(batch[k]) for k in ('input_features','input_features_mask')}
                if features is not None and features!=current:raise ValueError('configuration altered audio evidence')
                features=current
                records.append(dict(id=item['id']+'::'+template+'::'+variant,recording_id=item['id'],
                    template=template,variant=variant,request=request.model_input(),binding=binding,
                    audio_sha256=item['audio_sha256'],audio_processing=receipt,input_tokens=length,
                    positions=positions.tolist(),candidate_mapping=mapping,input_ids=tensor_identity(batch['input_ids']),**current))
    return dict(passed=True,recordings=12,requests=len(records),records=records,maximum_input_tokens=max(r['input_tokens'] for r in records),
        role='engineering_train_inputs',task_sha256=file_hash(config['task']),roster_sha256=file_hash(config['roster']),
        scope='Real FSC TRAIN input-only CPU processor check. No target labels extracted, no model weights loaded, no transfer metrics.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text())))
        atomic_json(a.output/'terminal.json',{'exit_code':0})
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
