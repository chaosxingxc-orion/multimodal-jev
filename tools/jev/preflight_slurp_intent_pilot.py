#!/usr/bin/env python3
"""Full admitted D/Q/P learning population through the real CPU ASR processor."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.slurp_intent_pilot import load_pilot
    from multimodal_jev.span_contract import encode_span_request,span_labels
    from multimodal_jev.span_request_identity import tensor_identity
    torch.set_num_threads(2)
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('pinned asset changed: '+path)
    if config['context_limit']!=4096:raise ValueError('fixed intended engine context required')
    processor=Qwen3ASRProcessor.from_pretrained(config['model'],local_files_only=True)
    records=[];admission={};per_audio={};request_hashes={}
    for arm in ('D','Q','P'):
        items,admission[arm]=load_pilot(config['pilot'],config['manifest_sha256'],arm)
        for item in items:
            # Q/P compiled inputs are identical; check identity before reusing encoding.
            if arm=='P':
                from multimodal_jev.slurp_training import content_hash
                if content_hash(item['request'].model_input())!=request_hashes[item['id']]:raise ValueError('Q/P inputs differ')
                continue
            request=item['request'];audio,receipt=load_audio(request.audio,return_receipt=True)
            inputs,positions=encode_span_request(processor,request,audio)
            mapping=span_labels(processor.tokenizer,request.questions)
            length=inputs['input_ids'].shape[-1]
            if length>config['context_limit']:raise ValueError('untruncated input exceeds engine context')
            feature={k:tensor_identity(inputs[k]) for k in ('input_features','input_features_mask')}
            if item['id'] in per_audio and feature!=per_audio[item['id']]:raise ValueError('acoustic evidence differs across arms')
            per_audio[item['id']]=feature
            from multimodal_jev.slurp_training import content_hash
            if arm=='Q':request_hashes[item['id']]=content_hash(request.model_input())
            # No labels/weights/reference text in the input-only engine artifact.
            records.append(dict(id=item['id']+'::'+arm,recording_id=item['id'],arm=arm,
                request=request.model_input(),audio_sha256=item['audio_sha256'],audio_processing=receipt,
                input_tokens=length,positions=positions.tolist(),candidate_mapping=mapping,
                input_ids=tensor_identity(inputs['input_ids']),**feature))
            atomic_json(output/'progress.json',dict(completed=len(records),total=178,updated_unix=time.time()))
    return dict(passed=True,records=records,admission=admission,recordings=len(per_audio),requests=len(records),
        Q_P_identical=True,maximum_input_tokens=max(r['input_tokens'] for r in records),
        candidate_widths={a:[len(m['values']) for m in next(r for r in records if r['arm']==a)['candidate_mapping']] for a in ('D','Q')},
        model_loaded=False,scope='Real CPU processor on all admitted learning audio; Q/P input equality. No weights, backend probabilities, training update or task efficacy.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    try:
        result=run(json.loads(args.config.read_text()),args.output);atomic_json(args.output/'result.json',result)
        atomic_json(args.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
