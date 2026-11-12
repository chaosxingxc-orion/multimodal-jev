#!/usr/bin/env python3
"""Real CPU ASR processor check for native 60-intent/55-type Jev inputs.

No model weights, NPU mapping, gold labels, inference or training. Input-only
engineering sample; not a benchmark population or context fit for all audio.
"""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.span_contract import encode_span_request,span_labels
    from multimodal_jev.span_request_identity import tensor_identity
    from multimodal_jev.slurp_jev_inputs import compile_requests
    torch.set_num_threads(2)
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('pinned input changed: '+p)
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only container required')
    bank=json.loads(Path(config['ontology']).read_text());defs=json.loads(Path(config['definitions']).read_text())
    processor=Qwen3ASRProcessor.from_pretrained(config['model'],local_files_only=True)
    if config.get('context_limit')!=32768:raise ValueError('explicit context limit required')
    source=Path(config['requests']);wanted=set(config['recording_ids'])
    requests={}
    for line in source.read_text().splitlines():
        r=json.loads(line)
        if r['id'] in wanted:
            if r['id'] in requests:raise ValueError('duplicate input')
            requests[r['id']]=r
    if set(requests)!=wanted:raise ValueError('missing selected audio input')
    records=[]
    for uid in config['recording_ids']:
        raw=requests[uid]
        if raw['role']!='learning' or raw['official_split']!='train':raise ValueError('learning-only engineering inputs')
        if file_hash(raw['input']['audio'])!=raw['audio_sha256']:raise ValueError('source audio changed')
        audio,receipt=load_audio(raw['input']['audio'],return_receipt=True)
        compiled=compile_requests(raw,bank,defs)
        for entry in compiled:
            request=entry['request']
            inputs,positions=encode_span_request(processor,request,audio)
            mapping=span_labels(processor.tokenizer,request.questions)
            length=inputs['input_ids'].shape[-1]
            if length>config['context_limit']:raise ValueError('request exceeds context limit')
            # Same audio must yield exactly the same acoustic features across
            # every question bundle. The text candidates may differ.
            feature=tensor_identity(inputs['input_features'])
            mask=tensor_identity(inputs['input_features_mask'])
            audio_id=processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
            audio_positions=(inputs['input_ids'][0]==audio_id).nonzero().flatten().tolist()
            if not audio_positions or max(audio_positions)>=int(positions.min()):raise ValueError('readout precedes audio')
            records.append(dict(id=entry['id'],recording_id=uid,task=entry['task'],fold=entry['fold'],
                request=request.model_input(),audio_sha256=raw['audio_sha256'],audio_processing=receipt,
                input_tokens=length,positions=positions.tolist(),candidate_mapping=mapping,
                audio_tokens=len(audio_positions),input_features=feature,input_features_mask=mask,
                input_ids=tensor_identity(inputs['input_ids'])))
        local=records[-len(compiled):]
        if len({json.dumps(r['input_features'],sort_keys=True) for r in local})!=1 or len({json.dumps(r['input_features_mask'],sort_keys=True) for r in local})!=1:
            raise ValueError('acoustic evidence changed across task bundles')
    return dict(passed=True,recordings=len(requests),requests=len(records),intent_candidates=60,type_spans_per_recording=55,
        maximum_input_tokens=max(r['input_tokens'] for r in records),maximum_spans_per_request=max(len(r['positions'][0]) for r in records),
        all_audio_before_readouts=True,all_candidates_distinct_single_tokens=True,
        all_acoustic_features_equal_across_tasks=True,records=records,
        definitions_approved=False,intent_descriptions='Scenario/action identifiers only; semantic descriptions not yet reviewed.',
        model_loaded=False,npu_devices=[],
        scope='Actual CPU processor and audio inputs for engineering sample only. No model probabilities, full-candidate backend parity, training admission or benchmark quality.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        r=run(json.loads(a.config.read_text()),a.output);atomic_json(a.output/'result.json',r)
        atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
