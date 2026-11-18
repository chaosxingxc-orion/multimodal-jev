#!/usr/bin/env python3
"""Real median/long audio processor/cache parity on CPU, no model training."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def run(config, output):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache, binding
    from multimodal_jev.span_contract import encode_span_request, span_labels
    from multimodal_jev.span_request_identity import bind_span_request, tensor_identity
    from multimodal_jev.slurp_intent_structure import compile_intent
    from multimodal_jev.slurp_training import content_hash
    if list(Path('/dev').glob('davinci[0-9]*')): raise ValueError('CPU only')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest: raise ValueError('pinned source differs: '+path)
    torch.set_num_threads(2)
    processor=Qwen3ASRProcessor.from_pretrained(config['model'],local_files_only=True)
    task=json.loads(Path(config['tasks']).read_text())['intent']
    processor_hash=content_hash(config['pins'])
    cache=EncodedAudioCache(output/'cache')
    records=[]
    for asset in config['audio']:
        if file_hash(asset['audio']) != asset['sha256']: raise ValueError('audio changed')
        request=compile_intent(asset['audio'],task,'D')
        start=time.monotonic()
        wave,receipt=load_audio(asset['audio'],return_receipt=True)
        inputs,pos=encode_span_request(processor,request,wave)
        cold=time.monotonic()-start
        identity=binding(request=request,audio_sha256=asset['sha256'],processor_sha256=processor_hash,
                         preprocessing=receipt['config'])
        candidates=[m['token_ids'] for m in span_labels(processor.tokenizer,request.questions)]
        baseline=bind_span_request(request,receipt,inputs,pos,candidates,'probe-policy-v1')
        start=time.monotonic();path=cache.write(identity,inputs,pos,receipt);write_seconds=time.monotonic()-start
        reads=[]
        for _ in range(3):
            start=time.monotonic();restored=cache.read(identity);reads.append(time.monotonic()-start)
            for k in inputs:
                if not torch.equal(inputs[k],restored[0][k]): raise ValueError('cache input differs')
            if not torch.equal(pos,restored[1]) or receipt!=restored[2]:raise ValueError('cache metadata differs')
            if bind_span_request(request,restored[2],restored[0],restored[1],candidates,'probe-policy-v1')!=baseline:
                raise ValueError('same-policy request identity differs')
        changed=bind_span_request(request,restored[2],restored[0],restored[1],candidates,'probe-policy-v2')
        if changed[0]==baseline[0]:raise ValueError('policy must bind at consumption')
        # Independent re-encoding rather than only comparing save with reload.
        wave2,receipt2=load_audio(asset['audio'],return_receipt=True)
        fresh,pos2=encode_span_request(processor,request,wave2)
        if bind_span_request(request,receipt2,fresh,pos2,candidates,'probe-policy-v1')!=baseline:
            raise ValueError('fresh processor disagrees with cached request')
        records.append(dict(audio_id=asset['id'],audio_sha256=asset['sha256'],
            tokens=inputs['input_ids'].shape[-1],cold_prepare_seconds=cold,write_seconds=write_seconds,
            warm_read_seconds=reads,cache_bytes=path.stat().st_size,cache_sha256=file_hash(path),
            inputs={k:tensor_identity(v) for k,v in inputs.items()},positions=pos.tolist(),
            same_policy_request_id=baseline[0],next_policy_request_id=changed[0],
            exact_cached_and_fresh_inputs=True))
        atomic_json(output/'progress.json',dict(completed=len(records)))
    return dict(passed=True,records=records,training_dispatch_allowed=False,
        scope='Two actual audio CPU processor/cache identity cases; timings include cold start and are not training speedup, full population or NPU admission.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
