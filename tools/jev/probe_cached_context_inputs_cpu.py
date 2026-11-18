#!/usr/bin/env python3
"""Real processor cache correctness/cold-hot service for context learning256."""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash


def run(c,out):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.context_candidate_task import compile_candidates
    from multimodal_jev.context_decision_contract import encode_context_request
    from multimodal_jev.context_policy_artifacts import read_execution_bundle
    from multimodal_jev.cached_context_preparation import CachedContextPreparer,processor_cache_identity
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    require_training_storage(out);torch.set_num_threads(1)
    for p,h in c['pins'].items():
        if file_hash(p)!=h:raise ValueError('dependency changed '+p)
    bundle=read_execution_bundle(c['source_policy'],expected_policy_id=c['source_policy_id'],
        expected_contract=c['source_contract'],expected_base_id=c['base_id'])
    factory=lambda:Qwen3ASRProcessor.from_pretrained(bundle['processor'],local_files_only=True)
    processor=factory();task=json.loads(Path(c['tasks']).read_text())['intent']
    candidates=[dict(id=x['id'],definition=x['description']) for x in task['schema']['candidates']]
    kwargs=dict(decision_token_id=bundle['contract']['token']['id'],audio_token_id=c['audio_token_id'],max_tokens=4096)
    processor_sha=processor_cache_identity(c['source_policy'])
    prepare=CachedContextPreparer(factory,EncodedAudioCache(out/'cache'),processor_sha256=processor_sha,**kwargs)
    jobs=[(compile_candidates(a['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event'),a['sha256']) for a in c['audio']]
    if len(jobs)!=256:raise ValueError('full learning256 engineering population required')
    expected=[];records=[]
    for request,digest in jobs:
        if file_hash(request.audio)!=digest:raise ValueError('source changed')
        inputs,positions=encode_context_request(processor,request,load_audio(request.audio),**kwargs)
        inputs={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in inputs.items()}
        expected.append(dict(positions=tensor_identity(positions),**{k:tensor_identity(v) for k,v in inputs.items()}))
    for mode in ('cold','hot','prefetch'):
        started=time.monotonic();index=0;hits=0
        def batch(group):return [prepare(job) for job in group]
        groups=[jobs[:128],jobs[128:]]
        service=OrderedPrefetch(batch,groups,workers=1,ahead=1) if mode=='prefetch' else map(batch,groups)
        try:
            for group in service:
                for value in group:
                    identity=dict(positions=tensor_identity(value.positions),**{k:tensor_identity(v) for k,v in value.inputs.items()})
                    if identity!=expected[index]:raise ValueError('cached context differs from fresh independent processor')
                    hits+=value.cache_hit;index+=1
        finally:
            if mode=='prefetch':service.close()
        if index!=256 or hits!=(0 if mode=='cold' else 256):raise ValueError('complete cache cold/hit population required')
        records.append(dict(mode=mode,seconds=time.monotonic()-started,queries=index,hits=hits))
        atomic_json(out/'progress.json',dict(completed_mode=mode,records=records))
    atomic_json(out/'result.json',dict(passed=True,records=records,inputs=expected,
        scope='Real CPU processor/cache bitwise identity; service including hashing, not NPU-overlap throughput.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    run(json.loads(a.config.read_text()),a.output)
