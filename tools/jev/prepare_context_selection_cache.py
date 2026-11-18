#!/usr/bin/env python3
"""Warm complete selection inputs without labels or NPU allocation."""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash


def run(c,out):
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.cached_context_preparation import CachedContextPreparer,processor_cache_identity
    from multimodal_jev.context_candidate_task import compile_candidates
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache,key_for
    from multimodal_jev.local_audio_mirror import LocalAudioMirror
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    from multimodal_jev.slurp_portable_release import load
    from multimodal_jev.slurp_training import content_hash
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    require_training_storage(out);torch.set_num_threads(1)
    for p,h in c['pins'].items():
        if file_hash(p)!=h:raise ValueError('pinned dependency changed: '+p)
    rows,task,_=load(c['release'],c['release_sha256'],'selection',purpose='evaluate')
    if len(rows)!=9226:raise ValueError('complete selection population required')
    ledger={}
    with Path(c['ledger']).open() as stream:
        for line in stream:
            record=json.loads(line)
            if record['role']=='selection':
                if record['id'] in ledger:raise ValueError('duplicate preflight record')
                ledger[record['id']]=record
    if set(ledger)!={r['id'] for r in rows}:raise ValueError('complete independent preflight required')
    rows=[{k:row[k] for k in ('id','audio','audio_sha256')} for row in rows]
    candidates=[dict(id=x['id'],definition=x['description']) for x in task['schema']['candidates']]
    if len(candidates)!=60:raise ValueError('complete candidate menu required')
    processor_sha=processor_cache_identity(c['source_policy'])
    def factory():
        processor=Qwen3ASRProcessor.from_pretrained(Path(c['source_policy'])/'processor',local_files_only=True)
        # This instance owns immutable, pinned assets and never adds tokens.
        # Return independent dictionaries so callers cannot mutate the snapshot.
        vocab=processor.tokenizer.get_vocab()
        processor.tokenizer.get_vocab=lambda:dict(vocab)
        return processor
    cache=EncodedAudioCache(c['shared_cache']);require_training_storage(c['shared_cache'])
    prepare=CachedContextPreparer(factory,cache,processor_sha256=processor_sha,
        decision_token_id=151706,audio_token_id=c['audio_token_id'],max_tokens=4096)
    def one(row):
        request=compile_candidates(row['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        prior=ledger[row['id']]
        if (prior['audio_sha256']!=row['audio_sha256'] or prior['audio']!=row['audio'] or
                prior['request_sha256']!=content_hash(request.model_input())):
            raise ValueError('input differs from independent preflight')
        value=prepare((request,row['audio_sha256']))
        if (tensor_identity(value.inputs['input_ids'])!=prior['input_ids'] or
                value.positions.tolist()!=prior['positions'] or
                content_hash(value.audio_receipt)!=prior['audio_receipt_sha256'] or
                list(value.inputs['input_features'].shape)!=prior['input_features']['shape']):
            raise ValueError('local processor differs from preflight')
        key=key_for(prepare.identity(request,row['audio_sha256']))
        return dict(id=row['id'],cache_key=key,cache_sha256=file_hash(cache.root/(key+'.pt')),
            request_sha256=prior['request_sha256'],audio_sha256=row['audio_sha256'],cache_hit=value.cache_hit,
            tensors={k:tensor_identity(v) for k,v in dict(value.inputs,positions=value.positions).items()})
    started=time.monotonic();hits=0
    with (out/'inputs.jsonl').open('x') as stream:
        with OrderedPrefetch(one,rows,workers=2,ahead=4) as prepared:
            for index,row in enumerate(prepared):
                stream.write(json.dumps(row,separators=(',',':'))+'\n');hits+=row['cache_hit']
                if (index+1)%256==0:
                    stream.flush()
                    atomic_json(out/'progress.json',dict(completed=index+1,total=len(rows),hits=hits,
                        seconds=time.monotonic()-started))
        stream.flush()
        import os
        os.fsync(stream.fileno())
    report=dict(schema='context-selection-cache-v1',role='selection',recordings=len(rows),release_sha256=c['release_sha256'],
        processor_sha256=processor_sha,request_version='jev-context-exclusive_event-v1',
        inputs_sha256=file_hash(out/'inputs.jsonl'),cache_root=str(cache.root),
        seconds=time.monotonic()-started,existing_hits=hits,
        scope='Full selection input-only CPU cache; tokens/positions/audio receipt checked against prior preflight. No predictions, optimization or NPU speedup evidence.')
    commit_json(out/'MANIFEST.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    print(json.dumps(run(json.loads(a.config.read_text()),a.output),indent=2))
