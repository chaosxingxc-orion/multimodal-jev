"""Full TRAIN compact-request CPU audit; no model, labels, sampling or NPU.

Decode each admitted waveform once and actually encode both comparison inputs.
Records bind all resulting CPU tensors without storing another feature cache.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import threading
import time

from multimodal_jev.baseline import file_hash, atomic_json


def validate_sources(inputs, audio, expected_count):
    if (len(inputs)!=expected_count or len(audio)!=expected_count
            or len({r['id'] for r in inputs})!=expected_count
            or len({r['id'] for r in audio})!=expected_count):
        raise ValueError('Complete unique input and audio populations required')
    byid={r['id']:r for r in audio}
    if set(byid)!={r['id'] for r in inputs}:raise ValueError('Audio population mismatch')
    for r in inputs:
        if set(r)!={'id','audio','audio_sha256','catalog_sha256'}:
            raise ValueError('Only label-free input metadata required')
        if byid[r['id']]['audio_sha256']!=r['audio_sha256']:
            raise ValueError('Admitted audio identity mismatch')
    return byid


def validate_pair(records):
    if [r['arm'] for r in records]!=['direct','q_input']:
        raise ValueError('Both input arms in fixed order required')
    for r in records:
        count=60 if r['arm']=='direct' else 124
        positions=r['positions']
        if (r['decisions']!=count or len(positions)!=count or positions!=sorted(set(positions))
                or not positions or positions[0]<0 or positions[-1]>=r['tokens'] or not 0<r['tokens']<=4096
                or not 0<r['audio_tokens']<positions[0]):
            raise ValueError('Complete bounded readouts required')
    for key in ('input_features','input_features_mask'):
        if records[0]['tensors'][key]!=records[1]['tensors'][key]:
            raise ValueError('Comparison arms changed acoustic evidence')
    if records[0]['audio_tokens']!=records[1]['audio_tokens']:
        raise ValueError('Audio expansion differs between arms')


def run(package, output, mirror_root):
    import numpy as np
    import torch
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.audio_policy import load_audio
    from multimodal_jev.local_audio_mirror import LocalAudioMirror
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    from multimodal_jev.span_request_identity import tensor_identity
    from tools.reports.build_context_attribute_candidate import request, digest
    from tools.reports.context_attribute_compact_candidate import encode, pack, unpack

    package=Path(package);output=Path(output)
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only audit required')
    require_training_storage(output)
    pins=json.loads((package/'FILES.json').read_text())
    for name,h in pins.items():
        if file_hash(package/name)!=h:raise ValueError('Source package changed')
    inputs=[json.loads(line) for line in (package/'inputs.jsonl').read_text().splitlines()]
    metadata=[json.loads(line) for line in (package/'audio.jsonl').read_text().splitlines()]
    byid=validate_sources(inputs,metadata,40513)
    menu=json.loads((package/'catalog.json').read_text())
    if any(r['catalog_sha256']!=digest(menu) for r in inputs):raise ValueError('Catalog identity mismatch')
    if Counter(n['family'] for n in menu['nodes'])!={'scenario':18,'action':46,'final':60}:
        raise ValueError('Complete native attribute menu required')
    mirror=LocalAudioMirror(mirror_root,read_only=True)
    torch.set_num_threads(1);local=threading.local()
    def one(row):
        source=mirror.resolve(row['audio_sha256']);audio=load_audio(source)
        meta=byid[row['id']]
        if (len(audio)!=meta['samples'] or hashlib.sha256(np.ascontiguousarray(audio,dtype=np.float32).tobytes()).hexdigest()!=meta['waveform_sha256']):
            raise ValueError('Actual decoded waveform differs')
        if not hasattr(local,'processor'):
            local.processor=Qwen3ASRProcessor.from_pretrained(package/'processor',local_files_only=True)
        records=[]
        for arm in ('direct','q_input'):
            req=request(row['audio'],menu,arm=arm)
            if unpack(req.audio,pack(req)).model_input()!=req.model_input():raise ValueError('Lossy input')
            if arm=='q_input' and req.model_input()!=request(row['audio'],menu,arm='q_credit').model_input():
                raise ValueError('Q-input and Q-credit differ')
            batch,positions=encode(local.processor,req,audio,decision_token_id=151706,audio_token_id=151676,max_tokens=4096)
            records.append(dict(arm=arm,tokens=batch['input_ids'].shape[1],decisions=positions.shape[1],
                audio_tokens=int((batch['input_ids']==151676).sum()),positions=positions[0].tolist(),
                request_sha256=digest(req.model_input()),
                tensors={k:tensor_identity(v) for k,v in batch.items()}))
        validate_pair(records)
        return dict(id=row['id'],audio_sha256=row['audio_sha256'],samples=len(audio),
                    waveform_sha256=meta['waveform_sha256'],records=records)
    started=time.monotonic();count=0;hist={arm:Counter() for arm in ('direct','q_input')}
    try:
        with (output/'records.jsonl').open('x') as stream:
            with OrderedPrefetch(one,inputs,workers=2,ahead=4) as results:
                for record in results:
                    stream.write(json.dumps(record,separators=(',',':'))+'\n');count+=1
                    for r in record['records']:hist[r['arm']][r['tokens']]+=1
                    if count%256==0:
                        stream.flush();atomic_json(output/'progress.json',dict(completed=count,total=40513,seconds=time.monotonic()-started))
            stream.flush()
            import os
            os.fsync(stream.fileno())
        result=dict(passed=True,recordings=count,encoded_requests=2*count,
            histogram={k:dict(sorted(v.items())) for k,v in hist.items()},records_sha256=file_hash(output/'records.jsonl'),
            package_manifest_sha256=file_hash(package/'FILES.json'),seconds=time.monotonic()-started,
            scope='Actual complete clean TRAIN CPU inputs and lossless semantic fields. No selection, model, NPU capacity or efficacy admission.')
        commit_json(output/'result.json',result)
        commit_json(output/'terminal.json',dict(exit_code=0,completed=count,finished_unix=time.time()))
        return result
    except BaseException as exc:
        commit_json(output/'terminal.json',dict(exit_code=1,completed=count,error=repr(exc),finished_unix=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mirror-root',required=True)
    a=p.parse_args();run(a.package,a.output,a.mirror_root)
