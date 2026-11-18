"""Independently reconstruct full audit population, templates and readout geometry.

Consumes collected terminal artifacts. Does not infer model quality or compare
uncollected feature tensors; tensor digests are processor execution receipts.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tools.reports.build_context_attribute_candidate import request,digest


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inspect_records(records, inputs, metadata, menu, anchors, *, expected_count):
    if len(inputs)!=expected_count or len({r['id'] for r in inputs})!=expected_count:
        raise ValueError('Complete unique source population required')
    byid={r['id']:r for r in metadata}
    if len(metadata)!=expected_count or len(byid)!=expected_count or set(byid)!={r['id'] for r in inputs}:
        raise ValueError('Complete audio metadata required')
    anchor_map={}
    for arm in ('direct','q_input'):
        rows=[r for r in anchors if r['arm']==arm]
        if len(rows)!=3:raise ValueError('Three actual processor anchors required')
        offsets=[[p-r['audio_tokens'] for p in r['positions'][0]] for r in rows]
        constants={r['tokens']-r['audio_tokens'] for r in rows}
        if len(constants)!=1 or any(x!=offsets[0] for x in offsets):raise ValueError('Anchor layout differs')
        anchor_map[arm]=(next(iter(constants)),offsets[0])
    hist={arm:Counter() for arm in anchor_map};count=0
    for source,record in zip(inputs,records,strict=True):
        meta=byid[source['id']]
        if (record['id']!=source['id'] or record['audio_sha256']!=source['audio_sha256']
                or record['samples']!=meta['samples'] or record['waveform_sha256']!=meta['waveform_sha256']):
            raise ValueError('Actual population or waveform differs')
        arms=record['records']
        if [r['arm'] for r in arms]!=['direct','q_input']:raise ValueError('Missing or reordered input arms')
        for r in arms:
            req=request(source['audio'],menu,arm=r['arm'])
            constant,offsets=anchor_map[r['arm']]
            if (r['request_sha256']!=digest(req.model_input()) or r['decisions']!=len(req.questions)
                    or r['positions']!=[p+r['audio_tokens'] for p in offsets]
                    or r['tokens']!=constant+r['audio_tokens'] or not 0<r['tokens']<=4096):
                raise ValueError('Request identity, geometry or context differs')
            # Qwen3-ASR chunks into 100-frame windows; each has three stride-2 reductions.
            frames=(meta['samples']+159)//160
            blocks,rest=divmod(frames,100)
            expected_audio=blocks*13+(rest+7)//8
            if r['audio_tokens']!=expected_audio:raise ValueError('Audio expansion differs from full waveform')
            tensors=r['tensors']
            if set(tensors)!={'input_ids','attention_mask','input_features','input_features_mask'}:
                raise ValueError('Unexpected processor tensors')
            if tensors['input_ids']['shape']!=[1,r['tokens']] or tensors['attention_mask']['shape']!=[1,r['tokens']]:
                raise ValueError('Token tensor shape differs')
            for value in tensors.values():
                h=value['sha256']
                if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('Invalid tensor digest')
            hist[r['arm']][r['tokens']]+=1
        for k in ('input_features','input_features_mask'):
            if arms[0]['tensors'][k]!=arms[1]['tensors'][k]:raise ValueError('Arms have different audio features')
        count+=1
    if count!=expected_count:raise ValueError('Incomplete records')
    return dict(recordings=count,requests=count*2,histogram={k:{str(n):v for n,v in sorted(h.items())} for k,h in hist.items()})


def verify(collection, package, transport_path, anchor_path):
    collection=Path(collection);package=Path(package)
    t=json.loads(Path(transport_path).read_text())
    if (t['state']['Status']!='exited' or t['state']['ExitCode']!=0 or t['state']['OOMKilled']
            or t['terminal']['exit_code']!=0 or t['terminal']['completed']!=40513
            or any(t['counts'][k] for k in ('failures','errors','skipped'))):
        raise ValueError('Complete successful fixed-image CPU execution required')
    for name,h in t['files'].items():
        if sha(collection/name)!=h:raise ValueError('Collected file changed')
    pins=json.loads((collection/'package/FILES.json').read_text())
    if sha(package/'FILES.json')!=t['source_manifest_sha256']:raise ValueError('Local package identity differs')
    for name,h in pins.items():
        if sha(package/name)!=h:raise ValueError('Source package changed')
    anchor_proof=json.loads(Path(anchor_path).read_text());anchor_root=Path(anchor_proof['collection'])
    for name,h in anchor_proof['files'].items():
        if sha(anchor_root/name)!=h:raise ValueError('Anchor collection changed')
    anchor_pins=json.loads((anchor_root/'package/FILES.json').read_text())
    for name in ('catalog.json',*[n for n in pins if n.startswith('processor/')]):
        if pins[name]!=anchor_pins[name]:raise ValueError('Different catalog or processor')
    if t['image']!=anchor_proof['image']:raise ValueError('Different runtime image')
    candidate=json.loads((package/'candidate-manifest.json').read_text())
    source=Path(candidate['release'])
    if sha(source/'MANIFEST.json')!=candidate['release_sha256']:
        raise ValueError('Original clean release identity differs')
    source_manifest=json.loads((source/'MANIFEST.json').read_text())
    if sha(package/'audio.jsonl')!=source_manifest['files']['learning.audio.jsonl']:
        raise ValueError('Audio population is not the frozen TRAIN population')
    for name in ('inputs.jsonl','catalog.json'):
        if sha(package/name)!=candidate['files'][name]:raise ValueError('Candidate input binding differs')
    inputs=[json.loads(s) for s in (package/'inputs.jsonl').read_text().splitlines()]
    metadata=[json.loads(s) for s in (package/'audio.jsonl').read_text().splitlines()]
    menu=json.loads((package/'catalog.json').read_text())
    anchors=json.loads((anchor_root/'run/result.json').read_text())['records']
    with (collection/'run/records.jsonl').open() as f:
        result=inspect_records(map(json.loads,f),inputs,metadata,menu,anchors,expected_count=40513)
    observed=json.loads((collection/'run/result.json').read_text())
    if (observed['recordings']!=result['recordings'] or observed['encoded_requests']!=result['requests']
            or observed['histogram']!=result['histogram'] or observed['records_sha256']!=sha(collection/'run/records.jsonl')
            or observed['package_manifest_sha256']!=t['source_manifest_sha256']):
        raise ValueError('Summary differs from full independent records')
    return dict(passed=True,**result,container_id=t['container_id'],image=t['image'],
                transport_sha256=sha(transport_path),records_sha256=observed['records_sha256'],
                scope='Full TRAIN population, request identities, all readout positions, context and audio expansion independently reconstructed. Feature hashes bind execution receipts; no independent full processor recomputation, model/NPU or efficacy proof.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection',required=True);p.add_argument('--package',required=True)
    p.add_argument('--transport',required=True);p.add_argument('--anchors',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=verify(a.collection,a.package,a.transport,a.anchors)
    with Path(a.output).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
