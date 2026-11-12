#!/usr/bin/env python3
"""Freeze source-disjoint development review candidates, without label approval.

Uniform hash selection by source, not class; no test or model predictions read.
Reference text remains offline review evidence, never an actor input.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_training import content_hash


def select_sources(requests,learning_groups,*,budget,seed):
    if type(budget)!=int or budget<1 or type(seed)!=int:raise ValueError('fixed positive source budget and seed required')
    groups=defaultdict(list);ids=set()
    for row in requests:
        if row['role']!='selection' or row['official_split']!='train':raise ValueError('selection from official TRAIN required')
        if row['source_group'] in learning_groups:raise ValueError('development source overlaps learning')
        if row['id'] in ids:raise ValueError('duplicate development recording')
        ids.add(row['id']);groups[row['source_group']].append(row)
    if budget>len(groups):raise ValueError('insufficient distinct development sources')
    order=lambda value:hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()
    return [min(groups[g],key=lambda r:order(r['id'])) for g in sorted(groups,key=order)[:budget]]


def run(source_receipt,output,budget=300,seed=941):
    source_receipt=Path(source_receipt);output=Path(output);read=lambda p:json.loads(p.read_text())
    receipt=read(source_receipt);source=Path(receipt['release']);manifest=read(source/'MANIFEST.json')
    if file_hash(source/'MANIFEST.json')!=receipt['manifest_sha256']:raise ValueError('source manifest differs')
    evidence={str(source_receipt.resolve()):file_hash(source_receipt),str(source/'MANIFEST.json'):file_hash(source/'MANIFEST.json')}
    for name in ('learning.requests.jsonl','selection.requests.jsonl','selection.targets.jsonl'):
        path=source/name
        if file_hash(path)!=manifest['files'][name]:raise ValueError('source file changed')
        evidence[str(path)]=file_hash(path)
    learning=[json.loads(s) for s in (source/'learning.requests.jsonl').read_text().splitlines()]
    requests=[json.loads(s) for s in (source/'selection.requests.jsonl').read_text().splitlines()]
    if len(learning)!=manifest['counts']['learning'] or len(requests)!=manifest['counts']['selection']:
        raise ValueError('complete source partition required')
    selected=select_sources(requests,{r['source_group'] for r in learning},budget=budget,seed=seed)
    metadata=Path(manifest['source_root'])/'metadata/train.jsonl'
    if file_hash(metadata)!=manifest['identity']['input_hashes']['clean_metadata_sha256']:raise ValueError('clean metadata changed')
    evidence[str(metadata)]=file_hash(metadata)
    by_source={str(r['slurp_id']):r for r in map(json.loads,metadata.read_text().splitlines())}
    targets={};request_map={r['id']:r for r in requests}
    for row in map(json.loads,(source/'selection.targets.jsonl').read_text().splitlines()):
        if row['id'] in targets or row['id'] not in request_map:raise ValueError('distinct selection target binding required')
        if row['request_sha256']!=content_hash(request_map[row['id']]):raise ValueError('selection target binding differs')
        targets[row['id']]=row
    if set(targets)!=set(request_map):raise ValueError('complete selection target population required')
    packet=[]
    for request in selected:
        target=targets[request['id']]
        if target['annotation_status']!='available' or target['optimization_allowed'] is not False:
            raise ValueError('unapproved source labels required')
        rows=[by_source[str(k)] for k in target['source_ids']];texts={r['sentence'] for r in rows}
        if len(texts)!=1:raise ValueError('reference conflict')
        for row in rows:
            if (row['scenario']!=target['native_target']['scenario'] or row['action']!=target['native_target']['action']
                    or request['id'].removeprefix('slurp:') not in {r['file'] for r in row['recordings']}):
                raise ValueError('reference source membership differs')
        packet.append(dict(request=request,target=target,reference=next(iter(texts)),review_status='pending',
            optimization_allowed=False,evaluation_allowed=False))
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/'packet.json',packet)
    evidence[str(Path(__file__).resolve())]=file_hash(__file__)
    counts=Counter(r['target']['native_target']['scenario']+'_'+r['target']['native_target']['action'] for r in packet)
    report=dict(schema='slurp-intent-development-review-v1',source_manifest_sha256=receipt['manifest_sha256'],
        seed=seed,recordings=len(packet),source_groups=len(packet),role='development',official_split='train',
        selection='uniform hash over selection source groups; one hash-selected recording per source; no class balancing',
        learning_source_overlap=0,observed_classes=len(counts),class_counts=dict(sorted(counts.items())),
        packet_sha256=file_hash(output/'packet.json'),evidence_files=evidence,test_opened=False,
        model_predictions_used=False,optimization_allowed=False,evaluation_allowed=False,
        scope='Pending offline reference/audio review. Source-level sample, not official validation/test distribution; no training reward or benchmark admission.')
    atomic_json(output/'report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-receipt',required=True);p.add_argument('--output',required=True)
    p.add_argument('--budget',type=int,default=300);p.add_argument('--seed',type=int,default=941)
    a=p.parse_args();print(json.dumps(run(a.source_receipt,a.output,a.budget,a.seed),indent=2))
