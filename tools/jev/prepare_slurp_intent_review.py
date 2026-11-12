#!/usr/bin/env python3
"""Freeze a deterministic class-balanced, learning-only intent review packet.

No automatic semantic approval. Reference text is offline review evidence,
never part of the compiled actor request. No selection/test labels are read.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.task_schema import Candidate,DecisionSchema
from multimodal_jev.thinking import ThinkingSpec
from multimodal_jev.slurp_jev_inputs import REQUIREMENTS


def select_groups(groups, labels, per_class, seed):
    if type(per_class) is not int or per_class<1:raise ValueError('positive class budget required')
    if not groups or len(labels)!=len(set(labels)):raise ValueError('nonempty unique class inventory required')
    buckets=defaultdict(list)
    for gid,rows in groups.items():
        values={r['target']['native_target']['scenario']+'_'+r['target']['native_target']['action'] for r in rows}
        if len(values)!=1 or not values<=set(labels):raise ValueError('source intent conflict')
        buckets[next(iter(values))].append(gid)
    def rank(value):return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()
    selected=[];coverage={}
    for label in labels:
        pool=sorted(buckets[label],key=rank);chosen=pool[:per_class]
        coverage[label]=dict(available_groups=len(pool),selected_groups=len(chosen),shortfall=max(0,per_class-len(chosen)))
        for gid in chosen:
            row=min(groups[gid],key=lambda r:rank(r['request']['id']))
            selected.append(dict(**row,native_intent=label,review_status='pending',optimization_allowed=False))
    return selected,coverage


def run(source_receipt,ontology_path,ontology_audit,output,per_class=2,seed=317):
    source_receipt,ontology_path,ontology_audit,output=map(Path,(source_receipt,ontology_path,ontology_audit,output))
    read=lambda p:json.loads(p.read_text())
    receipt=read(source_receipt);source=Path(receipt['release']);manifest=read(source/'MANIFEST.json')
    if file_hash(source/'MANIFEST.json')!=receipt['manifest_sha256']:raise ValueError('source changed')
    audit=read(ontology_audit)
    if file_hash(ontology_path)!=audit['ontology_sha256']:raise ValueError('ontology changed')
    if audit['source_files'][str(source/'MANIFEST.json')]!=receipt['manifest_sha256']:raise ValueError('ontology source mismatch')
    bank=read(ontology_path);metadata=Path(manifest['source_root'])/'metadata/train.jsonl'
    if file_hash(metadata)!=manifest['identity']['input_hashes']['clean_metadata_sha256']:raise ValueError('metadata changed')
    for name in ('learning.requests.jsonl','learning.targets.jsonl'):
        if file_hash(source/name)!=manifest['files'][name]:raise ValueError('learning file changed')
    by_source={str(r['slurp_id']):r for r in map(json.loads,metadata.read_text().splitlines())}
    requests={r['id']:r for r in map(json.loads,(source/'learning.requests.jsonl').read_text().splitlines())}
    groups=defaultdict(list);seen=set()
    for target in map(json.loads,(source/'learning.targets.jsonl').read_text().splitlines()):
        uid=target['id'];request=requests[uid]
        if uid in seen or target['request_sha256']!=content_hash(request):raise ValueError('target binding differs')
        seen.add(uid)
        if request['role']!='learning' or request['official_split']!='train':raise ValueError('learning TRAIN only')
        if target['annotation_status']!='available' or target['optimization_allowed'] is not False:raise ValueError('pending clean targets required')
        rows=[by_source[k] for k in target['source_ids']]
        references={r['sentence'] for r in rows}
        if len(references)!=1:raise ValueError('ambiguous reference')
        for row in rows:
            if (request['id'].removeprefix('slurp:') not in {r['file'] for r in row['recordings']}
                    or row['scenario']!=target['native_target']['scenario']
                    or row['action']!=target['native_target']['action']):raise ValueError('source membership differs')
        groups[request['source_group']].append(dict(request=request,target=target,reference=next(iter(references))))
    if len(seen)!=len(requests) or len(seen)!=receipt['counts']['learning']:raise ValueError('incomplete source roster')
    intents=bank['intents'];labels=[r['id'] for r in intents]
    selected,coverage=select_groups(groups,labels,per_class,seed)
    task=dict(kind='intent',candidate_pairs={r['id']:[r['scenario'],r['action']] for r in intents},
        schema=asdict(DecisionSchema('slurp-intent-60-v1','choice',
            'Which listed scenario and action best describe the spoken request?',
            tuple(Candidate(r['id'],'Scenario: '+r['scenario']+'; action: '+r['action']+'.') for r in intents),allow_unknown=False)),
        thinking=ThinkingSpec('slurp-native-intent-v1',REQUIREMENTS).to_dict())
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'tasks.json',{'intent':task})
    atomic_json(output/'packet.json',selected)
    paths=[source_receipt,source/'MANIFEST.json',source/'learning.requests.jsonl',source/'learning.targets.jsonl',metadata,ontology_path,ontology_audit,Path(__file__).resolve()]
    report=dict(schema='slurp-intent-learning-review-packet-v1',source_manifest_sha256=receipt['manifest_sha256'],
        learning_recordings=len(seen),learning_groups=len(groups),selected_recordings=len(selected),
        selected_groups=len({r['request']['source_group'] for r in selected}),per_class=per_class,seed=seed,coverage=coverage,
        evidence_files={str(p.resolve()):file_hash(p) for p in paths},packet_sha256=file_hash(output/'packet.json'),
        tasks_sha256=file_hash(output/'tasks.json'),selection_targets_opened=False,test_opened=False,optimization_ready=False,
        scope='Fixed hash class-balanced learning source review only. No semantic approval, audio adjudication, negative type supervision or task-quality score.')
    atomic_json(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source-receipt','ontology','ontology-audit','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--per-class',type=int,default=2);p.add_argument('--seed',type=int,default=317)
    a=p.parse_args();r=run(a.source_receipt,a.ontology,a.ontology_audit,a.output,a.per_class,a.seed)
    print(json.dumps({k:r[k] for k in ('selected_recordings','selected_groups','learning_groups','optimization_ready')}))
