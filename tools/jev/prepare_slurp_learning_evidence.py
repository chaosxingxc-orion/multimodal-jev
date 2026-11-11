#!/usr/bin/env python3
"""Prepare private, learning-only annotation evidence for semantic review.

No negative labels, model prompts, approved definitions or optimization data.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical


def prepare(root, evidence, output):
    if output.exists():raise FileExistsError('preserve evidence release')
    verified=json.loads(evidence.read_text())
    if str(root)!=verified['release'] or file_hash(root/'MANIFEST.json')!=verified['manifest_sha256']:
        raise ValueError('verified source release required')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name in ('learning.requests.jsonl','learning.targets.jsonl'):
        if file_hash(root/name)!=manifest['files'][name]:raise ValueError('learning source changed')
    requests={r['id']:r for r in map(json.loads,(root/'learning.requests.jsonl').open())}
    sources={};recordings=Counter();intents=Counter();types=defaultdict(list)
    for target in map(json.loads,(root/'learning.targets.jsonl').open()):
        req=requests.pop(target['id'])
        if req['role']!='learning' or req['official_split']!='train':raise ValueError('learning only')
        if target['request_sha256']!=hashlib.sha256(canonical(req)).hexdigest():raise ValueError('target binding')
        if target['annotation_status']!='available':raise ValueError('conflict must remain isolated')
        for uid in target['source_ids']:
            if uid in sources and sources[uid]['native_target']!=target['native_target']:
                raise ValueError('inconsistent source targets')
            sources[uid]=target
            recordings[uid]+=1
    if requests:raise ValueError('unmatched learning requests')
    # Metadata is authenticated as a whole; only explicitly admitted learning
    # IDs are used to form cards. No selection request/target file is opened.
    meta=Path(manifest['source_root'])/'metadata/train.jsonl'
    if file_hash(meta)!=manifest['identity']['input_hashes']['clean_metadata_sha256']:
        raise ValueError('metadata changed')
    seen=set()
    for row in map(json.loads,meta.open()):
        uid=str(row['slurp_id'])
        if uid not in sources:continue
        seen.add(uid);target=sources[uid]['native_target']
        entities=[{'type':e['type'],'filler':' '.join(row['tokens'][i]['surface'].lower() for i in e['span'])} for e in row['entities']]
        if target!={'scenario':row['scenario'],'action':row['action'],'entities':entities}:
            raise ValueError('metadata target mismatch')
        intent=row['scenario']+'_'+row['action'];intents[intent]+=1
        for kind in sorted({e['type'] for e in entities}):
            types[kind].append(dict(source_id=uid,recordings=recordings[uid],
                text_reference=row['sentence'],intent=intent,
                entities=entities,review_status='pending',
                scope='Annotated example for offline review, not approved positive/negative semantic supervision'))
    if seen!=set(sources):raise ValueError('missing source metadata')
    cards=[]
    for kind,rows in sorted(types.items()):
        ordered=sorted(rows,key=lambda r:hashlib.sha256(('evidence-v1:'+r['source_id']).encode()).hexdigest())
        # Pick distinct intent contexts before another example of the same intent.
        chosen=[];used_intents=set()
        for row in ordered:
            if row['intent'] not in used_intents:
                chosen.append(row);used_intents.add(row['intent'])
            if len(chosen)==6:break
        for row in ordered:
            if len(chosen)==6:break
            if row not in chosen:chosen.append(row)
        cards.append(dict(entity_type=kind,source_utterances=len(rows),
            annotated_occurrences=sum(sum(e['type']==kind for e in r['entities']) for r in rows),
            intent_support=dict(sorted(Counter(r['intent'] for r in rows).items())),
            examples=chosen,semantic_definition=None,presence_evidence=None,
            confusable_boundaries=None,absence_policy='unreviewed; absence is not a negative label',
            review_status='pending',optimization_allowed=False))
    output.mkdir(parents=True)
    path=output/'type-review-cards.json';path.write_bytes(canonical(cards));path.chmod(0o600)
    receipt=dict(schema='slurp-learning-definition-evidence-v1',source_release=str(root),
        verified_receipt_sha256=file_hash(evidence),source_manifest_sha256=file_hash(root/'MANIFEST.json'),
        metadata_sha256=file_hash(meta),learning_requests_sha256=manifest['files']['learning.requests.jsonl'],
        learning_targets_sha256=manifest['files']['learning.targets.jsonl'],
        implementation_sha256=file_hash(__file__),source_utterances=len(sources),
        learning_recordings=verified['counts']['learning'],intent_classes=len(intents),entity_types=len(cards),
        examples=sum(len(c['examples']) for c in cards),cards_sha256=file_hash(path),
        test_opened=False,selection_targets_opened=False,optimization_ready=False,
        scope='Learning-only annotation evidence and pending review cards; full TRAIN metadata authenticated then filtered by admitted learning IDs. No definitions/negative labels approved or training launched.')
    (output/'receipt.json').write_bytes(canonical(receipt))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','evidence','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.evidence,a.output),indent=2))
