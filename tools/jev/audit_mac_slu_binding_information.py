#!/usr/bin/env python3
"""Native MAC-SLU relation structure; no absent-edge semantic negatives."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def projection(target):
    intents=set(); roles=set(); edges=set(); instances=[]
    for frame in target['frames']:
        d,i=frame['domain'],frame['intent']
        intents.add((d,i)); instances.append((d,i))
        for s in frame['slots']:
            roles.add((d,s['name'])); edges.add((d,i,s['name']))
    product={(d,i,s) for d,i in intents for dd,s in roles if dd==d}
    return dict(intents=sorted(intents), roles=sorted(roles), edges=sorted(edges),
        unannotated_product_edges=sorted(product-edges),
        repeated_intent_instances=len(instances)-len(set(instances)))


def audit(targets):
    if len({r['id'] for r in targets})!=len(targets):raise ValueError('duplicate source IDs')
    cells=defaultdict(list); counts=Counter(); records=[]
    for target in targets:
        p=projection(target); clean=not target['unresolved']
        counts['records']+=1; counts['records_without_empty_containers']+=clean
        counts['distinct_intent_multiple_records']+=len(p['intents'])>1
        counts['explicit_edges']+=len(p['edges'])
        counts['unannotated_product_edges']+=len(p['unannotated_product_edges'])
        counts['records_with_unannotated_product_edges']+=bool(p['unannotated_product_edges'])
        counts['no_empty_records_with_unannotated_product_edges']+=clean and bool(p['unannotated_product_edges'])
        counts['repeated_intent_instance_records']+=bool(p['repeated_intent_instances'])
        if clean:
            key=(tuple(p['intents']),tuple(p['roles']))
            cells[key].append((target['id'],tuple(p['edges'])))
        records.append(dict(id=target['id'],**p,unresolved=target['unresolved'],
            absent_edge_status='unavailable_not_negative',optimization_allowed=False))
    collisions=[]
    for (intents,roles),group in sorted(cells.items()):
        variants=defaultdict(list)
        for uid,edges in group:variants[edges].append(uid)
        if len(variants)>1:
            collisions.append(dict(intents=intents,roles=roles,
                variants=[dict(edges=k,ids=v) for k,v in sorted(variants.items())]))
    counts['no_empty_IS_signatures']=len(cells)
    counts['IS_signatures_with_multiple_bindings']=len(collisions)
    counts['records_in_binding_collision_signatures']=sum(len(v['ids']) for c in collisions for v in c['variants'])
    return dict(counts=dict(counts),records=records,collisions=collisions)


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.release/'MANIFEST.json').read_text())
    pins={str(a.release/'MANIFEST.json'):sha(a.release/'MANIFEST.json')}
    for name in ('train.targets.jsonl','train.inputs.jsonl','bank.json'):
        pins[str(a.release/name)]=manifest['files'][name]
        if sha(a.release/name)!=manifest['files'][name]:raise ValueError('release changed')
    targets=[json.loads(l) for l in (a.release/'train.targets.jsonl').read_text().splitlines()]
    inputs={r['id']:r for r in map(json.loads,(a.release/'train.inputs.jsonl').read_text().splitlines())}
    if len(targets)!=manifest['records'] or {r['id'] for r in targets}!=set(inputs):raise ValueError('population differs')
    if any(r['split']!='train' for r in inputs.values()):raise ValueError('TRAIN only')
    result=audit(targets);a.output.mkdir(parents=True)
    for row in result['records']:row['audio_present']=inputs[row['id']]['audio'] is not None
    (a.output/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in result.pop('records')))
    (a.output/'collisions.json').write_text(json.dumps(result.pop('collisions'),ensure_ascii=False,indent=2)+'\n')
    result.update(schema='mac-slu-binding-information-v1',pins=pins,implementation_sha256=sha(__file__),
        training_dispatch_allowed=False,semantic_supervision_admitted=False,
        scope='All TRAIN native frames; empty-container-free is structural only, not semantic completeness. No test or audio listening.',
        files={n:sha(a.output/n) for n in ('records.jsonl','collisions.json')})
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['counts'],indent=2))


if __name__=='__main__':main()
