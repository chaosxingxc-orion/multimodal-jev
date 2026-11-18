#!/usr/bin/env python3
"""Bind complete reference definition-fit reviews without claiming I_class labels."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def content_sha(x):
    return hashlib.sha256((json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n').encode()).hexdigest()


def bind(packet, authored, definition):
    sources={r['source_id']:r for r in packet['sources']}
    if len(sources)!=len(packet['sources']):raise ValueError('duplicate evidence source')
    keys=[r['source_id'] for r in authored]
    if len(set(keys))!=len(keys) or set(keys)!=set(sources):
        raise ValueError('exact complete source review required')
    decisions=[];bindings=[]
    for r in authored:
        sid=r['source_id'];s=sources[sid]
        if r['semantic_id']!=definition['semantic_id'] or r['definition_fit'] not in ('supported','not_supported','unresolved'):
            raise ValueError('invalid definition or fit')
        if not r['rationale'].strip() or not r['scope'].strip():raise ValueError('rationale and scope required')
        value={'supported':True,'not_supported':False,'unresolved':None}[r['definition_fit']]
        member=s['native_intent']==definition['semantic_id']
        decision=dict(source_id=sid,native_intent=s['native_intent'],semantic_id=definition['semantic_id'],
            reference=s['reference'],metadata_sha256=s['metadata_sha256'],definition_sha256=content_sha(definition),
            definition_fit=r['definition_fit'],proposed_definition_fit=value,
            native_membership=member,disagrees_with_native_membership=value is not None and value!=member,
            rationale=r['rationale'],scope=r['scope'],
            exclusive_I_class_truth=None,optimization_allowed=False,audio_listening=False)
        decisions.append(decision)
        for rec in s['recordings']:
            bindings.append(dict(source_id=sid,recording_id=rec['id'],source_group=rec['source_group'],
                audio_sha256=rec['audio_sha256'],source_target_sha256=rec['target_sha256'],
                review_sha256=content_sha(decision),definition_fit=r['definition_fit'],
                complete_source_review=all(str(x) in sources for x in rec['all_source_ids']),
                optimization_allowed=False))
    return decisions,bindings


def main():
    p=argparse.ArgumentParser()
    for arg in ('packet','reviews','definitions','output'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--packet-sha256',required=True)
    p.add_argument('--semantic-id',required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.packet)!=a.packet_sha256:raise ValueError('packet identity mismatch')
    packet=json.loads(a.packet.read_text())
    pins=dict(packet['pins'])
    pins.update({str(a.packet):a.packet_sha256,str(a.reviews):sha(a.reviews),str(a.definitions):sha(a.definitions)})
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('changed dependency: '+path)
    definition=next(x for x in json.loads(a.definitions.read_text())['cards'] if x['semantic_id']==a.semantic_id)
    with a.reviews.open() as f:authored=list(csv.DictReader(f,delimiter='\t'))
    decisions,bindings=bind(packet,authored,definition)
    if len({r['recording_id'] for r in bindings})!=packet['recordings']:raise ValueError('recording coverage mismatch')
    a.output.mkdir(parents=True)
    for name,data in [('decisions.jsonl',decisions),('recording-bindings.jsonl',bindings)]:
        (a.output/name).write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in data))
    report=dict(schema='slurp-definition-fit-reference-v1',pins=pins,implementation_sha256=sha(__file__),
        source_utterances=len(decisions),recordings=packet['recordings'],
        source_fit_counts=dict(Counter(x['definition_fit'] for x in decisions)),
        recording_binding_fit_counts=dict(Counter(x['definition_fit'] for x in bindings)),
        per_native_class={k:dict(Counter(x['definition_fit'] for x in decisions if x['native_intent']==k)) for k in packet['native_intents']},
        disagrees_with_native_membership=[x['source_id'] for x in decisions if x['disagrees_with_native_membership']],
        files={n:sha(a.output/n) for n in ('decisions.jsonl','recording-bindings.jsonl')},
        semantic_supervision_admitted=False,training_dispatch_allowed=False,
        scope='Complete selected native-class learning reference cohort only; fit is not exclusive class truth; no negatives for other definitions inferred; no audio listening or native relabelling.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pins','files')},indent=2))


if __name__=='__main__':main()
