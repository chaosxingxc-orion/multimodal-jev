#!/usr/bin/env python3
"""Bind explicit complete reference proposals, never synthesize omitted negatives."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256((json.dumps(value,sort_keys=True,ensure_ascii=False)+'\n').encode()).hexdigest()


def validate_record(record, source, types):
    if (record['source_id']!=source['source_id'] or
            record['metadata_sha256']!=source['metadata_sha256'] or
            record['reference_sha256']!=hashlib.sha256(source['reference'].encode()).hexdigest()):
        raise ValueError('changed reference identity')
    if (record['optimization_allowed'] is not False or record['actor_input_allowed'] is not False or
            record['independent_semantic_review'] is not False or record['modality']!='reference_text'):
        raise ValueError('reference evidence cannot upgrade admission')
    positive,negative,unavailable=record['positive'],record['negative_types'],record['unavailable']
    groups=[set(positive),set(negative),set(unavailable)]
    if (len(negative)!=len(set(negative)) or set.union(*groups)!=set(types) or
            sum(map(len,groups))!=len(types)):
        raise ValueError('explicit disjoint complete type coverage required')
    coverage=record['negative_coverage']
    if (coverage['explicitly_assessed_all55'] is not True or not coverage['basis'].strip() or
            coverage['source_summary']!=source['reference']):
        raise ValueError('explicit reference absence assessment required')
    for detail in list(positive.values())+list(unavailable.values()):
        if not detail['quote'] or detail['quote'] not in source['reference'] or not detail['rationale'].strip():
            raise ValueError('quoted reference evidence and rationale required')
    for detail in unavailable.values():
        if not detail['reason_kind'].strip():raise ValueError('ambiguity kind required')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reviews',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    read=lambda p:json.loads(Path(p).read_text())
    review=read(args.reviews);root=Path(review['workpack']);report=read(root/'report.json')
    if sha(root/'report.json')!=review['workpack_report_sha256'] or sha(root/'ontology.json')!=review['ontology_sha256']:
        raise ValueError('changed workpack')
    pins=dict(report['pins'])
    pins.update({str(root/n):h for n,h in report['files'].items()})
    pins.update({str(args.reviews):sha(args.reviews),str(root/'report.json'):sha(root/'report.json')})
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('changed dependency: '+path)
    sources=[json.loads(line) for line in (root/'sources.jsonl').read_text().splitlines()]
    ontology=read(root/'ontology.json');slots={s['entity_type']:s for s in ontology['slots']}
    policies={p['entity_type']:p for p in ontology['policies']}
    if len(slots)!=55 or set(slots)!=set(policies):raise ValueError('complete55 slot definitions/policies required')
    boundaries=ontology['slot_boundary_document']
    schedule=review['schedule'];start,stop=schedule['start'],schedule['stop']
    if (type(start) is not int or type(stop) is not int or not 0<=start<stop<=len(sources) or
            schedule['full_population_sources']!=len(sources) or
            schedule['order']!='existing_annotation_workpack_v1_hash_order'):
        raise ValueError('contiguous complete-population schedule required')
    ordered=sorted(sources,key=lambda r:hashlib.sha256(('annotation-workpack-v1:'+r['source_id']).encode()).hexdigest())
    if [r['source_id'] for r in ordered]!=[r['source_id'] for r in sources]:raise ValueError('workpack order changed')
    selected=sources[start:stop]
    if [r['source_id'] for r in review['records']]!=[r['source_id'] for r in selected]:
        raise ValueError('review does not match fixed source interval')
    decisions=[];bindings=[]
    for record,source in zip(review['records'],selected,strict=True):
        validate_record(record,source,slots)
        native={e['type'] for e in source['native_annotations']}
        for t in slots:
            value=True if t in record['positive'] else None if t in record['unavailable'] else False
            evidence=(record['positive'].get(t) or record['unavailable'].get(t) or
                      dict(rationale=record['negative_coverage']['basis'],quote=None))
            d=dict(schema='jev-complete-slot-reference-proposal-v1',source_id=source['source_id'],
                operator='S_present',definition_id=t,metadata_record_sha256=source['metadata_sha256'],
                definition_sha256=content_sha(slots[t]),policy_sha256=content_sha(policies[t]),
                boundary_document_sha256=content_sha(boundaries),reference_assessment_sha256=content_sha(record),
                semantic_proposal=dict(value=value,status='unresolved' if value is None else 'proposed',
                    evidence=evidence,evidence_modality='reference_text',independent_semantic_review=False),
                native_projection=dict(value=t in native,agreement='unassessed' if value is None else
                    'agrees' if value==(t in native) else 'disagrees',native_annotation_error_certified=False),
                admission=dict(status='pending',value=None,optimization_allowed=False),actor_input_allowed=False)
            decisions.append(d)
            for b in source['recording_bindings']:
                bindings.append(dict(b,source_id=source['source_id'],definition_id=t,
                    proposal_sha256=content_sha(d),proposed_value=value,admitted_value=None,
                    complete_multisource_admission=False,optimization_allowed=False))
    args.output.mkdir(parents=True)
    for name,items in [('proposals.jsonl',decisions),('recording-bindings.jsonl',bindings)]:
        (args.output/name).write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n' for x in items))
    summary=dict(schema='slurp-sequential-complete-slot-reference-v1',pins=pins,implementation_sha256=sha(__file__),
        schedule=schedule,source_count=len(selected),decision_count=len(decisions),recording_decision_bindings=len(bindings),
        unique_recordings=len({b['recording_id'] for b in bindings}),
        proposal_value_counts=dict(Counter('null' if d['semantic_proposal']['value'] is None else
            str(d['semantic_proposal']['value']).lower() for d in decisions)),
        native_agreement_counts=dict(Counter(d['native_projection']['agreement'] for d in decisions)),
        admitted_decisions=0,training_dispatch_allowed=False,
        files={n:sha(args.output/n) for n in ('proposals.jsonl','recording-bindings.jsonl')},
        scope='Incremental full-type reference proposal coverage within full9192-source schedule. No independent review, audio listening, whole-population completion or training admission.')
    (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('pins','files')}))


if __name__=='__main__':main()
