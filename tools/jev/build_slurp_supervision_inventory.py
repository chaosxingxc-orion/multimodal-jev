#!/usr/bin/env python3
"""Inventory evidence on every learning recording, without changing gold/admission."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(map(json.loads,Path(p).read_text().splitlines()))
def content_sha(v):return hashlib.sha256((json.dumps(v,sort_keys=True,ensure_ascii=False)+'\n').encode()).hexdigest()


def classify(decision,kind):
    if kind=='slot':
        return 'slot_'+decision['status']
    fit=decision['definition_fit']
    if decision['native_membership']:
        return {'supported':'native_member_definition_supported',
                'not_supported':'native_member_definition_not_supported',
                'unresolved':'native_member_definition_unresolved'}[fit]
    return {'supported':'other_native_class_definition_supported',
            'not_supported':'other_native_class_definition_not_supported',
            'unresolved':'other_native_class_definition_unresolved'}[fit]


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence-config',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.release/'MANIFEST.json')!=a.manifest_sha256:raise ValueError('release identity mismatch')
    manifest=read(a.release/'MANIFEST.json');spec=read(a.evidence_config)
    pins={str(a.release/'MANIFEST.json'):a.manifest_sha256,str(a.release/'learning.jsonl'):manifest['files']['learning.jsonl'],str(a.evidence_config):sha(a.evidence_config)}
    if sha(a.release/'learning.jsonl')!=manifest['files']['learning.jsonl']:raise ValueError('population changed')
    population=rows(a.release/'learning.jsonl');index={r['id']:r for r in population}
    if len(index)!=len(population) or len(index)!=40513:raise ValueError('full unique population required')
    by_recording=defaultdict(list);decisions_by_family=defaultdict(dict);seen=set();review_counts={}
    for source in spec['sources']:
        rp=Path(source['report']);root=rp.parent
        if sha(rp)!=source['sha256']:raise ValueError('evidence identity mismatch')
        report=read(rp);pins[str(rp)]=source['sha256']
        for path,h in report['pins'].items():
            if sha(path)!=h:raise ValueError('changed review dependency '+path)
            pins[path]=h
        for name,h in report['files'].items():
            if sha(root/name)!=h:raise ValueError('changed evidence file')
            pins[str(root/name)]=h
        decisions=rows(root/'decisions.jsonl');bindings=rows(root/'recording-bindings.jsonl')
        lookup={content_sha(d):d for d in decisions}
        if len(lookup)!=len(decisions):raise ValueError('duplicate decision')
        kind=source['kind'];family='S' if kind=='slot' else 'I_fit'
        for d in decisions:
            definition=d['entity_type'] if kind=='slot' else d['semantic_id']
            key=(d['source_id'],definition)
            if key in decisions_by_family[family]:raise ValueError('duplicate or divergent reference decision')
            if d['optimization_allowed'] is not False:raise ValueError('unexpected admission; inventory is reference-only')
            decisions_by_family[family][key]=classify(d,kind)
        for b in bindings:
            d=lookup[b['review_sha256']];uid=b['recording_id'];r=index[uid]
            if d['source_id']!=b['source_id'] or any(b[k]!=r[k] for k in ('audio_sha256','source_group','source_target_sha256')):
                raise ValueError('recording identity mismatch')
            definition=d['entity_type'] if kind=='slot' else d['semantic_id']
            identity=(family,b['source_id'],definition,uid)
            if identity in seen:raise ValueError('duplicate binding')
            seen.add(identity)
            by_recording[uid].append(dict(family=family,definition_id=definition,source_id=b['source_id'],
                evidence_status=classify(d,kind),review_sha256=b['review_sha256'],
                complete_source_review=b['complete_source_review'],optimization_allowed=False))
        review_counts[source['id']]=dict(decisions=len(decisions),bindings=len(bindings))
    # Explicitly count but do not automatically exclude. Shared group influence is reported separately.
    risk_groups={index[uid]['source_group'] for uid,ev in by_recording.items()
                 if any(e['evidence_status']=='native_member_definition_not_supported' for e in ev)}
    records=[]
    for r in population:
        evidence=by_recording[r['id']]
        records.append(dict(recording_id=r['id'],source_group=r['source_group'],audio_sha256=r['audio_sha256'],
            native_final_label=r['intent'],native_label_basis='official_clean_v7_unchanged',
            reference_evidence=evidence,
            own_native_definition_not_supported=any(e['evidence_status']=='native_member_definition_not_supported' for e in evidence),
            source_group_has_native_definition_issue=r['source_group'] in risk_groups,
            new_semantic_training_targets_admitted=0,
            disposition='inventory_only_not_training_manifest'))
    a.output.mkdir(parents=True)
    (a.output/'recordings.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    summary={}
    for family,mapping in decisions_by_family.items():
        relevant=[r for r in records if any(e['family']==family for e in r['reference_evidence'])]
        summary[family]=dict(reference_decisions=len(mapping),reference_sources=len({k[0] for k in mapping}),
            definitions=len({k[1] for k in mapping}),recordings_with_any_evidence=len(relevant),
            recordings_without_any_evidence=len(records)-len(relevant),
            reference_status_counts=dict(Counter(mapping.values())),
            recording_decision_bindings=sum(e['family']==family for r in relevant for e in r['reference_evidence']))
    issue=[r for r in records if r['own_native_definition_not_supported']]
    report=dict(schema='slurp-supervision-inventory-v1',pins=pins,implementation_sha256=sha(__file__),
        population_recordings=len(records),population_source_groups=len({r['source_group'] for r in records}),
        evidence_sources=review_counts,families=summary,
        any_reference_recordings=sum(bool(r['reference_evidence']) for r in records),
        native_definition_issue_recordings=len(issue),native_definition_issue_groups=len(risk_groups),
        issue_group_recordings=sum(r['source_group_has_native_definition_issue'] for r in records),
        issue_native_classes=dict(Counter(r['native_final_label'] for r in issue)),
        semantic_I_class_admitted_decisions=0,semantic_S_present_admitted_decisions=0,
        records_sha256=sha(a.output/'recordings.jsonl'),training_dispatch_allowed=False,
        scope='Full native learning population evidence inventory; selected new-card reference reviews only. Legacy native pilot reviews are not new-card semantic admission. No exclusions, native relabelling, source propagation of truth or training permissions.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='pins'},indent=2))


if __name__=='__main__':main()
