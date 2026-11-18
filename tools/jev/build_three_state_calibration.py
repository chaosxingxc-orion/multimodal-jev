#!/usr/bin/env python3
"""Freeze a balanced diagnostic from current author references, never gold admission."""
from collections import defaultdict, Counter
import argparse
import json
from pathlib import Path

from build_blind_semantic_annotation import render
from consolidate_intent_presence_references import obj, read, rows, require, sha, unique


def reference_coordinates(intent, slot):
    grouped = defaultdict(list)
    for a in intent:
        if a['incompatible_fields']:
            continue
        d = a['decision']
        require(not a['optimization_allowed'] and not d['optimization_allowed'], 'reference-only inputs required')
        grouped['I_present', d['source_id'], d['semantic_id']].append(
            (d['status'], d['reference'], obj(a)))
    for a in slot:
        if a['applicability'] != 'current_semantics_reference_only':
            continue
        require(not a['optimization_allowed'], 'reference-only inputs required')
        value = a['proposed_value']
        require(value is None or type(value) is bool, 'invalid reference value')
        status = 'unavailable' if value is None else ('positive' if value else 'negative')
        grouped['S_present', a['source_id'], a['definition_id']].append(
            (status, a['original_assessment']['reference'], obj(a)))
    result = []
    for (family, sid, cid), evidence in sorted(grouped.items()):
        require(len({e[1] for e in evidence}) == 1, 'reference text conflict')
        if len({e[0] for e in evidence}) != 1:
            continue  # Never vote or silently use the most recent assessment.
        status = evidence[0][0]
        require(status in ('positive', 'negative', 'unavailable'), 'invalid reference status')
        result.append(dict(family=family, source_id=sid, semantic_id=cid,
            expected_status=status, reference_text=evidence[0][1],
            assessment_sha256=sorted(e[2] for e in evidence)))
    return result


def select(coordinates, per_cell=2):
    require(type(per_cell) is int and per_cell > 0, 'positive count required')
    require(len({(c['family'],c['source_id'],c['semantic_id']) for c in coordinates}) == len(coordinates),
            'duplicate coordinate')
    selected = []
    for family in ('I_present', 'S_present'):
        for status in ('positive', 'negative', 'unavailable'):
            pool = sorted((c for c in coordinates if c['family']==family and c['expected_status']==status),
                key=lambda c: (obj(['three-state-calibration-v1',family,c['source_id'],c['semantic_id']]),
                               c['source_id'],c['semantic_id']))
            sources = set(); cell = []
            for c in pool:
                if c['source_id'] not in sources:
                    sources.add(c['source_id']); cell.append(c)
                if len(cell) == per_cell:
                    break
            require(len(cell) == per_cell, 'insufficient distinct sources in '+family+'/'+status)
            selected.extend(cell)
    return selected


def build(intent, slot, workpack, output):
    require(not output.exists(), 'output exists')
    pins = {}
    for root in (intent,slot,workpack):
        report = read(root/'report.json');pins[str(root/'report.json')] = sha(root/'report.json')
        for name, digest in report['files'].items():
            require(sha(root/name)==digest, 'input file changed')
            pins[str(root/name)] = digest
    selected = select(reference_coordinates(rows(intent/'assessments.jsonl'),rows(slot/'assessments.jsonl')))
    sources = unique(rows(workpack/'sources.jsonl'),'review_id')
    bindings = unique(rows(workpack/'bindings.offline.jsonl'),'source_id')
    templates = unique(rows(workpack/'templates.jsonl'),'panel_id')
    panels = {(t['family'],q['id']):t for t in templates.values() for q in t['questions']}
    requests = {}
    for c in selected:
        source = sources[bindings[c['source_id']]['review_id']]
        require(source['reference_text']==c.pop('reference_text'),'source text mismatch')
        t = panels[c['family'],c['semantic_id']]
        rid = obj([source['review_id'],obj(t)])
        c.update(request_id=rid,panel_id=t['panel_id'],review_id=source['review_id'],optimization_allowed=False)
        requests[rid] = dict(request_id=rid,panel_id=t['panel_id'],review_id=source['review_id'],
            template_sha256=obj(t),user=render(t,source))
    output.mkdir(parents=True)
    for name, values in [('cases.offline.jsonl',selected),('requests.jsonl',sorted(requests.values(),key=lambda r:r['request_id']))]:
        (output/name).write_text(''.join(json.dumps(v,ensure_ascii=False,sort_keys=True)+chr(10) for v in values))
    report=dict(schema='three-state-calibration-v1',pins=pins,implementation_sha256=sha(__file__),
        files={n:sha(output/n) for n in ('cases.offline.jsonl','requests.jsonl')},
        cases=len(selected),requests=len(requests),counts=dict(Counter(c['family']+'/'+c['expected_status'] for c in selected)),
        selection='Two distinct sources per family/state, fixed hash ordering; current-compatible agreeing author references only. No model-output selection.',
        scope='Balanced engineering calibration, not representative accuracy, independent gold or training data. Complete fixed panels on wire; selected reference answers remain offline.',
        API_sent=False,semantic_admission=0,training_dispatch_allowed=False)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+chr(10))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for k in ('intent','slot','workpack','output'):parser.add_argument('--'+k,type=Path,required=True)
    a=parser.parse_args();r=build(a.intent,a.slot,a.workpack,a.output)
    print(json.dumps({k:r[k] for k in ('cases','requests','counts','API_sent','semantic_admission')},indent=2))
