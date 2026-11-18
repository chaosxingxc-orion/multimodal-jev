#!/usr/bin/env python3
"""Fixed reference-agreement calibration; never a training or runtime Q selector."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from build_blind_semantic_annotation import render
from consolidate_intent_presence_references import obj, read, require, rows, sha, unique


def select(assessments, ids, ambiguity_limit=16):
    groups = defaultdict(list)
    for a in assessments:
        d = a['decision']
        require(d['semantic_id'] in ids, 'unknown semantic identity')
        require(not a['incompatible_fields'], 'stale reference contract')
        require(not a['optimization_allowed'], 'reference-only inventory required')
        require(d['status'] in ('positive', 'negative', 'unavailable'), 'invalid status')
        groups[d['source_id'], d['semantic_id']].append(a)
    eligible = []
    for (sid, cid), evidence in groups.items():
        # Conflicting duplicate assessments cannot be voted away or latest-wins.
        if len({a['decision']['status'] for a in evidence}) != 1:
            continue
        require(len({a['decision']['reference'] for a in evidence}) == 1, 'source text mismatch')
        eligible.append(dict(source_id=sid, semantic_id=cid,
            expected_status=evidence[0]['decision']['status'],
            reference_text=evidence[0]['decision']['reference'],
            assessment_sha256=sorted(obj(a) for a in evidence)))
    order = lambda c: obj(['intent-calibration-v1', c['source_id'], c['semantic_id']])
    selected = []
    for cid in sorted(ids):
        for status in ('positive', 'negative'):
            pool = [c for c in eligible if c['semantic_id'] == cid and c['expected_status'] == status]
            require(pool, f'missing polarity reference: {cid}/{status}')
            selected.append(min(pool, key=order))
    ambiguous = sorted((c for c in eligible if c['expected_status'] == 'unavailable'), key=order)
    return selected + ambiguous[:ambiguity_limit]


def build(inventory, workpack, output):
    require(not output.exists(), 'output exists')
    pins = {}
    for root in (inventory, workpack):
        report = read(root / 'report.json')
        pins[str(root / 'report.json')] = sha(root / 'report.json')
        for name, digest in report['files'].items():
            require(sha(root / name) == digest, 'changed input: ' + name)
    templates = unique([t for t in rows(workpack / 'templates.jsonl') if t['family'] == 'I_present'], 'panel_id')
    panel_of = {q['id']: t['panel_id'] for t in templates.values() for q in t['questions']}
    require(len(panel_of) == 65 and sum(len(t['questions']) for t in templates.values()) == 65,
            'complete nonduplicated intent ontology required')
    cases = select(rows(inventory / 'assessments.jsonl'), set(panel_of))
    bindings = unique(rows(workpack / 'bindings.offline.jsonl'), 'source_id')
    sources = unique(rows(workpack / 'sources.jsonl'), 'review_id')
    requests = {}
    for c in cases:
        source = sources[bindings[c['source_id']]['review_id']]
        require(source['reference_text'] == c.pop('reference_text'), 'reference binding mismatch')
        t = templates[panel_of[c['semantic_id']]]
        rid = obj(['intent-calibration-v1', source['review_id'], obj(t)])
        c.update(request_id=rid, panel_id=t['panel_id'], review_id=source['review_id'],
                 optimization_allowed=False)
        requests[rid] = dict(request_id=rid, panel_id=t['panel_id'], review_id=source['review_id'],
                            template_sha256=obj(t), wire=render(t, source))
    output.mkdir(parents=True, mode=0o700)
    for name, values in [('cases.offline.jsonl', cases), ('requests.jsonl', [requests[k] for k in sorted(requests)])]:
        (output / name).write_text(''.join(json.dumps(v, ensure_ascii=False, sort_keys=True) + '\n' for v in values))
    report = dict(schema='slurp-intent-teacher-calibration-v1', pins=pins,
        implementation_sha256=sha(__file__), files={n: sha(output / n) for n in ('cases.offline.jsonl', 'requests.jsonl')},
        summary=dict(cards=65, positive=65, negative=65, unavailable=len(cases)-130,
                     cases=len(cases), requests=len(requests), sources=len({c['source_id'] for c in cases}),
                     returned_decisions_if_complete=sum(len(r['wire']['questions']) for r in requests.values())),
        selection='One positive and one negative per card, plus16 hash-ordered ambiguous coordinates; duplicate evidence retained, conflicting coordinates excluded.',
        scope='Selected reference agreement, not independent gold accuracy, population accuracy or training admission. Selection uses offline references only; wire preserves complete fixed panels and contains no past answers.',
        evaluation='Report all130 known targets as denominator, including request failures; separate ambiguity agreement, coverage and per-polarity results. Other returned decisions are unscored proposals.',
        semantic_admission=0, training_dispatch_allowed=False, API_sent=False)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--workpack', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.inventory, a.workpack, a.output)['summary']))
