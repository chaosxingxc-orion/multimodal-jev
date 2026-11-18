#!/usr/bin/env python3
"""Freeze a label-blind quality cohort; no API dispatch or label admission."""
import argparse
from pathlib import Path
from collections import defaultdict
import json

from consolidate_intent_presence_references import read, rows, sha, obj, require
from build_blind_semantic_annotation import render


def unique_records(records, key):
    indexed = {}
    for record in records:
        value = record.get(key)
        require(isinstance(value, str) and bool(value) and value not in indexed,
                'duplicate or missing identity: ' + key)
        indexed[value] = record
    require(bool(indexed), 'empty collection: ' + key)
    return indexed


def load_quality_cohort(cohort):
    """Validate all scheduled source/panel pairs, not just aggregate size."""
    report = read(cohort/'report.json')
    require(report.get('schema') == 'jev-semantic-quality-audit-cohort-v1', 'invalid quality cohort')
    for name, digest in report['files'].items():
        require(sha(cohort/name) == digest, 'cohort file changed: ' + name)
    for name in ('sources.jsonl', 'templates.jsonl', 'schedule.jsonl', 'bindings.offline.jsonl'):
        require(name in report['files'], 'required cohort file not pinned')
    sources = unique_records(rows(cohort/'sources.jsonl'), 'review_id')
    templates = unique_records(rows(cohort/'templates.jsonl'), 'panel_id')
    schedule = unique_records(rows(cohort/'schedule.jsonl'), 'request_id')
    bindings = unique_records(rows(cohort/'bindings.offline.jsonl'), 'review_id')
    require(set(sources) == set(bindings), 'source/binding identity mismatch')
    require(len(sources) == report['sources'], 'source count mismatch')
    pairs = set()
    for rid, task in schedule.items():
        require(task['review_id'] in sources and task['panel_id'] in templates, 'unknown scheduled identity')
        require(task['template_sha256'] == obj(templates[task['panel_id']]), 'template binding changed')
        require(rid == obj([task['review_id'], task['template_sha256']]), 'request identity changed')
        pair = task['review_id'], task['panel_id']
        require(pair not in pairs, 'duplicate source/panel pair')
        pairs.add(pair)
    require(pairs == {(s, t) for s in sources for t in templates}, 'incomplete source/panel coverage')
    require(len(schedule) == report['requests_per_pass'], 'request count mismatch')
    groups = set()
    for binding in bindings.values():
        group = {b['source_group'] for b in binding['bindings']}
        require(len(group) == 1, 'ambiguous source group')
        groups.update(group)
    require(len(groups) == report['groups'], 'group count mismatch')
    for template in templates.values():
        unique_records(template['questions'], 'id')
    require(sum(len(templates[s['panel_id']]['questions']) for s in schedule.values())
            == report['decisions_per_pass'], 'decision count mismatch')
    return report, sources, templates, schedule, bindings


def prepare_inputs(cohort, system_prompt, role, output):
    """Freeze exact label-blind messages. Contains no API client or credentials."""
    require(role in ('proposer', 'blind_reviewer'), 'invalid review role')
    require(not output.exists(), 'output exists')
    report, sources, templates, schedule, _ = load_quality_cohort(cohort)
    prompt = system_prompt.read_text()
    require(bool(prompt.strip()), 'nonempty system prompt required')
    output.mkdir(parents=True)
    frozen_prompt = output/'system-prompt.txt'
    frozen_prompt.write_text(prompt)
    requests = []
    for rid, task in schedule.items():
        messages = [dict(role='system', content=prompt), dict(role='user',
            content=json.dumps(render(templates[task['panel_id']], sources[task['review_id']]),
                               ensure_ascii=False, sort_keys=True))]
        path = output/(rid+'.messages.json')
        path.write_text(json.dumps(messages, ensure_ascii=False, indent=2)+chr(10))
        requests.append(dict(request_id=rid, path=str(path.resolve()), sha256=sha(path)))
    manifest = dict(schema='jev-semantic-quality-input-manifest-v1', role=role,
        cohort_report_sha256=sha(cohort/'report.json'), cohort=str(cohort.resolve()),
        system_prompt_path=str(frozen_prompt.resolve()), system_prompt_sha256=sha(frozen_prompt),
        requests=requests, implementation_sha256=sha(__file__), API_sent=False,
        semantic_admission=0, training_dispatch_allowed=False,
        scope='Exact two-message inputs only. No reviewer identity, API configuration, independent-review certification or dispatch.')
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+chr(10))
    return manifest


def select_groups(bindings, excluded_sources, count, seed):
    require(type(count) is int and count > 0, 'positive group count required')
    groups = defaultdict(list)
    seen = set()
    for b in bindings:
        require(b['source_id'] not in seen, 'duplicate source')
        seen.add(b['source_id'])
        group_ids = {r['source_group'] for r in b['bindings']}
        require(len(group_ids) == 1, 'one explicit source group required')
        groups[next(iter(group_ids))].append(b)
    require(set(excluded_sources) <= seen, 'exclusion outside learning population')
    eligible = [g for g, members in groups.items()
                if not any(m['source_id'] in excluded_sources for m in members)]
    require(len(eligible) >= count, 'insufficient unexposed groups')
    selected = sorted(eligible, key=lambda g: (obj([seed, g]), g))[:count]
    return sorted((m for g in selected for m in groups[g]), key=lambda b: b['review_id'])


def prepare(workpack, intent, slot, output, count=128, seed='jev-semantic-quality-v1'):
    require(not output.exists(), 'output exists')
    pins = {}
    for root in (workpack, intent, slot):
        report = read(root/'report.json')
        pins[str(root/'report.json')] = sha(root/'report.json')
        for n, h in report['files'].items():
            require(sha(root/n) == h, 'changed input: '+n)
            pins[str(root/n)] = h
    bindings = rows(workpack/'bindings.offline.jsonl')
    exclusions = defaultdict(set)
    for a in rows(intent/'assessments.jsonl'):
        exclusions[a['decision']['source_id']].add('intent_reference_assessment')
    for a in rows(slot/'assessments.jsonl'):
        exclusions[a['source_id']].add('slot_reference_assessment')
    for s in rows(slot/'sources.jsonl'):
        if s['historical_reference_evidence']:
            exclusions[s['source_id']].add('historical_definition_reference')
    selected = select_groups(bindings, exclusions, count, seed)
    ids = {b['review_id'] for b in selected}
    sources = [s for s in rows(workpack/'sources.jsonl') if s['review_id'] in ids]
    schedule = [s for s in rows(workpack/'schedule.jsonl') if s['review_id'] in ids]
    templates = rows(workpack/'templates.jsonl')
    require(len(sources) == len(selected), 'source identity mismatch')
    require(len(schedule) == len(sources)*len(templates), 'incomplete schedule')
    require(len({s['request_id'] for s in schedule}) == len(schedule), 'duplicate request')
    decisions = sum(len(t['questions']) for t in templates)*len(sources)
    output.mkdir(parents=True)
    values = {'sources.jsonl': sources, 'bindings.offline.jsonl': selected,
              'schedule.jsonl': schedule, 'templates.jsonl': templates,
              'exclusions.offline.jsonl': [dict(source_id=s, reasons=sorted(v))
                                         for s,v in sorted(exclusions.items())]}
    for n, v in values.items():
        (output/n).write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in v))
    result = dict(schema='jev-semantic-quality-audit-cohort-v1', pins=pins,
        files={n:sha(output/n) for n in values}, implementation_sha256=sha(__file__),
        selection=dict(seed=seed, unit='source_group', requested_groups=count,
            excluded_sources=len(exclusions), group_closure=True,
            ordering='sha256 canonical JSON [seed, source_group]', label_based_sampling=False),
        sources=len(sources), groups=count, requests_per_pass=len(schedule),
        decisions_per_pass=decisions, full_population_sources=len(bindings),
        semantic_admission=0, API_sent=False, training_dispatch_allowed=False,
        scope='Random learning-source quality cohort excluding inventoried author-reference groups. Not a target-test split, not proof that no source was ever read, and not independent annotation. Paired reviewer outputs still required; full population remains training objective.')
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for k in ('workpack','intent','slot','cohort','system-prompt'): p.add_argument('--'+k,type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',choices=('proposer','blind_reviewer'))
    a=p.parse_args()
    if a.cohort is not None:
        if a.system_prompt is None or a.role is None or any(v is not None for v in (a.workpack,a.intent,a.slot)):
            p.error('input preparation requires --cohort --system-prompt --role; no source inventories')
        result=prepare_inputs(a.cohort,a.system_prompt,a.role,a.output)
        print(json.dumps(dict(requests=len(result['requests']),API_sent=False,semantic_admission=0),indent=2))
    else:
        if any(v is None for v in (a.workpack,a.intent,a.slot)) or a.system_prompt is not None or a.role is not None:
            p.error('cohort preparation requires --workpack --intent --slot')
        print(json.dumps(prepare(a.workpack,a.intent,a.slot,a.output),indent=2))
