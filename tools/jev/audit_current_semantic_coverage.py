#!/usr/bin/env python3
"""Count current reference coverage without converting proposals into training labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).open()]


def reference_state(assessments):
    current = [a['proposed_value'] for a in assessments
               if a['applicability'] == 'current_semantics_reference_only']
    if any(v is not None and type(v) is not bool for v in current):
        raise ValueError('reference values must be boolean or null')
    known = {v for v in current if v is not None}
    if len(known) > 1:
        return 'conflicting_current_proposals', None
    if known and None in current:
        return 'current_proposal_with_unresolved_assessment', None
    if known:
        return 'current_reference_proposal', next(iter(known))
    return ('current_reference_unavailable' if current else 'requires_current_rule_review'), None


def recording_state(source_ids, definition, coordinates):
    """Require complete, agreeing source coverage; never propagate source-group truth."""
    values = []
    for sid in source_ids:
        state, value = coordinates.get((sid, definition), ('unassessed', None))
        if state != 'current_reference_proposal':
            return 'incomplete_or_unresolved'
        values.append(value)
    if not values:
        raise ValueError('recording must have at least one source')
    if len(set(values)) > 1:
        return 'source_disagreement'
    return 'positive_reference' if values[0] else 'negative_reference'


def audit(sources, definitions, assessments, coordinates):
    src = {s['source_id']: s for s in sources}
    if len(src) != len(sources) or len(set(definitions)) != len(definitions):
        raise ValueError('duplicate source or definition')
    groups = defaultdict(list)
    ids = set()
    for a in assessments:
        if a['assessment_id'] in ids:
            raise ValueError('duplicate assessment')
        ids.add(a['assessment_id'])
        if a['source_id'] not in src or a['definition_id'] not in definitions:
            raise ValueError('unknown reference coordinate')
        if a['admitted_value'] is not None or a['optimization_allowed']:
            raise ValueError('reference-only audit cannot process admitted truth')
        if a['applicability'] not in ('current_semantics_reference_only', 'stale_semantics_requires_rereview'):
            raise ValueError('unknown applicability')
        groups[(a['source_id'], a['definition_id'])].append(a)
    lookup = {}
    for c in coordinates:
        key = c['source_id'], c['definition_id']
        if key in lookup or key not in groups:
            raise ValueError('duplicate or orphan coordinate')
        group = groups[key]
        if Counter(c['assessment_ids']) != Counter(a['assessment_id'] for a in group):
            raise ValueError('assessment membership differs')
        state = reference_state(group)
        if c['status'] != state[0] or c['admitted_value'] is not None or c['optimization_allowed']:
            raise ValueError('coordinate summary or admission differs')
        lookup[key] = state
    if set(lookup) != set(groups):
        raise ValueError('missing coordinate summary')
    recording_sources = {}
    recording_groups = {}
    actual_bindings = defaultdict(set)
    for sid, s in src.items():
        if s['optimization_allowed'] or s['semantic_annotation'] is not None:
            raise ValueError('unexpected source admission')
        local = set()
        for b in s['recording_bindings']:
            rid = b['recording_id']
            all_sources = tuple(sorted(b['all_source_ids']))
            if rid in local or len(set(all_sources)) != len(all_sources) or sid not in all_sources:
                raise ValueError('duplicate or inconsistent source binding')
            local.add(rid)
            if rid in recording_sources and (recording_sources[rid] != all_sources or recording_groups[rid] != b['source_group']):
                raise ValueError('recording identity differs')
            recording_sources[rid] = all_sources
            recording_groups[rid] = b['source_group']
            actual_bindings[rid].add(sid)
    for rid, expected in recording_sources.items():
        if set(expected) != actual_bindings[rid]:
            raise ValueError('incomplete full-population source bindings')
    result = []
    for name in sorted(definitions):
        states = Counter(lookup.get((sid, name), ('unassessed', None))[0] for sid in src)
        pos = [sid for sid in src if lookup.get((sid, name)) == ('current_reference_proposal', True)]
        neg = [sid for sid in src if lookup.get((sid, name)) == ('current_reference_proposal', False)]
        rstates = {rid: recording_state(sids, name, lookup) for rid, sids in recording_sources.items()}
        rc = Counter(rstates.values())
        positive_groups = Counter(recording_groups[rid] for rid, s in rstates.items() if s == 'positive_reference')
        negative_groups = Counter(recording_groups[rid] for rid, s in rstates.items() if s == 'negative_reference')
        assessed = len(src) - states['unassessed']
        clear = len(pos) + len(neg)
        result.append(dict(definition_id=name,source_status_counts=dict(states),
            positive_reference_sources=len(pos),negative_reference_sources=len(neg),
            any_reference_source_fraction=assessed/len(src),
            current_clear_source_fraction=clear/len(src),
            positive_share_among_clear_references=len(pos)/clear if clear else None,
            constant_false_agreement_on_clear_references=len(neg)/clear if clear else None,
            recording_status_counts=dict(rc),positive_reference_source_groups=len(positive_groups),
            negative_reference_source_groups=len(negative_groups),
            largest_positive_group_recording_share=max(positive_groups.values(),default=0)/sum(positive_groups.values()) if positive_groups else None,
            polarity_support='both' if pos and neg else 'positive_only' if pos else 'negative_only' if neg else 'neither',
            admitted_positive_sources=0,admitted_negative_sources=0))
    summary = dict(sources=len(src),recordings=len(recording_sources),source_groups=len(set(recording_groups.values())),
        definitions=len(definitions),total_source_type_coordinates=len(src)*len(definitions),
        reference_coordinates=len(lookup),assessments=len(assessments),
        source_status_counts=dict(sum((Counter(r['source_status_counts']) for r in result),Counter())),
        polarity_support_counts=dict(Counter(r['polarity_support'] for r in result)),
        no_positive_reference_types=[r['definition_id'] for r in result if not r['positive_reference_sources']],
        no_negative_reference_types=[r['definition_id'] for r in result if not r['negative_reference_sources']],
        single_positive_source_types=[r['definition_id'] for r in result if r['positive_reference_sources']==1],
        admitted_coordinates=0)
    return dict(summary=summary,per_type=result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workpack', type=Path, required=True)
    p.add_argument('--report-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report_path = args.workpack/'report.json'
    if sha(report_path) != args.report_sha256:
        raise ValueError('workpack report identity differs')
    report = read(report_path)
    for name, expected in report['files'].items():
        if sha(args.workpack/name) != expected:
            raise ValueError('workpack file identity differs: '+name)
    result = audit(rows(args.workpack/'sources.jsonl'),
                   [t['entity_type'] for t in read(args.workpack/'ontology.json')['slots']],
                   rows(args.workpack/'assessments.jsonl'),rows(args.workpack/'coordinates.jsonl'))
    if result['summary']['sources'] != report['source_utterances'] or result['summary']['recordings'] != report['learning_recordings']:
        raise ValueError('full population differs')
    result.update(schema='current-semantic-reference-coverage-v1',workpack=str(args.workpack),
        workpack_report_sha256=args.report_sha256,implementation_sha256=sha(__file__),
        input_files=report['files'],training_dispatch_allowed=False,
        scope='Reference-only coverage, not semantic admission, model accuracy or population prevalence. Diagnostic review selection is nonrandom. No training weights derived.',
        recording_policy='All source IDs must have clear agreeing current proposals; no source-group propagation. Recording counts are bindings, not independently listened-to evidence.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__ == '__main__':
    main()
