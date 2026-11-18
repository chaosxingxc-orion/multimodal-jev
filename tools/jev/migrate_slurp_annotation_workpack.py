#!/usr/bin/env python3
"""Migrate reference evidence to a pinned ontology without admitting labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, ensure_ascii=False)+'\n').encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).open()]


def semantic_contract(ontology, name, fields):
    """Incident boundary changes invalidate evidence even if the card is unchanged."""
    project = lambda row, keys: {k: row[k] for k in keys if k in row}
    card = next(r for r in ontology['slots'] if r['entity_type'] == name)
    policy = next(r for r in ontology['policies'] if r['entity_type'] == name)
    boundaries = sorted((r for r in ontology['slot_boundary_document']['rules'] if name in r['types']),
                        key=lambda r: tuple(r['types']))
    return dict(definition=project(card, fields['type']), policy=project(policy, fields['policy']),
                boundaries=[dict(types=r['types'], **project(r, fields['boundary'])) for r in boundaries])


def coordinate_summary(assessments):
    current = [r for r in assessments if r['applicability'] == 'current_semantics_reference_only']
    values = {r['proposed_value'] for r in current if r['proposed_value'] is not None}
    status = ('conflicting_current_proposals' if len(values) > 1 else
              'current_proposal_with_unresolved_assessment' if values and any(r['proposed_value'] is None for r in current) else
              'current_reference_proposal' if values else
              'current_reference_unavailable' if current else 'requires_current_rule_review')
    return dict(status=status, assessment_ids=[r['assessment_id'] for r in assessments],
                current_assessments=len(current), admitted_value=None, optimization_allowed=False,
                independent_review_inferred=False)


def sequential_values(review, source_index, names):
    """Validate declared source interval and complete judgments before merging."""
    schedule = review['schedule']
    start, stop = schedule['start'], schedule['stop']
    if (type(start) is not int or type(stop) is not int or
            not 0 <= start < stop <= len(source_index) or
            schedule['full_population_sources'] != len(source_index) or
            schedule['order'] != 'existing_annotation_workpack_v1_hash_order'):
        raise ValueError('invalid sequential schedule')
    order = sorted(source_index, key=lambda sid: hashlib.sha256(('annotation-workpack-v1:'+sid).encode()).hexdigest())
    if [r['source_id'] for r in review['records']] != order[start:stop]:
        raise ValueError('sequential source interval mismatch')
    values = []
    for r in review['records']:
        source = source_index[r['source_id']]
        pos, neg, unavailable = set(r['positive']), set(r['negative_types']), set(r['unavailable'])
        if (pos | neg | unavailable != names or len(pos)+len(neg)+len(unavailable) != len(names)
                or len(neg) != len(r['negative_types'])):
            raise ValueError('nonexhaustive or duplicated sequential review')
        if (r['reference_sha256'] != hashlib.sha256(source['reference'].encode()).hexdigest() or
                r['metadata_sha256'] != source['metadata_sha256']):
            raise ValueError('sequential reference mismatch')
        coverage = r['negative_coverage']
        if (coverage['explicitly_assessed_all55'] is not True or
                coverage['source_summary'] != source['reference'] or not coverage['basis'].strip()):
            raise ValueError('explicit negative assessment required')
        if (r['optimization_allowed'] is not False or r['actor_input_allowed'] is not False or
                r['independent_semantic_review'] is not False or r['modality'] != 'reference_text'):
            raise ValueError('unsupported sequential authority')
        for detail in list(r['positive'].values())+list(r['unavailable'].values()):
            if not detail['quote'] or detail['quote'] not in source['reference'] or not detail['rationale'].strip():
                raise ValueError('invalid sequential evidence')
        for n in sorted(names):
            values.append((r['source_id'], n, True if n in pos else None if n in unavailable else False, r))
    return values


def build(config_path, output):
    cfg = read(config_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pins = {str(config_path): sha(config_path)}

    def pin(path, expected=None):
        actual = sha(path)
        if expected is not None and expected != actual:
            raise ValueError('changed dependency: '+str(path))
        pins[str(path)] = actual

    def review_context(report_path, expected):
        """Read the exact ontology belonging to a reference's original workpack."""
        pin(report_path, expected)
        context = read(report_path)
        root = Path(report_path).parent
        for filename, checksum in context['files'].items():
            pin(root/filename, checksum)
        return read(root/'ontology.json')

    for f, h in cfg['pins'].items():
        pin(f, h)
    base = Path(cfg['base_workpack'])
    report = read(base/'report.json')
    for f, h in report['files'].items():
        pin(base/f, h)
    sources = rows(base/'sources.jsonl')
    source_index = {r['source_id']: r for r in sources}
    if len(source_index) != len(sources) or len(sources) != cfg['expected_source_count']:
        raise ValueError('source population changed')
    old = read(base/'ontology.json')
    pack = read(cfg['package'])
    for k in ('definitions', 'policy', 'boundaries'):
        if pack[k] not in cfg['pins']:
            raise ValueError('unpinned package component')
    current = dict(old, slots=read(pack['definitions'])['types'],
                   policies=read(pack['policy'])['policies'],
                   slot_boundary_document=read(pack['boundaries']))
    if read(pack['policy'])['source_definition_sha256'] != sha(pack['definitions']):
        raise ValueError('policy source mismatch')
    bd = current['slot_boundary_document']
    if bd['source_definition_sha256'] != sha(pack['definitions']) or bd['source_policy_sha256'] != sha(pack['policy']):
        raise ValueError('boundary source mismatch')
    names = {r['entity_type'] for r in current['slots']}
    if len(names) != 55 or names != {r['entity_type'] for r in current['policies']}:
        raise ValueError('incomplete ontology')
    fields = pack['runtime_field_groups']
    current_hashes = {n: digest(semantic_contract(current, n, fields)) for n in names}
    assessments = []

    def add(sid, name, value, raw, origin, prior_ontology):
        if sid not in source_index or name not in names:
            raise ValueError('unknown source or type')
        if value is not None and type(value) is not bool:
            raise ValueError('proposal must be boolean or unavailable')
        metadata_hash = raw.get('metadata_sha256', raw.get('metadata_record_sha256'))
        if metadata_hash != source_index[sid]['metadata_sha256'] or raw.get('optimization_allowed') is not False:
            raise ValueError('source identity or admission mismatch')
        if 'reference' in raw and raw['reference'] != source_index[sid]['reference']:
            raise ValueError('reference changed')
        previous = digest(semantic_contract(prior_ontology, name, fields))
        a = dict(source_id=sid, definition_id=name, proposed_value=value,
                 original_assessment=raw, original_assessment_sha256=digest(raw), provenance=origin,
                 previous_semantic_contract_sha256=previous, current_semantic_contract_sha256=current_hashes[name],
                 applicability='current_semantics_reference_only' if previous == current_hashes[name] else 'stale_semantics_requires_rereview',
                 admitted_value=None, optimization_allowed=False, independent_review_inferred=False)
        a['assessment_id'] = digest(a)
        assessments.append(a)

    for s in sources:
        for e in s['existing_reference_evidence']:
            if e['family'] != 'S':
                continue
            for a in [e]+e.get('additional_reference_assessments', []):
                d = a['decision']
                if digest(d) != a['decision_sha256']:
                    raise ValueError('historical assessment hash changed')
                add(s['source_id'], e['definition_id'], d['proposed_semantic_value'], d,
                    dict(kind='historical_workpack', report=a['report']), old)
    sequential_paths = cfg.get('sequential_reviews')
    if sequential_paths is None:
        sequential_paths = [cfg['sequential_review']]
    if len(sequential_paths) != len(set(sequential_paths)):
        raise ValueError('duplicate sequential input')
    for path in sequential_paths:
        if path not in cfg['pins']:
            raise ValueError('unpinned sequential review')
        sequential = read(path)
        context_root = Path(sequential['workpack'])
        previous = review_context(context_root/'report.json', sequential['workpack_report_sha256'])
        pin(context_root/'ontology.json', sequential['ontology_sha256'])
        for sid, n, value, record in sequential_values(sequential, source_index, names):
            add(sid, n, value, record, dict(kind='sequential_review', path=path), previous)
    for spec in cfg['boundary_reviews']:
        if spec['receipt'] not in cfg['pins']:
            raise ValueError('unpinned boundary receipt')
        receipt = read(spec['receipt'])
        pin(receipt['review'], receipt['review_sha256'])
        pin(receipt['rule'], receipt['rule_sha256'])
        package = receipt.get('new_package', receipt.get('new_definition_package'))
        pin(package['path'], package['sha256'])
        for f, h in package['files'].items():
            pin(f, h)
        previous_pack = read(package['path'])
        previous = dict(slots=read(previous_pack['definitions'])['types'],
                        policies=read(previous_pack['policy'])['policies'],
                        slot_boundary_document=read(previous_pack['boundaries']))
        review = read(receipt['review'])
        if review['rule_sha256'] != receipt['rule_sha256']:
            raise ValueError('boundary review identity mismatch')
        review_context(review['workpack_report'], review['workpack_report_sha256'])
        for r in review['records']:
            n = r.get('definition_id', spec.get('single_type'))
            if r['independent_semantic_review'] is not False or r['evidence_modality'] != 'reference_text':
                raise ValueError('unsupported review authority')
            add(r['source_id'], n, r['proposed_value'], r,
                dict(kind='boundary_review', path=receipt['review'], package=package['path']), previous)
    if len({a['assessment_id'] for a in assessments}) != len(assessments):
        raise ValueError('duplicate assessment')
    groups = defaultdict(list)
    for a in assessments:
        groups[(a['source_id'], a['definition_id'])].append(a)
    coordinates = [dict(source_id=sid, definition_id=n, **coordinate_summary(aa))
                   for (sid, n), aa in sorted(groups.items())]
    per_source = defaultdict(list)
    for c in coordinates:
        per_source[c['source_id']].append(c)
    # Preserve all original evidence verbatim; no alias as a current label.
    migrated = []
    for s in sources:
        item = dict(s)
        item['historical_reference_evidence'] = item.pop('existing_reference_evidence')
        item['slot_reference_coordinates'] = per_source[s['source_id']]
        item['semantic_annotation'] = None
        item['current_semantic_contracts'] = 'ontology-contracts.json'
        item['reference_scope'] = 'Current references remain proposals; stale references require rereview. No voting, latest-wins selection or automatic admission.'
        migrated.append(item)
    output.mkdir(parents=True)
    for filename, data in [('sources.jsonl', migrated), ('assessments.jsonl', assessments), ('coordinates.jsonl', coordinates)]:
        (output/filename).write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False)+'\n' for r in data))
    (output/'ontology.json').write_text(json.dumps(current, ensure_ascii=False, indent=2)+'\n')
    (output/'ontology-contracts.json').write_text(json.dumps(current_hashes, sort_keys=True, indent=2)+'\n')
    summary = dict(schema='slurp-reference-migration-workpack-v1', pins=pins, implementation_sha256=sha(__file__),
        source_utterances=len(sources), learning_recordings=report['learning_recordings'],
        source_recording_bindings=sum(len(s['recording_bindings']) for s in sources),
        slot_assessments=len(assessments), slot_reference_coordinates=len(coordinates),
        coordinate_status_counts=dict(Counter(c['status'] for c in coordinates)),
        assessment_applicability_counts=dict(Counter(a['applicability'] for a in assessments)),
        unassessed_slot_coordinates=len(sources)*len(names)-len(coordinates),
        historical_intent_coordinates=sum(e['family']=='I_fit' for s in sources for e in s['existing_reference_evidence']),
        admitted_coordinates=0, training_dispatch_allowed=False, full_population_preserved=True,
        scope='Evidence migration only. Neither current-compatible nor agreeing proposals constitute semantic approval.',
        files={f.name: sha(f) for f in output.iterdir()})
    (output/'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.config, args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('pins','files')}, indent=2))
