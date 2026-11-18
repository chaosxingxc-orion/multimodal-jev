#!/usr/bin/env python3
"""Consolidate reference-only I_present proposals, preserving policy provenance.

Run from repository root. This is an annotation inventory, never a training loader.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def obj(value):
    return hashlib.sha256((json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(values, field):
    result = {str(v[field]): v for v in values}
    require(len(result) == len(values), 'duplicate ' + field)
    return result


def pin(pins, path, digest):
    path = str(path)
    require(path not in pins or pins[path] == digest, 'conflicting file identity: ' + path)
    require(sha(path) == digest, 'changed dependency: ' + path)
    pins[path] = digest


def catalog(path, pins):
    """Resolve imported cards without modifying the object used by historical hashes."""
    pin(pins, path, sha(path))
    data = read(path)
    for p, h in data['policies'].items():
        pin(pins, p, h)
    cards = unique(data['authored_cards'], 'semantic_id')
    acts = {key: c['act_policy'] for key, c in cards.items()}
    for ref in data['imported_cards']:
        pin(pins, ref['path'], sha(ref['path']))
        card = unique(read(ref['path'])['cards'], 'semantic_id')[ref['semantic_id']]
        require(obj(card) == ref['card_sha256'], 'imported card changed')
        require(ref['semantic_id'] not in cards, 'duplicate imported card')
        cards[ref['semantic_id']] = card
        acts[ref['semantic_id']] = ref['act_policy']
    require(len(cards) == data['semantic_definition_count'], 'catalog count mismatch')
    return cards, {key: contract(c, acts[key], data['policies']) for key, c in cards.items()}


def contract(card, act, policies):
    require(card['operator'] == 'I_present', 'operator is not I_present')
    require(act in ('request', 'greeting', 'preference'), 'unknown act policy')
    suffix = '/intent-request-semantics-' if act == 'request' else '/intent-nonrequest-semantics-'
    matching = [p for p in policies if suffix in p]
    require(len(matching) == 1, 'missing or ambiguous historical act policy')
    result = dict(card_sha256=obj(card), act_policy=act, act_policy_sha256=policies[matching[0]])
    if card.get('boundary_policy'):
        require(card['boundary_policy'] in policies, 'missing historical boundary policy')
        result['boundary_policy_sha256'] = policies[card['boundary_policy']]
    return result


def incompatibilities(old, current):
    return sorted(k for k in old.keys() | current.keys() if old.get(k) != current.get(k))


def coordinate(assessments):
    """Count unique coordinates, never majority-vote or latest-write-wins."""
    if not assessments:
        return 'unassessed'
    active = [a for a in assessments if not a['incompatible_fields']]
    if not active:
        return 'needs_rereview'
    statuses = {a['decision']['status'] for a in active}
    return next(iter(statuses)) if len(statuses) == 1 else 'conflict'


def recording_state(source_ids, semantic_id, assessments):
    require(bool(source_ids), 'empty recording sources')
    states = [coordinate(assessments.get((str(s), semantic_id), [])) for s in source_ids]
    if len(set(states)) == 1:
        return states[0]
    if 'conflict' in states:
        return 'conflict'
    if 'unassessed' in states:
        return 'partial_source_review'
    if 'needs_rereview' in states:
        return 'needs_rereview'
    return 'source_disagreement'


def verify_decision(d, metadata, original_contract):
    require(d['metadata_sha256'] == obj(metadata), 'metadata mismatch')
    require(d['reference'] == metadata['sentence'], 'reference mismatch')
    require(d['native_intent'] == metadata['scenario'] + '_' + metadata['action'], 'native label mismatch')
    require(d['definition_sha256'] == original_contract['card_sha256'], 'historical card mismatch')
    for key in ('request_policy_sha256', 'boundary_policy_sha256'):
        if key in d:
            field = 'act_policy_sha256' if key.startswith('request') else key
            require(d[key] == original_contract[field], 'historical policy mismatch')
    require(d.get('operator', 'I_present') == 'I_present', 'wrong decision operator')
    values = {'positive': True, 'negative': False, 'unavailable': None}
    require(d['status'] in values and d['proposed_value'] is values[d['status']], 'status/value mismatch')
    require(d['optimization_allowed'] is False and d['audio_listening'] is False
            and d['independent_semantic_review'] is False and d.get('exclusive_I_class_truth') is None,
            'reference-only permissions violated')


def build(spec_path, output):
    require(not output.exists(), 'output exists; publish a new version')
    spec = read(spec_path)
    pins = {}
    pin(pins, spec_path, sha(spec_path))
    for p, h in spec['pins'].items():
        pin(pins, p, h)
    require(all(spec[k] in spec['pins'] for k in ('catalog', 'learning', 'targets', 'metadata')),
            'all primary inputs must be explicitly pinned')
    cards, current = catalog(spec['catalog'], pins)
    learning = rows(spec['learning'])
    recs = unique(learning, 'id')
    targets = unique(rows(spec['targets']), 'id')
    all_metadata = unique(rows(spec['metadata']), 'slurp_id')
    by_source = defaultdict(list)
    for r in learning:
        require(r['role'] == 'learning' and r['official_split'] == 'train'
                and not r['known_pending_source'], 'invalid learning role')
        t = targets[r['id']]
        require(obj(t) == r['source_target_sha256'], 'target mismatch')
        require(r['intent'] == r['scenario'] + '_' + r['action'], 'invalid native class')
        require(t['native_target']['scenario'] == r['scenario']
                and t['native_target']['action'] == r['action'], 'native target mismatch')
        ids = list(map(str, t['source_ids']))
        require(bool(ids) and len(ids) == len(set(ids)), 'invalid source IDs')
        for sid in ids:
            meta = all_metadata[sid]
            require(meta['scenario'] + '_' + meta['action'] == r['intent'], 'source label mismatch')
            require(Path(r['audio']).name in {x['file'] for x in meta['recordings']}, 'source audio mismatch')
            by_source[sid].append(r)
    require(len(recs) == spec['expected_recordings'] and len(by_source) == spec['expected_sources'], 'population mismatch')

    assessments = defaultdict(list)
    all_assessments, package_counts = [], {}
    require(len({p['report'] for p in spec['packages']}) == len(spec['packages']), 'duplicate package')
    for package in spec['packages']:
        path = Path(package['report'])
        pin(pins, path, package['sha256'])
        report = read(path)
        require(report['semantic_admission'] == 0 and report['training_dispatch_allowed'] is False, 'not reference-only package')
        for p, h in report['pins'].items():
            pin(pins, p, h)
        for p, h in report['files'].items():
            pin(pins, path.parent / p, h)
        definition_path = package['definitions']
        require(definition_path in report['pins'], 'original definitions not pinned by report')
        definition = read(definition_path)
        if 'authored_cards' in definition:
            _, original = catalog(definition_path, pins)
        else:
            policy = definition['request_policy']
            require(policy in report['pins'], 'original policy not pinned by report')
            original = {c['semantic_id']: contract(c, 'request', {policy: report['pins'][policy]})
                        for c in definition['cards']}
        decisions = rows(path.parent / 'decisions.jsonl')
        reviewed = {(str(d['source_id']), d['semantic_id']) for d in decisions}
        expected_bindings = []
        for ordinal, d in enumerate(decisions):
            sid, key = str(d['source_id']), d['semantic_id']
            require(sid in by_source and key in current, 'out-of-population decision')
            verify_decision(d, all_metadata[sid], original[key])
            a = dict(package=str(path), package_row=ordinal, decision=d, review_sha256=obj(d),
                     original_contract=original[key], current_contract=current[key],
                     incompatible_fields=incompatibilities(original[key], current[key]),
                     optimization_allowed=False)
            all_assessments.append(a)
            assessments[sid, key].append(a)
            for r in by_source[sid]:
                expected_bindings.append(dict(source_id=sid, semantic_id=key, recording_id=r['id'],
                    audio_sha256=r['audio_sha256'], source_group=r['source_group'],
                    source_target_sha256=r['source_target_sha256'], review_sha256=obj(d),
                    proposed_value=d['proposed_value'], optimization_allowed=False,
                    complete_source_review=all((str(s), key) in reviewed for s in targets[r['id']]['source_ids'])))
        actual = rows(path.parent / 'recording-bindings.jsonl')
        require(Counter(map(obj, expected_bindings)) == Counter(map(obj, actual)), 'source bindings mismatch')
        package_counts[str(path)] = dict(assessments=len(decisions), bindings=len(actual))

    source_rows = []
    for sid in sorted(by_source, key=int):
        known = {key: dict(status=coordinate(assessments[sid, key]),
                          assessment_refs=[dict(package=a['package'], row=a['package_row']) for a in assessments[sid, key]])
                 for key in cards if (sid, key) in assessments}
        source_rows.append(dict(source_id=sid, metadata_sha256=obj(all_metadata[sid]),
            recording_ids=sorted(r['id'] for r in by_source[sid]), assessments=known,
            absent_coordinate_state='unassessed', operator='I_present', optimization_allowed=False))
    recording_rows = []
    for r in learning:
        ids = list(map(str, targets[r['id']]['source_ids']))
        keys = {key for sid in ids for key in cards if (sid, key) in assessments}
        recording_rows.append(dict(recording_id=r['id'], source_ids=ids, source_group=r['source_group'],
            audio_sha256=r['audio_sha256'], source_target_sha256=r['source_target_sha256'],
            assessments={key: recording_state(ids, key, assessments) for key in sorted(keys)},
            absent_coordinate_state='unassessed', optimization_allowed=False))
    card_rows = []
    for key in sorted(cards):
        sc = Counter(coordinate(assessments.get((sid, key), [])) for sid in by_source)
        rc = Counter(r['assessments'].get(key, 'unassessed') for r in recording_rows)
        card_rows.append(dict(semantic_id=key, current_contract=current[key], source_counts=dict(sc),
                              recording_counts=dict(rc), semantic_admission=0))
    counts = Counter()
    for row in card_rows:
        counts.update(row['source_counts'])
    summary = dict(recordings=len(recs), sources=len(by_source), semantic_cards=len(cards),
        assessments=len(all_assessments), assessed_coordinates=len(assessments),
        sources_with_assessments=len({sid for sid, _ in assessments}),
        source_coordinate_counts=dict(counts),
        cards_with_positive_and_negative=sum(c['source_counts'].get('positive', 0) > 0
            and c['source_counts'].get('negative', 0) > 0 for c in card_rows),
        cards_without_assessments=sum(c['source_counts'].get('unassessed', 0) == len(by_source) for c in card_rows),
        incompatible_assessments=sum(bool(a['incompatible_fields']) for a in all_assessments),
        duplicate_coordinates=sum(len(a) > 1 for a in assessments.values()), semantic_admission=0)
    output.mkdir(parents=True)
    files = {}
    for name, values in [('assessments.jsonl', all_assessments), ('sources.jsonl', source_rows),
                         ('recordings.jsonl', recording_rows), ('cards.jsonl', card_rows)]:
        (output / name).write_text(''.join(json.dumps(v, ensure_ascii=False, sort_keys=True) + '\n' for v in values))
        files[name] = sha(output / name)
    result = dict(schema='slurp-intent-presence-reference-inventory-v1', pins=pins, files=files,
        implementation_sha256=sha(__file__), summary=summary, packages=package_counts,
        semantic_admission=0, training_dispatch_allowed=False,
        scope='Full population inventory of reference-text proposals. Current means matching card and applicable policies, not independent semantic approval. Sparse missing coordinates mean unassessed, never false. All proposals retained; no majority vote, native-membership conversion, source-group propagation or training admission.')
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.spec, args.output), indent=2))
