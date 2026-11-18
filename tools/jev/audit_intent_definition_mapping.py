#!/usr/bin/env python3
"""Reconstruct full-population intent fit coverage; never manufacture truth."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, ensure_ascii=False) + '\n').encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(items, key):
    result = {key(item): item for item in items}
    require(len(result) == len(items), 'duplicate identity')
    return result


def recording_fit(source_ids, definition, judgments):
    """Consensus of explicit source IDs only; missing is not negative."""
    values = [judgments.get((str(s), definition), {}).get('definition_fit') for s in source_ids]
    require(bool(values), 'empty source identity')
    if any(v is None for v in values):
        return 'unassessed' if all(v is None for v in values) else 'partial_source_review'
    if len(set(values)) != 1:
        return 'source_fit_disagreement'
    return values[0]


def verify_decision(d, metadata, card):
    label = metadata['scenario'] + '_' + metadata['action']
    require(d['metadata_sha256'] == object_sha(metadata), 'metadata identity mismatch')
    require(d['reference'] == metadata['sentence'], 'reference mismatch')
    require(d['native_intent'] == label, 'native class mismatch')
    require(d['semantic_id'] == card['semantic_id'], 'definition identity mismatch')
    require(d['definition_sha256'] == object_sha(card), 'definition revision mismatch')
    require(d['definition_fit'] in ('supported', 'not_supported', 'unresolved'), 'invalid fit')
    value = {'supported': True, 'not_supported': False, 'unresolved': None}[d['definition_fit']]
    member = label == card['semantic_id']
    require(d['native_membership'] is member, 'native membership mismatch')
    require(d['proposed_definition_fit'] is value, 'fit value mismatch')
    require(d['disagrees_with_native_membership'] is (value is not None and value != member), 'mismatch flag')
    require(d['exclusive_I_class_truth'] is None and d['optimization_allowed'] is False,
            'fit cannot be exclusive truth or training admission')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-config', type=Path, required=True)
    parser.add_argument('--definitions', type=Path, required=True)
    parser.add_argument('--authoring-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'output already exists')
    spec = read(args.evidence_config)
    cards = unique(read(args.definitions)['cards'], lambda c: c['semantic_id'])
    authoring = read(args.authoring_audit)
    require(authoring['cards_sha256'] == sha(args.definitions), 'authoring cards changed')
    packet = read(authoring['packet'])
    require(sha(authoring['packet']) == authoring['packet_sha256'], 'authoring packet changed')
    pins = dict(packet['pins'])
    pins.update({str(p): sha(p) for p in (args.evidence_config, args.definitions, args.authoring_audit)})
    pins[authoring['packet']] = authoring['packet_sha256']
    for path, digest in pins.items():
        require(sha(path) == digest, 'changed dependency: ' + path)
    find = lambda suffix: next(Path(p) for p in pins if p.endswith(suffix))
    learning = rows(find('/learning.jsonl'))
    recs = unique(learning, lambda r: r['id'])
    targets = unique(rows(find('/learning.targets.jsonl')), lambda r: r['id'])
    source_recs = defaultdict(list)
    for r in learning:
        require(r['role'] == 'learning' and r['official_split'] == 'train' and not r['known_pending_source'], 'invalid learning role')
        t = targets[r['id']]
        require(object_sha(t) == r['source_target_sha256'], 'target identity mismatch')
        require(t['native_target']['scenario'] == r['scenario'] and t['native_target']['action'] == r['action'], 'target label mismatch')
        require(r['intent'] == r['scenario'] + '_' + r['action'], 'recording label mismatch')
        ids = [str(s) for s in t['source_ids']]
        require(bool(ids) and len(ids) == len(set(ids)), 'invalid source IDs')
        for sid in ids:
            source_recs[sid].append(r)
    metadata = unique([r for r in rows(find('/metadata/train.jsonl')) if str(r['slurp_id']) in source_recs], lambda r: str(r['slurp_id']))
    require(set(metadata) == set(source_recs), 'missing source metadata')
    for sid, meta in metadata.items():
        require(all(r['intent'] == meta['scenario'] + '_' + meta['action'] for r in source_recs[sid]), 'source label disagreement')
    require(len(recs) == 40513 and len(metadata) == 9192 and len(cards) == 60, 'unexpected population')
    require(set(cards) == {r['intent'] for r in learning}, 'class population mismatch')

    judgments, binding_counts = {}, {}
    for evidence in spec['sources']:
        if evidence['kind'] != 'intent_fit':
            continue
        report_path = Path(evidence['report'])
        require(sha(report_path) == evidence['sha256'], 'review report changed')
        report = read(report_path)
        require(report['semantic_supervision_admitted'] is False, 'unexpected semantic admission')
        dependencies = {str(report_path): evidence['sha256'], **report['pins'],
                        **{str(report_path.parent / p): h for p, h in report['files'].items()}}
        for path, digest in dependencies.items():
            require(sha(path) == digest, 'changed review dependency: ' + path)
            require(path not in pins or pins[path] == digest, 'conflicting identity')
            pins[path] = digest
        decisions = rows(report_path.parent / 'decisions.jsonl')
        cohort = {d['source_id'] for d in decisions}
        expected = []
        for d in decisions:
            sid, label = d['source_id'], d['semantic_id']
            require((sid, label) not in judgments, 'duplicate source-definition review')
            verify_decision(d, metadata[sid], cards[label])
            judgments[sid, label] = d
            for r in source_recs[sid]:
                expected.append(dict(source_id=sid, recording_id=r['id'], source_group=r['source_group'],
                    audio_sha256=r['audio_sha256'], source_target_sha256=r['source_target_sha256'],
                    review_sha256=object_sha(d), definition_fit=d['definition_fit'],
                    complete_source_review=all(str(s) in cohort for s in targets[r['id']]['source_ids']),
                    optimization_allowed=False))
        actual = rows(report_path.parent / 'recording-bindings.jsonl')
        require(Counter(map(object_sha, expected)) == Counter(map(object_sha, actual)), 'source-recording bindings differ')
        binding_counts[evidence['id']] = len(actual)

    source_rows = []
    for sid in sorted(metadata, key=int):
        m = metadata[sid]
        label = m['scenario'] + '_' + m['action']
        assessed = {c: judgments[sid, c] for c in cards if (sid, c) in judgments}
        source_rows.append(dict(source_id=sid, metadata_sha256=object_sha(m), native_intent=label,
            reference=m['sentence'], recording_ids=sorted(r['id'] for r in source_recs[sid]),
            own_definition_fit=assessed.get(label, {}).get('definition_fit', 'unassessed'),
            assessed_definition_fits={c: dict(fit=d['definition_fit'], review_sha256=object_sha(d)) for c, d in assessed.items()},
            exclusive_I_class_truth=None, optimization_allowed=False))

    per_card = []
    for label, card in sorted(cards.items()):
        own = [r for r in source_rows if r['native_intent'] == label]
        native_recs = [r for r in learning if r['intent'] == label]
        assessed = [d for (_, c), d in judgments.items() if c == label]
        per_card.append(dict(semantic_id=label, definition_sha256=object_sha(card),
            native_sources=len(own), native_recordings=len(native_recs),
            own_source_fit_counts=dict(Counter(r['own_definition_fit'] for r in own)),
            own_recording_fit_counts=dict(Counter(recording_fit(targets[r['id']]['source_ids'], label, judgments) for r in native_recs)),
            cross_native_source_fit_counts=dict(Counter(d['definition_fit'] for d in assessed if not d['native_membership'])),
            total_assessed_sources=len(assessed), total_unassessed_sources=len(metadata)-len(assessed),
            authoring_findings=authoring['issues'].get(label),
            ready_for_exclusive_semantic_supervision=False))
    pair_matrices = []
    for left, right in combinations(sorted(cards), 2):
        both = [r['source_id'] for r in source_rows if (r['source_id'], left) in judgments and (r['source_id'], right) in judgments]
        if both:
            cells = defaultdict(list)
            for sid in both:
                cells[judgments[sid, left]['definition_fit'] + '|' + judgments[sid, right]['definition_fit']].append(sid)
            pair_matrices.append(dict(left=left, right=right, assessed_sources=len(both),
                cells={k: dict(sources=len(ids), source_ids=ids) for k, ids in sorted(cells.items())}))
    own_counts = Counter(r['own_definition_fit'] for r in source_rows)
    recording_counts = Counter(recording_fit(targets[r['id']]['source_ids'], r['intent'], judgments) for r in learning)
    summary = dict(population_recordings=len(learning), population_sources=len(metadata), classes=len(cards),
        assessed_source_definition_coordinates=len(judgments), sources_with_any_fit=sum(bool(r['assessed_definition_fits']) for r in source_rows),
        own_source_fit_counts=dict(own_counts), own_recording_fit_counts=dict(recording_counts),
        classes_with_any_fit=sum(c['total_assessed_sources'] > 0 for c in per_card),
        total_unassessed_source_definition_coordinates=len(metadata)*len(cards)-len(judgments),
        cross_native_supported_coordinates=sum(not d['native_membership'] and d['definition_fit'] == 'supported' for d in judgments.values()),
        source_recording_review_bindings=binding_counts, semantic_supervision_admitted=0)
    args.output.mkdir(parents=True)
    for name, values in [('sources.jsonl', source_rows), ('cards.jsonl', per_card)]:
        (args.output / name).write_text(''.join(json.dumps(v, ensure_ascii=False, sort_keys=True)+'\n' for v in values))
    (args.output / 'paired-fits.json').write_text(json.dumps(pair_matrices, ensure_ascii=False, indent=2)+'\n')
    report = dict(schema='slurp-intent-definition-mapping-audit-v1', pins=pins,
        implementation_sha256=sha(__file__), summary=summary,
        files={name: sha(args.output/name) for name in ('sources.jsonl', 'cards.jsonl', 'paired-fits.json')},
        training_dispatch_allowed=False,
        scope='Full learning population identity and existing reference-fit coverage only. No new semantic judgments, independent semantic certification, audio listening, exclusive truth, native errors, exclusions or training release inferred. Unassessed cross-class coordinates are not assumed necessary to annotate exhaustively.')
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
