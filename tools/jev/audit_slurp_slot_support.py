#!/usr/bin/env python3
"""Count native slot evidence over the frozen learning population, not labels for RL."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--definitions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = read(a.release/'MANIFEST.json')
    if sha(a.release/'MANIFEST.json') != a.manifest_sha256:
        raise ValueError('release identity mismatch')
    source = Path(manifest['source'])
    sm = read(source/'MANIFEST.json')
    metadata_path = source.parent/'metadata/train.jsonl'
    pins = {str(a.release/'MANIFEST.json'): a.manifest_sha256,
            str(a.release/'learning.jsonl'): manifest['files']['learning.jsonl'],
            str(source/'MANIFEST.json'): sha(source/'MANIFEST.json'),
            str(source/'learning.targets.jsonl'): sm['files']['learning.targets.jsonl'],
            str(metadata_path): sm['identity']['input_hashes']['clean_metadata_sha256'],
            str(a.definitions): sha(a.definitions)}
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError(f'changed dependency: {path}')
    definitions = read(a.definitions)
    cards = {c['entity_type']: c for c in definitions['types']}
    if len(cards) != 55:
        raise ValueError('expected 55 distinct type definitions')
    evidence_path = Path(definitions['evidence_release'])/'type-review-cards.json'
    if sha(evidence_path) != definitions['evidence_sha256']:
        raise ValueError('old definition evidence changed')
    pins[str(evidence_path)] = sha(evidence_path)
    learning = rows(a.release/'learning.jsonl')
    targets = {r['id']: r for r in rows(source/'learning.targets.jsonl')}
    wanted = set()
    support, groups, source_refs = Counter(), defaultdict(set), defaultdict(set)
    cooccurrence, multiplicity = Counter(), Counter()
    positive_recordings = []
    for row in learning:
        if row['role'] != 'learning' or row['official_split'] != 'train' or row['known_pending_source']:
            raise ValueError('unexpected population role')
        target = targets[row['id']]
        content = (json.dumps(target, sort_keys=True, ensure_ascii=False)+'\n').encode()
        if hashlib.sha256(content).hexdigest() != row['source_target_sha256']:
            raise ValueError('native target mismatch')
        ids = {str(s) for s in target['source_ids']}
        wanted.update(ids)
        entities = target['native_target']['entities']
        types = {e['type'] for e in entities}
        if not types <= cards.keys():
            raise ValueError('unknown entity type')
        support.update(types)
        multiplicity[len(types)] += 1
        cooccurrence.update(itertools.combinations(sorted(types), 2))
        for t in types:
            groups[t].add(row['source_group'])
            source_refs[t].update(ids)
        positive_recordings.append(dict(id=row['id'], source_group=row['source_group'],
                                        annotated_types=sorted(types)))
    if len(learning) != manifest['counts']['learning']['recordings']:
        raise ValueError('population count differs')
    metadata = {str(r['slurp_id']): r for r in rows(metadata_path) if str(r['slurp_id']) in wanted}
    if set(metadata) != wanted:
        raise ValueError('missing source references')
    for row in metadata.values():
        tokens = {token['id']: token['surface'] for token in row['tokens']}
        row['entities'] = [dict(type=e['type'], span=e['span'],
                                filler=' '.join(tokens[i] for i in e['span'])) for e in row['entities']]
    by_type, fillers = defaultdict(list), defaultdict(Counter)
    for sid, row in metadata.items():
        for t in {e['type'] for e in row['entities']}:
            by_type[t].append(sid)
        for entity in row['entities']:
            fillers[entity['type']][entity['filler']] += 1
    output = []
    for t, card in sorted(cards.items()):
        ids = sorted(by_type[t], key=lambda s: hashlib.sha256(('slot-policy-v1:'+t+':'+s).encode()).hexdigest())[:3]
        examples = [dict(source_id=s, reference=metadata[s]['sentence'],
                         annotations=[e for e in metadata[s]['entities'] if e['type'] == t]) for s in ids]
        old_ids = set(card['evidence_source_ids'])
        output.append(dict(entity_type=t, annotated_recordings=support[t],
            annotation_absent_recordings=len(learning)-support[t], source_groups=len(groups[t]),
            annotated_source_utterances=len(by_type[t]),
            previous_card_sources_outside_current_learning=sorted(old_ids-wanted),
            examples=examples, most_common_annotated_fillers=fillers[t].most_common(6),
            semantic_positive_admission='not_established_by_this_audit',
            semantic_negative_admission='not_established_by_annotation_omission',
            activated_role_supervision='no_independent_annotation_field'))
    a.output.mkdir(parents=True)
    records_path = a.output/'records.jsonl'
    records_path.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in positive_recordings))
    result = dict(schema='slurp-slot-support-audit-v1', pins=pins,
        implementation_sha256=sha(__file__), recordings=len(learning), source_utterances=len(wanted),
        types=len(output), native_type_count_histogram=dict(sorted(multiplicity.items())),
        cooccurrence=[dict(types=list(pair), recordings=count) for pair, count in sorted(cooccurrence.items())],
        per_type=output, records_sha256=sha(records_path),
        scope='Complete learning native annotation counts; examples and omission are not semantic supervision admission. No selection/test examples, audio listening or training.')
    (a.output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('recordings','source_utterances','types','native_type_count_histogram')}))


if __name__ == '__main__':
    main()
