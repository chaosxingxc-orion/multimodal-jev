#!/usr/bin/env python3
"""Inventory declared neighbor boundaries from authenticated learning metadata."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines()]


def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--support-report', type=Path, required=True)
    p.add_argument('--report-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.support_report) != a.report_sha256:
        raise ValueError('support report changed')
    report = read(a.support_report)
    for path, expected in report['pins'].items():
        if sha(path) != expected:
            raise ValueError(f'changed input: {path}')
    paths = [Path(x) for x in report['pins']]
    get = lambda name: next(x for x in paths if x.name == name)
    learning = rows(get('learning.jsonl'))
    targets = {x['id']: x for x in rows(get('learning.targets.jsonl'))}
    wanted = {str(s) for r in learning for s in targets[r['id']]['source_ids']}
    metadata = {str(x['slurp_id']): x for x in rows(get('train.jsonl')) if str(x['slurp_id']) in wanted}
    if len(learning) != report['recordings'] or len(metadata) != report['source_utterances']:
        raise ValueError('population mismatch')
    cards = read(get('type-definition-candidates-v1.json'))['types']
    names = {c['entity_type'] for c in cards}
    pairs = {tuple(sorted((c['entity_type'], t))) for c in cards for t in c['confusable_types']}
    if any(u == v or u not in names or v not in names for u, v in pairs):
        raise ValueError('invalid neighbor pair')
    by_type = defaultdict(set)
    by_filler = defaultdict(lambda: defaultdict(set))
    entities = {}
    for sid, r in metadata.items():
        tokens = {t['id']: t['surface'] for t in r['tokens']}
        entities[sid] = [dict(type=e['type'], filler=' '.join(tokens[i] for i in e['span']),
                             span=e['span']) for e in r['entities']]
        for e in entities[sid]:
            by_type[e['type']].add(sid)
            by_filler[e['type']][norm(e['filler'])].add(sid)

    def sample(ids, key, n=2):
        return sorted(ids, key=lambda s: hashlib.sha256((key+':'+s).encode()).hexdigest())[:n]

    def example(sid):
        r = metadata[sid]
        return dict(source_id=sid, reference=r['sentence'], native_intent=r['intent'], entities=entities[sid])

    records = []
    for u, v in sorted(pairs):
        joint = by_type[u] & by_type[v]
        shared = set(by_filler[u]) & set(by_filler[v])
        lexical = []
        for filler in sample(shared, 'boundary-filler-v1:'+u+':'+v):
            lexical.append(dict(normalized_filler=filler,
                examples_by_type={t:[example(s) for s in sample(by_filler[t][filler], 'boundary-source-v1:'+t+':'+filler)]
                                  for t in (u,v)}))
        records.append(dict(types=[u,v], coannotated_source_utterances=len(joint),
            shared_normalized_fillers=len(shared),
            coannotation_examples=[example(s) for s in sample(joint, 'boundary-coexist-v1:'+u+':'+v)],
            shared_filler_examples=lexical,
            positive_examples_by_type={t:[example(s) for s in sample(by_type[t], 'boundary-positive-v1:'+u+':'+v+':'+t)] for t in (u,v)},
            implication='Neither same wording across contexts nor coannotation establishes error, exclusive precedence, absence, or intent-slot binding.'))
    result = dict(schema='slurp-type-boundary-evidence-v1', support_report=str(a.support_report),
        support_report_sha256=a.report_sha256, implementation_sha256=sha(__file__),
        recordings=len(learning), source_utterances=len(metadata), pairs=len(records),
        pairs_with_coannotation=sum(r['coannotated_source_utterances'] > 0 for r in records),
        pairs_with_shared_filler=sum(r['shared_normalized_fillers'] > 0 for r in records),
        normalization='NFKC_casefold_whitespace_no_stemming', records=records,
        scope='Complete declared neighbor-pair counts, deterministic learning reference samples; no semantic admission, selection/test examples, audio review or relabelling.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
