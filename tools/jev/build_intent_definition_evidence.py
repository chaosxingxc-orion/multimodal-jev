#!/usr/bin/env python3
"""Build a deterministic TRAIN-learning reference packet; never admit labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.release
    if digest(root/'MANIFEST.json') != args.manifest_sha256:
        raise ValueError('release identity mismatch')
    manifest = read(root/'MANIFEST.json')
    if manifest['schema'] != 'slurp-official-clean-intent-v1':
        raise ValueError('official intent release required')
    pins = {str(root/'MANIFEST.json'): args.manifest_sha256}
    for name in ('learning.jsonl', 'tasks.json'):
        pins[str(root/name)] = manifest['files'][name]
    source = Path(manifest['source'])
    source_manifest = read(source/'MANIFEST.json')
    pins[str(source/'MANIFEST.json')] = digest(source/'MANIFEST.json')
    pins[str(source/'learning.targets.jsonl')] = source_manifest['files']['learning.targets.jsonl']
    metadata_path = source.parent/'metadata/train.jsonl'
    pins[str(metadata_path)] = source_manifest['identity']['input_hashes']['clean_metadata_sha256']
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError(f'changed source: {path}')
    learning = rows(root/'learning.jsonl')
    targets = {r['id']: r for r in rows(source/'learning.targets.jsonl')}
    pairs = read(root/'tasks.json')['intent']['candidate_pairs']
    counts = Counter(r['intent'] for r in learning)
    if dict(counts) != manifest['counts']['learning']['classes']:
        raise ValueError('class population mismatch')
    refs, groups = defaultdict(set), defaultdict(set)
    for row in learning:
        if row['role'] != 'learning' or row['official_split'] != 'train' or row['known_pending_source']:
            raise ValueError('only admitted learning TRAIN sources allowed')
        target = targets[row['id']]
        target_hash = hashlib.sha256((json.dumps(target, sort_keys=True, ensure_ascii=False)+'\n').encode()).hexdigest()
        if target_hash != row['source_target_sha256']:
            raise ValueError('source target identity mismatch')
        native = target['native_target']
        if pairs[row['intent']] != [native['scenario'], native['action']]:
            raise ValueError('native label mismatch')
        refs[row['intent']].update(str(s) for s in target['source_ids'])
        groups[row['intent']].add(row['source_group'])
    wanted = set().union(*refs.values())
    # Parse TRAIN metadata, but retain/access semantic content only for learning IDs.
    metadata = {str(r['slurp_id']): r for r in rows(metadata_path) if str(r['slurp_id']) in wanted}
    if set(metadata) != wanted:
        raise ValueError('missing learning source ID')
    cards = []
    for label in sorted(pairs):
        ids = sorted(refs[label], key=lambda s: hashlib.sha256(('intent-card-v1:'+s).encode()).hexdigest())[:5]
        examples = []
        for sid in ids:
            r = metadata[sid]
            if [r['scenario'], r['action']] != pairs[label]:
                raise ValueError('metadata label mismatch')
            examples.append(dict(source_id=sid, reference=r['sentence'], native_intent=r['intent']))
        cards.append(dict(semantic_id=label, recordings=counts[label], source_groups=len(groups[label]),
                          source_utterances=len(refs[label]), examples=examples))
    result = dict(schema='slurp-intent-definition-evidence-v1', pins=pins,
                  implementation_sha256=digest(__file__), selection='first_5_source_ids_by_sha256_intent-card-v1_colon_id',
                  recordings=len(learning), source_utterances=len(wanted), cards=cards,
                  scope='Learning reference evidence for authoring; no audio review, label admission or selection/test example access.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(classes=len(cards), examples=sum(len(c['examples']) for c in cards),
                          sha256=digest(args.output))))


if __name__ == '__main__':
    main()
