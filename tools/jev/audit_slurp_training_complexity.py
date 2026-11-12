#!/usr/bin/env python3
"""Audit complete clean TRAIN learning population without reading TEST labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_complexity import ontology_strata, surface_flags


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--learning-packet', type=Path, required=True)
    p.add_argument('--learning-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('immutable analysis output already exists')
    manifest = json.loads((a.source / 'MANIFEST.json').read_text())
    if manifest['schema'] != 'slurp-clean-source-bundles-v1':
        raise ValueError('clean source bundles required')
    for name in ('learning.requests.jsonl', 'learning.targets.jsonl'):
        if file_hash(a.source / name) != manifest['files'][name]:
            raise ValueError('source hash changed')
    read = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
    requests = read(a.source / 'learning.requests.jsonl')
    targets = {x['id']: x for x in read(a.source / 'learning.targets.jsonl')}
    pairs = json.loads(a.tasks.read_text())['intent']['candidate_pairs']
    strata = ontology_strata(pairs)
    reverse = {tuple(v): k for k, v in pairs.items()}
    audit = json.loads(a.learning_audit.read_text())
    if file_hash(a.learning_packet) != audit['packet_sha256']:
        raise ValueError('review packet changed')
    packet = json.loads(a.learning_packet.read_text())
    pending_groups = {packet[int(i)]['request']['source_group'] for i in audit['pending']}
    metadata_path = a.source.parent / 'metadata/train.jsonl'
    if file_hash(metadata_path) != manifest['identity']['input_hashes']['clean_metadata_sha256']:
        raise ValueError('clean TRAIN metadata changed')
    # Reference text is accessed only for source IDs belonging to learning.
    wanted = {s for t in targets.values() for s in t['source_ids']}
    metadata = {str(x['slurp_id']): x for x in read(metadata_path) if str(x['slurp_id']) in wanted}
    if set(metadata) != wanted:
        raise ValueError('missing learning reference')
    counts, eligible, flags, source_counts = Counter(), Counter(), Counter(), Counter()
    rows = []
    group_labels = {}
    for request in requests:
        if request['role'] != 'learning' or request['official_split'] != 'train':
            raise ValueError('learning TRAIN only')
        target = targets[request['id']]
        native = target['native_target']
        label = reverse[(native['scenario'], native['action'])]
        s = strata[label]
        counts[label] += 1
        blocked = request['source_group'] in pending_groups
        if not blocked:
            eligible[label] += 1
        group_labels.setdefault(request['source_group'], set()).add(label)
        rows.append(dict(id=request['id'], source_group=request['source_group'], intent=label,
                         **s, known_pending_source=blocked,
                         acoustic_difficulty='unmeasured', empirical_difficulty='unmeasured'))
    for x in metadata.values():
        label = reverse[(x['scenario'], x['action'])]
        source_counts[label] += 1
        flags.update(k for k, v in surface_flags(x['sentence'], [e['type'] for e in x['entities']]).items()
                     if isinstance(v, bool) and v)
    if len(rows) != manifest['counts']['learning'] or len(targets) != len(rows):
        raise ValueError('population differs')
    summary = dict(schema='slurp-learning-complexity-v1',
        label_basis='official_clean_v7_not_full_semantic_review',
        recordings=len(rows), source_groups=len(group_labels), source_utterances=len(metadata),
        classes=len(counts), ontology_classes=len(pairs),
        ontology_class_strata=dict(Counter(str(v['minimum_components']) for v in strata.values())),
        recording_strata=dict(Counter(str(r['minimum_components']) for r in rows)),
        source_utterance_strata=dict(Counter({str(n): sum(v for k, v in source_counts.items()
            if strata[k]['minimum_components'] == n) for n in (1, 2)})),
        known_pending_groups=len(pending_groups),
        recordings_after_known_pending_mask=sum(eligible.values()),
        classes_after_known_pending_mask=len(eligible),
        class_count_min=min(counts.values()), class_count_max=max(counts.values()),
        class_counts=dict(sorted(counts.items())),
        conflicting_label_source_groups=sum(len(v) > 1 for v in group_labels.values()),
        source_text_triage_flags=dict(flags),
        joint_disambiguation_classes=[k for k, v in strata.items() if v['minimum_components'] == 2],
        pins={str(f): file_hash(f) for f in [a.source/'MANIFEST.json', metadata_path,
            a.tasks, a.learning_packet, a.learning_audit, Path(__file__),
            Path(__file__).resolve().parents[2]/'src/multimodal_jev/slurp_complexity.py']},
        scope='Learning-only ontology and reference triage. No audio listening, training admission, '
              'test label access, sample correctness or model efficacy claim.')
    a.output.mkdir(parents=True)
    (a.output/'records.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows))
    summary['records_sha256'] = file_hash(a.output/'records.jsonl')
    (a.output/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('pins', 'class_counts')}, indent=2))


if __name__ == '__main__':
    main()
