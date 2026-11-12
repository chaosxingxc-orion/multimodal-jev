#!/usr/bin/env python3
"""Offline complexity and question support audit of authenticated TRAIN learning."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_complexity import ontology_strata, question_pack_profile, surface_flags
from multimodal_jev.slurp_official_training import load, read, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('analysis output is immutable')
    population, task, manifest = load(args.release, args.manifest_sha256, 'learning', purpose='optimize')
    strata = ontology_strata(task['candidate_pairs'])
    counts = Counter({k: 0 for k in strata})
    counts.update(r['intent'] for r in population)
    groups = defaultdict(set)
    assets = {r['id']: r for r in rows(args.release/'learning.audio.jsonl')}
    source = Path(manifest['source'])
    source_manifest = read(source/'MANIFEST.json')
    target_path = source/'learning.targets.jsonl'
    metadata_path = source.parent/'metadata/train.jsonl'
    if (file_hash(target_path) != source_manifest['files'][target_path.name]
            or file_hash(metadata_path) != source_manifest['identity']['input_hashes']['clean_metadata_sha256']):
        raise ValueError('source references changed')
    wanted_ids = {r['id'] for r in population}
    targets = {r['id']: r for r in rows(target_path) if r['id'] in wanted_ids}
    wanted_sources = {s for r in targets.values() for s in r['source_ids']}
    metadata = {str(r['slurp_id']): r for r in rows(metadata_path) if str(r['slurp_id']) in wanted_sources}
    if set(metadata) != wanted_sources or targets.keys() != wanted_ids:
        raise ValueError('incomplete reference binding')
    flags = {k: surface_flags(r['sentence'], [e['type'] for e in r['entities']]) for k, r in metadata.items()}
    records = []
    for row in population:
        groups[row['intent']].add(row['source_group'])
        refs = [flags[s] for s in targets[row['id']]['source_ids']]
        # A union is triage across linked references, never a semantic label.
        triggers = sorted({k for f in refs for k, v in f.items() if isinstance(v, bool) and v})
        asset = assets[row['id']]
        records.append(dict(id=row['id'], source_group=row['source_group'], intent=row['intent'],
            minimum_components=strata[row['intent']]['minimum_components'],
            reference_triage_flags=triggers, seconds=asset['seconds'], rms=asset['rms'],
            clipping_fraction=asset['clipping_fraction'],
            semantic_complexity='not_individually_reviewed', acoustic_difficulty='unrated',
            empirical_difficulty='unmeasured', use='offline_analysis_only_not_input_routing'))
    question_config = read(args.questions)
    report = dict(schema='slurp-admitted-question-support-v1',
        release=str(args.release), manifest_sha256=args.manifest_sha256,
        recordings=len(population), source_groups=len({r['source_group'] for r in population}),
        source_utterances=len(metadata), label_basis=manifest['label_basis'],
        ontology_class_strata=dict(Counter(str(v['minimum_components']) for v in strata.values())),
        recording_strata=dict(Counter(str(r['minimum_components']) for r in records)),
        class_counts=dict(counts), class_source_groups={k: len(groups[k]) for k in counts},
        reference_triage_counts=dict(Counter(k for f in flags.values() for k, v in f.items()
                                            if isinstance(v, bool) and v)),
        recording_triage_counts=dict(Counter(k for r in records for k in r['reference_triage_flags'])),
        duration_quantiles_seconds=dict(zip(('min','p50','p95','max'),
            map(float, np.quantile([r['seconds'] for r in records], [0,.5,.95,1])))),
        recordings_with_clipping=sum(r['clipping_fraction'] > 0 for r in records),
        question_packs=question_pack_profile(question_config, counts),
        scope='TRAIN learning only. Ontology partitions and reference triage, not independently '
              'annotated reasoning, input routing, question admission, or measured task improvement.',
        pins={str(p.resolve()): file_hash(p) for p in (args.release/'MANIFEST.json', args.questions,
            target_path, metadata_path, Path(__file__),
            Path(__file__).resolve().parents[2]/'src/multimodal_jev/slurp_complexity.py')})
    args.output.mkdir(parents=True)
    path = args.output/'records.jsonl'
    path.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in records))
    report['records_sha256'] = file_hash(path)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('recordings','source_groups','source_utterances','recording_strata','question_packs')}, indent=2))


if __name__ == '__main__':
    main()
