#!/usr/bin/env python3
"""Build a content-addressed FSC Jev dataset, preserving official splits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

from multimodal_jev.task_schema import Candidate, DecisionSchema, JevExample


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True) + '\n').encode()


def build(root: Path, destination: Path, config: dict):
    raw = {s: (root / 'data' / f'{s}_data.csv').read_bytes() for s in ('train', 'valid', 'test')}
    source_hashes = {s: digest(v) for s, v in raw.items()}
    builder_sha = digest(Path(__file__).read_bytes())
    identity = digest(encoded({'sources': source_hashes, 'config': config, 'builder': builder_sha}))
    output = destination / identity
    if output.exists():
        raise FileExistsError(f'immutable release already exists: {output}')
    rows = {s: list(csv.DictReader(data.decode().splitlines())) for s, data in raw.items()}
    columns = ('action', 'object', 'location')
    # Candidates derive exclusively from TRAIN; test labels never define prompts.
    intents = sorted({tuple(r[c] for c in columns) for r in rows['train']})
    keys = {v: f'c{i:02d}' for i, v in enumerate(intents)}
    candidates = tuple(Candidate(keys[v], '; '.join(f'{c}={x}' for c, x in zip(columns, v))) for v in intents)
    schema = DecisionSchema('fsc.intent.v1', 'choice', 'Which complete command does the speaker request? Values none mean unspecified in the original label.', candidates)
    seen_paths, seen_speakers = {}, {}
    seen_audio_hashes = {}
    stats = {}
    destination.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.building-', dir=destination))
    try:
        for split, samples in rows.items():
            target = stage / f'{split}.jsonl'
            variants = 0
            with target.open('wb') as handle:
                for row in samples:
                    rel, speaker = row['path'], row['speakerId']
                    for registry, key in ((seen_paths, rel), (seen_speakers, speaker)):
                        if key in registry and registry[key] != split:
                            raise ValueError(f'cross-split leakage: {key}')
                        registry[key] = split
                    audio = (root / rel).resolve()
                    if not audio.is_relative_to(root.resolve()):
                        raise ValueError(f'audio outside dataset: {audio}')
                    audio_hash = digest(audio.read_bytes())
                    if audio_hash in seen_audio_hashes and seen_audio_hashes[audio_hash] != split:
                        raise ValueError(f'cross-split duplicate audio: {rel}')
                    seen_audio_hashes[audio_hash] = split
                    group = 'fsc:' + digest(rel.encode())[:24]
                    label = tuple(row[c] for c in columns)
                    if label not in keys:
                        raise ValueError(f'unseen native intent: {label}')
                    metadata = {'source_group': group, 'speaker_id': speaker, 'audio_sha256': audio_hash,
                                'transcript_reference': row['transcription'], 'native_labels': dict(zip(columns, label)),
                                'protocol': config['protocol'], 'family': 'command-routing'}
                    value = keys[label]
                    # All transformations remain in the source recording's split.
                    views = [('native', schema, value)]
                    if 'permutation' in config['variants']:
                        perm = tuple(sorted(candidates, key=lambda c: digest((group + c.id).encode())))
                        views.append(('permutation', DecisionSchema(schema.id, 'choice', schema.question, perm), value))
                    if 'rebind' in config['variants']:
                        remap = {c.id: candidates[(i + 1) % len(candidates)].id for i, c in enumerate(candidates)}
                        rebound = tuple(Candidate(remap[c.id], c.description) for c in candidates)
                        views.append(('rebind', DecisionSchema('fsc.intent.rebind.v1', 'choice', schema.question, rebound), remap[value]))
                    if 'missing_candidate' in config['variants']:
                        remaining = tuple(c for c in candidates if c.id != value)
                        views.append(('missing_candidate', DecisionSchema('fsc.intent.subset.v1', 'choice', schema.question, remaining), None))
                    if 'noul' in config['variants']:
                        # Balanced binary decisions; answer depends on question definition.
                        for offset, answer in ((0, True), (1, False)):
                            candidate = candidates[(list(keys).index(label) + offset) % len(candidates)]
                            binary = DecisionSchema('fsc.matches.v1', 'noul', f'Does the utterance request exactly this command: {candidate.description}?')
                            views.append((f'noul.{answer}', binary, answer))
                    for variant, current, answer in views:
                        example = JevExample(f'{group}:{variant}', split, str(audio), current, answer,
                                             {**metadata, 'variant': variant})
                        handle.write(encoded(example.to_record()))
                        variants += 1
            stats[split] = {'recordings': len(samples), 'decisions': variants,
                            'manifest_sha256': digest(target.read_bytes()),
                            'speakers': len({r['speakerId'] for r in samples})}
        receipt = {'release_id': identity, 'source_csv_sha256': source_hashes, 'builder_sha256': builder_sha,
                   'config': config, 'splits': stats, 'intent_count': len(intents),
                   'cross_split_audio_overlap': 0, 'cross_split_speaker_overlap': 0,
                   'notes': 'Schema-derived unknowns are not acoustic ambiguity or real OOD. Official splits preserved.'}
        (stage / 'MANIFEST.json').write_bytes(encoded(receipt))
        os.rename(stage, output)
    except BaseException:
        import shutil
        shutil.rmtree(stage)
        raise
    return output, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    output, receipt = build(args.root, args.output_root, json.loads(args.config.read_text()))
    print(json.dumps({'path': str(output), **receipt}, indent=2))


if __name__ == '__main__':
    main()
