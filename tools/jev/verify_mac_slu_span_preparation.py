#!/usr/bin/env python3
"""Independent population, input isolation and lossless source-label checks."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash


def verify(release):
    manifest = json.loads((release/'MANIFEST.json').read_text())
    if manifest['training_admitted'] is not False or manifest['split'] != 'train':
        raise ValueError('diagnostic TRAIN-only release required')
    for name, digest in manifest['files'].items():
        if file_hash(release/name) != digest: raise ValueError('release changed')
    source_path = next(Path(p) for p in manifest['source_files'] if p.endswith('/label/train_set.jsonl'))
    if file_hash(source_path) != manifest['source_files'][str(source_path)]: raise ValueError('source changed')
    source = [json.loads(line) for line in source_path.open()]
    inputs = [json.loads(line) for line in (release/'train.inputs.jsonl').open()]
    targets = [json.loads(line) for line in (release/'train.targets.jsonl').open()]
    bank = json.loads((release/'bank.json').read_text())
    if len(source) != manifest['records'] or len(source) != len(inputs) or len(source) != len(targets):
        raise ValueError('full source denominator not retained')
    known_audio, missing, represented = set(), [], 0
    for raw, request, target in zip(source, inputs, targets, strict=True):
        if raw['id'] != request['id'] or raw['id'] != target['id']: raise ValueError('row identity differs')
        if set(request) != {'id', 'split', 'audio', 'bank_id', 'expansion', 'training_admitted'}:
            raise ValueError('unexpected target/text field in model input envelope')
        if request['bank_id'] != bank['bank_id'] or request['training_admitted'] is not False:
            raise ValueError('per-record bank or admission differs')
        if target['annotation_complete'] is not False: raise ValueError('invented completeness')
        expected = []
        for instance, domains in raw['semantics'].items():
            for domain, slots in domains.items():
                expected.append((instance, domain, slots))
        actual = [(f['instance'], f['domain'], f['source_entries']) for f in target['frames']]
        if actual != expected or target['instance_order'] != list(raw['semantics']):
            raise ValueError('frame/source order, values, metadata or multiplicity lost')
        unresolved = ([] if raw['semantics'] else [dict(kind='empty_semantics', instance=None)])
        unresolved.extend(dict(kind='empty_instance', instance=k) for k,v in raw['semantics'].items() if not v)
        if target['unresolved'] != unresolved: raise ValueError('missing annotation status changed')
        for frame in target['frames']:
            if frame['slots'] != [v for v in frame['source_entries'] if v['name'] != 'intent']:
                raise ValueError('bound slot values changed')
            if [frame['intent']] != [v['value'] for v in frame['source_entries'] if v['name'] == 'intent']:
                raise ValueError('intent changed')
        official = target['official_format']
        if official['status'] == 'representable':
            if unresolved: raise ValueError('empty source silently scored')
            frames = [dict(domain=f['domain'], intent=f['intent'],
                           slots={s['name']:s['value'] for s in f['slots']}) for f in target['frames']]
            if (frames != official['semantics'] or
                    any(len(f['slots']) != len(g['slots']) for f,g in zip(target['frames'],frames,strict=True))):
                raise ValueError('official conversion discarded source slots')
            represented += 1
        elif official['status'] != 'unavailable' or official['semantics'] is not None:
            raise ValueError('unknown official target state')
        audio = request['audio']
        if audio is None:
            missing.append(raw['id'])
        else:
            path = Path(audio['path'])
            if path.parent != release/'audio' or path.name != 'id_'+raw['id']+'.wav':
                raise ValueError('audio source binding differs')
            if file_hash(path) != audio['sha256']: raise ValueError('extracted audio changed')
            known_audio.add(path.name)
    if (sorted(missing) != manifest['missing_audio_ids'] or len(known_audio) != manifest['audio_records']
            or known_audio != {p.name for p in (release/'audio').iterdir()}
            or represented != manifest['official_format_counts']['representable']):
        raise ValueError('full audio/target coverage differs')
    return dict(passed=True, records=len(source), audio_records=len(known_audio),
                missing_audio_ids=missing, official_representable=represented,
                training_admitted=False, manifest_sha256=file_hash(release/'MANIFEST.json'),
                scope='All TRAIN IDs, exact extracted audio hashes, frame multiplicity/source metadata '
                      'and input isolation independently verified. Not semantic or listening review.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--release', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = verify(a.release); atomic_json(a.output, result)
    print(json.dumps(result))
