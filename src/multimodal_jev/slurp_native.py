"""Native SLURP composite records, preserving every annotated entity occurrence."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from .baseline import atomic_json, file_hash
from .consumption import canonical
from .teacher_candidates import strict_json
from .slurp import speaker_index


def native_target(row):
    tokens = row['tokens']
    if [t['id'] for t in tokens] != list(range(len(tokens))):
        raise ValueError('official token positions must match token IDs')
    entities = []
    for entity in row['entities']:
        span = entity['span']
        if (not span or span != sorted(set(span)) or any(type(i) is not int or i < 0 or i >= len(tokens) for i in span)):
            raise ValueError('invalid native entity span')
        # Match pinned upstream util.release2prediction: lower each surface,
        # join by space, retain entity order and duplicate occurrences.
        entities.append({'type': entity['type'], 'filler': ' '.join(tokens[i]['surface'].lower() for i in span)})
    return {'scenario': row['scenario'], 'action': row['action'], 'entities': entities}


def parse_prediction(raw):
    value = strict_json(raw)
    if type(value) is not dict or set(value) != {'scenario', 'action', 'entities'}:
        raise ValueError('exact native composite output required')
    if any(type(value[k]) is not str or not value[k].strip() for k in ('scenario', 'action')):
        raise ValueError('nonempty scenario/action required')
    if type(value['entities']) is not list: raise ValueError('entity list required')
    for entity in value['entities']:
        if type(entity) is not dict or set(entity) != {'type', 'filler'}:
            raise ValueError('exact entity type/filler fields required')
        if any(type(entity[k]) is not str or not entity[k].strip() for k in ('type', 'filler')):
            raise ValueError('nonempty entity type/filler required')
    return value


def accepts(value, taxonomy):
    return ([value['scenario'], value['action']] in taxonomy['scenario_action_pairs']
            and all(e['type'] in taxonomy['entity_types'] for e in value['entities']))


def messages(request, taxonomy, evidence, instruction):
    if set(request) != {'id', 'split', 'source_group', 'audio_sha256', 'input'}:
        raise ValueError('label-free request required')
    if set(request['input']) != {'audio', 'schema_sha256'}:
        raise ValueError('audio and task definition only')
    if hashlib.sha256(canonical(taxonomy)).hexdigest() != request['input']['schema_sha256']:
        raise ValueError('task taxonomy changed')
    return [{'role': 'system', 'content': instruction + '\nTask definition:\n' + canonical(taxonomy).decode()},
            {'role': 'user', 'content': json.dumps({'transcription': evidence}, ensure_ascii=False)}]


def recording_index(rows):
    """Match official filename last-write semantics and merge overlapping utterance groups."""
    owners = {}; occurrences = {}; parent = {}
    def find(uid):
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid
    for split, items in rows.items():
        for index, row in enumerate(items):
            uid = 'slurp:' + str(row['slurp_id'])
            if uid in parent: raise ValueError('duplicate or cross-split utterance ID')
            parent[uid] = uid
            for rec_index, recording in enumerate(row['recordings']):
                filename = recording['file']
                entry = {'split': split, 'row_index': index, 'recording_index': rec_index,
                         'utterance_group': uid, 'target': native_target(row)}
                if filename in owners:
                    old = owners[filename]
                    if old['split'] != split: raise ValueError('recording filename crosses official splits')
                    a, b = sorted((find(old['utterance_group']), find(uid)))
                    parent[b] = a
                owners[filename] = entry
                occurrences.setdefault(filename, []).append(entry)
    components = {}
    for uid in parent: components.setdefault(find(uid), []).append(uid)
    group_map = {}; members = {}
    for values in components.values():
        values = sorted(values); key = values[0] if len(values) == 1 else 'slurp-linked:' + hashlib.sha256(canonical(values)).hexdigest()[:24]
        members[key] = values
        for uid in values: group_map[uid] = key
    duplicates = []
    for filename, entries in occurrences.items():
        if len(entries) > 1:
            duplicates.append({'file': filename, 'occurrences': entries,
                'kept': owners[filename], 'conflicting_targets': len({canonical(e['target']) for e in entries}) > 1,
                'source_group': group_map[owners[filename]['utterance_group']]})
    return owners, group_map, members, duplicates


def build(root, destination, state):
    root, destination, state = Path(root).resolve(), Path(destination), Path(state)
    files = {s: root / 'metadata' / (name + '.jsonl') for s, name in
             [('train', 'train'), ('valid', 'devel'), ('test', 'test')]}
    rows = {s: list(map(json.loads, p.read_text().splitlines())) for s, p in files.items()}
    metadata_path = root / 'metadata/metadata.json'
    speakers = speaker_index(json.loads(metadata_path.read_text()))
    owners, group_map, group_members, duplicates = recording_index(rows)
    taxonomy = {'schema': 'slurp-native-composite-v1',
                'scenario_action_pairs': [list(p) for p in sorted({(r['scenario'], r['action']) for r in rows['train']})],
                'entity_types': sorted({e['type'] for r in rows['train'] for e in r['entities']}),
                'output': {'scenario': 'one listed scenario', 'action': 'its paired action',
                           'entities': [{'type': 'listed entity type', 'filler': 'spoken lowercase surface words'}]},
                'entity_multiplicity': 'all occurrences, including repeated types or repeated identical fillers; empty list if none'}
    taxonomy_hash = hashlib.sha256(canonical(taxonomy)).hexdigest()
    identity = {'schema': 'slurp-native-release-v1', 'source_root': str(root),
                'source_sha256': {s: file_hash(p) for s, p in files.items()},
                'speaker_metadata_sha256': file_hash(metadata_path),
                'builder_sha256': file_hash(__file__), 'taxonomy_sha256': taxonomy_hash,
                'condition': 'all official real recordings; no transcript/domain filtering; no synthetic audio'}
    identity['duplicate_filename_policy'] = 'official scorer last occurrence per filename; overlapping utterance groups merged before any split selection'
    release_id = hashlib.sha256(canonical(identity)).hexdigest()
    destination.mkdir(parents=True, exist_ok=True); output = destination / release_id
    if output.exists(): raise FileExistsError('preserve immutable native release')
    stage = Path(tempfile.mkdtemp(prefix='.building-', dir=destination))
    (stage / 'taxonomy.json').write_bytes(canonical(taxonomy))
    seen_files, seen_audio = {}, {}
    counts = {}; start = time.monotonic(); processed = 0
    for split, items in rows.items():
        count = entities = aliases = 0
        with (stage / (split + '.requests.jsonl')).open('wb') as req, (stage / (split + '.targets.jsonl')).open('wb') as gold:
            for row_index, row in enumerate(items):
                utterance_group = 'slurp:' + str(row['slurp_id'])
                group = group_map[utterance_group]
                target = native_target(row)
                if not accepts(target, taxonomy): raise ValueError('evaluation taxonomy absent from TRAIN')
                aliases += row['intent'] != row['scenario'] + '_' + row['action']
                for recording_index_in_row, recording in enumerate(row['recordings']):
                    filename = recording['file']; path = (root / 'slurp_real' / filename).resolve()
                    owner = owners[filename]
                    if (owner['row_index'], owner['recording_index']) != (row_index, recording_index_in_row):
                        continue
                    if not path.is_relative_to(root / 'slurp_real') or filename in seen_files:
                        raise ValueError('audio path escape or duplicate recording')
                    if filename not in speakers: raise ValueError('recording speaker metadata missing')
                    digest = file_hash(path)
                    if digest in seen_audio and seen_audio[digest] != split: raise ValueError('cross-split identical audio')
                    seen_audio[digest] = split; seen_files[filename] = split
                    request = {'id': 'slurp:' + filename, 'split': split, 'source_group': group,
                               'audio_sha256': digest, 'input': {'audio': str(path), 'schema_sha256': taxonomy_hash}}
                    target_row = {'id': request['id'], 'target': target,
                                  'request_sha256': hashlib.sha256(canonical(request)).hexdigest(),
                                  'metadata': {'file': filename, 'slurp_id': row['slurp_id'],
                                               'utterance_group': utterance_group,
                                               'source_group_members': group_members[group],
                                               'speaker_id': speakers[filename],
                                               'raw_intent': row['intent'], 'native_entities': row['entities'],
                                               'native_tokens': row['tokens'], 'transcript_reference': row['sentence']}}
                    req.write(canonical(request)); gold.write(canonical(target_row))
                    count += 1; entities += len(target['entities']); processed += 1
                    if processed % 500 == 0:
                        atomic_json(state / 'progress.json', {'phase': 'hash_and_export', 'split': split,
                            'recordings': processed, 'heartbeat_unix': time.time(), 'elapsed_seconds': time.monotonic()-start})
        counts[split] = {'utterances': len(items), 'recordings': count,
                         'raw_recording_entries': sum(len(r['recordings']) for r in items),
                         'source_groups': len({group_map['slurp:'+str(r['slurp_id'])] for r in items}),
                         'entity_occurrences_recording_weighted': entities,
                         'raw_intent_alias_utterances': aliases,
                         'requests_sha256': file_hash(stage / (split + '.requests.jsonl')),
                         'targets_sha256': file_hash(stage / (split + '.targets.jsonl'))}
    receipt = {'release_id': release_id, 'identity': identity, 'splits': counts,
               'duplicate_recording_audit': duplicates,
               'cross_split_audio_hash_overlap': 0, 'elapsed_seconds': time.monotonic()-start,
               'scope': 'full native composite dataset; not derived Jev tasks, not an inference or metric result'}
    (stage / 'MANIFEST.json').write_bytes(canonical(receipt)); os.rename(stage, output)
    return output, receipt
