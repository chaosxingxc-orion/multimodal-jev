"""Export label-free Jev requests and separate evaluation provenance."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .baseline import file_hash
from .task_schema import JevExample, DecisionSchema


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False)+'\n').encode()


def request_from_example(example):
    return {'id': example.example_id, 'split': example.split,
            'source_group': example.metadata['source_group'],
            'input': example.model_input(), 'audio_sha256': example.metadata['audio_sha256']}


def validate_request(row):
    if set(row) != {'id', 'split', 'source_group', 'input', 'audio_sha256'}:
        raise ValueError('unexpected request fields; gold/provenance must remain separate')
    if set(row['input']) != {'audio', 'schema'}:
        raise ValueError('only audio and decision schema belong in runtime input')
    if row['split'] not in {'train', 'valid', 'test'} or not row['id'] or not row['source_group']:
        raise ValueError('request identity required')
    schema = row['input']['schema']
    if set(schema) != {'id','decision_type','question','candidates','allow_unknown','levels'}:
        raise ValueError('unexpected schema fields')
    DecisionSchema.from_dict(schema)
    return row


def export_release(manifests, destination, dataset_name, native_only=False):
    """Stream existing audited manifests; never copy audio or alter source splits."""
    if set(manifests) != {'train','valid','test'}:
        raise ValueError('three official split manifests required')
    destination = Path(destination)
    identity = {'dataset': dataset_name, 'native_only': native_only,
                'source_manifests': {s: {'path': str(p), 'sha256': file_hash(p)} for s,p in manifests.items()},
                'builder_sha256': file_hash(__file__), 'schema_sha256': file_hash(Path(__file__).with_name('task_schema.py'))}
    release = hashlib.sha256(canonical(identity)).hexdigest()
    output = destination/release
    if output.exists(): raise FileExistsError('immutable release already exists')
    destination.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.building-', dir=destination))
    counts = {}; groups = {}; audio_splits = {}
    for split, manifest in manifests.items():
        ids = set(); group_set = set(); count = 0
        with (stage/f'{split}.requests.jsonl').open('wb') as requests, (stage/f'{split}.targets.jsonl').open('wb') as targets:
            with Path(manifest).open() as handle:
                for line in handle:
                    example = JevExample.from_record(json.loads(line))
                    if example.split != split: raise ValueError('source split mismatch')
                    if native_only and example.metadata.get('variant') != 'native': continue
                    if example.example_id in ids: raise ValueError('duplicate request ID')
                    request = validate_request(request_from_example(example))
                    group = request['source_group']; audio = request['audio_sha256']
                    if group in groups and groups[group] != split: raise ValueError('source group crosses splits')
                    if audio in audio_splits and audio_splits[audio] != split: raise ValueError('audio crosses splits')
                    groups[group] = split; audio_splits[audio] = split
                    requests.write(canonical(request))
                    targets.write(canonical({'id':example.example_id,'target':example.target,
                        'metadata':example.metadata,'request_sha256':hashlib.sha256(canonical(request)).hexdigest()}))
                    ids.add(example.example_id); group_set.add(group); count += 1
        if not count: raise ValueError('empty release split')
        counts[split] = {'requests': count, 'source_groups': len(group_set),
                        'requests_sha256': file_hash(stage/f'{split}.requests.jsonl'),
                        'targets_sha256': file_hash(stage/f'{split}.targets.jsonl')}
    receipt = {'release_id': release, 'identity': identity, 'splits': counts,
               'contract': 'audio + typed decision definition; targets/provenance separate; generated JSON and native typed logits are different execution routes'}
    (stage/'MANIFEST.json').write_bytes(canonical(receipt))
    os.rename(stage, output)
    return output, receipt


def inference_messages(request, evidence, instruction):
    validate_request(request)
    if not isinstance(instruction, str) or not instruction.strip(): raise ValueError('instruction required')
    return [{'role':'system', 'content': instruction+'\nDecision definition:\n'+
             json.dumps(request['input']['schema'], sort_keys=True, ensure_ascii=False)},
            {'role':'user', 'content': json.dumps({'transcription': evidence}, ensure_ascii=False)}]
