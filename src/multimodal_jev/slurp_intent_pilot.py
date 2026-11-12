"""Authenticate D/Q/P pilot assets and rederive every admitted target.

This loader is for learning only. It never upgrades pending labels or admits
an evaluation population. Compiled inputs and supervision remain separate.
"""
from collections import Counter
import json
import math
from pathlib import Path

from .baseline import file_hash
from .slurp_training import content_hash,validate_admission
from .slurp_intent_structure import training_targets
from .task_schema import JevExample


def load_pilot(root,manifest_sha256,arm,*,verify_audio=True):
    root=Path(root)
    if arm not in ('D','Q','P'):raise ValueError('fixed D/Q/P arm required')
    if file_hash(root/'MANIFEST.json')!=manifest_sha256:raise ValueError('pilot manifest changed')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if manifest['schema']!='slurp-intent-dqp-pilot-v1':raise ValueError('intent pilot protocol required')
    if file_hash(Path(__file__).with_name('slurp_intent_structure.py'))!=manifest['implementation_sha256']:
        raise ValueError('pilot compiler changed')
    expected_files={f'{a}-{kind}.json' for a in ('D','Q','P') for kind in ('requests','targets')}
    if set(manifest['files'])!=expected_files:raise ValueError('complete fixed pilot arms required')
    for name,digest in manifest['files'].items():
        if file_hash(root/name)!=digest:raise ValueError('pilot asset changed: '+name)
    if (root/'Q-requests.json').read_bytes()!=(root/'P-requests.json').read_bytes():
        raise ValueError('Q/P model inputs differ')
    release=Path(manifest['release'])
    if file_hash(release/'MANIFEST.json')!=manifest['release_manifest_sha256']:raise ValueError('admitted release changed')
    release_manifest=json.loads((release/'MANIFEST.json').read_text())
    if release_manifest['schema']!='slurp-reviewed-training-release-v1':raise ValueError('reviewed source required')
    for name,digest in release_manifest['files'].items():
        if file_hash(release/name)!=digest:raise ValueError('admitted source file changed')
    ledger=json.loads((release/'review-ledger.json').read_text())
    for path,digest in ledger['evidence_files'].items():
        if not Path(path).is_absolute() or file_hash(path)!=digest:raise ValueError('review evidence changed')
    task=json.loads((release/'tasks.json').read_text())['intent']
    examples=[JevExample.from_record(json.loads(line)) for line in (release/'train.jsonl').read_text().splitlines()]
    if not examples or len({e.example_id for e in examples})!=len(examples):raise ValueError('distinct admitted examples required')
    if len({e.metadata['source_group'] for e in examples})!=len(examples):raise ValueError('pilot requires unique source groups')
    counts=Counter(e.target for e in examples)
    requests=json.loads((root/f'{arm}-requests.json').read_text());targets=json.loads((root/f'{arm}-targets.json').read_text())
    if len(requests)!=len(examples) or len(targets)!=len(examples):raise ValueError('complete admitted population required')
    items=[]
    for e,row,target in zip(examples,requests,targets,strict=True):
        validate_admission(e)
        expected_request,expected_target=training_targets(e,task,arm)
        expected_row=dict(id=e.example_id,request=expected_request.model_input())
        expected_target.update(id=e.example_id,request_sha256=content_hash(expected_row),request_weight=len(examples)/(len(counts)*counts[e.target]))
        if content_hash(row)!=content_hash(expected_row) or content_hash(target)!=content_hash(expected_target):
            raise ValueError('request or credit differs from admitted source')
        if verify_audio and file_hash(e.audio)!=e.metadata['audio_sha256']:raise ValueError('reviewed audio changed')
        items.append(dict(id=e.example_id,request=expected_request,target=target,audio_sha256=e.metadata['audio_sha256']))
    if not math.isclose(sum(i['target']['request_weight'] for i in items),len(items),rel_tol=1e-12):
        raise ValueError('global class weight differs')
    return items,dict(recordings=len(items),source_groups=len(items),covered_classes=len(counts),
        candidate_intents=len(task['candidate_pairs']),arm=arm,manifest_sha256=manifest_sha256,
        audio_verified=verify_audio,review_scope=release_manifest['review_scope'])
