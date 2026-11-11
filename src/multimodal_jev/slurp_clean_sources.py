"""Authenticated clean TRAIN source bundles, before any Jev task projection.

No type-negative labels or actor questions are synthesized here. The release
keeps the quality hold: native annotation availability is not semantic approval.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections import Counter

from .baseline import file_hash
from .consumption import canonical
from .slurp_clean_split import partition
from .slurp_native import native_target


def assemble(metadata, requests, split):
    """Revalidate the entire source closure, then join only clean recordings."""
    fresh = partition(metadata, requests)
    for key, expected in fresh.items():
        if split.get(key) != expected:
            raise ValueError('source partition mismatch: ' + key)
    by_file = {}
    for row in metadata:
        for recording in row['recordings']:
            by_file.setdefault(recording['file'], []).append(row)
    indexed = {r['id']: r for r in requests}
    bundles, targets, quarantine = [], [], []
    for record in split['records']:
        uid = record['id']
        if record['supervision_status'] != 'available':
            quarantine.append(record)
            continue
        req = indexed[uid]
        if set(req) != {'id', 'split', 'source_group', 'audio_sha256', 'input'}:
            raise ValueError('unexpected native request fields')
        if set(req['input']) != {'audio', 'schema_sha256'}:
            raise ValueError('native input must contain audio and schema reference only')
        audio = req['input']['audio']
        if not isinstance(audio, str) or not Path(audio).is_absolute() or Path(audio).name != uid.removeprefix('slurp:'):
            raise ValueError('native audio path mismatch')
        rows = by_file[uid.removeprefix('slurp:')]
        values = [native_target(row) for row in rows]
        if any(value != values[0] for value in values[1:]):
            raise ValueError('unquarantined native target conflict')
        bundle = dict(id=uid, official_split='train', role=record['role'],
                      source_group=record['source_group'], audio_sha256=record['audio_sha256'],
                      input={'audio': audio})
        import hashlib
        target = dict(id=uid, request_sha256=hashlib.sha256(canonical(bundle)).hexdigest(),
                      native_target=values[0], source_ids=record['source_ids'],
                      raw_intent_aliases=sorted({row['intent'] for row in rows}),
                      annotation_status='available', semantic_review_status='pending',
                      optimization_allowed=False)
        bundles.append(bundle)
        targets.append(target)
    taxonomy = dict(
        scenario_action_pairs=[list(p) for p in sorted({(r['scenario'], r['action']) for r in metadata})],
        raw_intent_names=sorted({r['intent'] for r in metadata}),
        entity_types=sorted({e['type'] for r in metadata for e in r['entities']}),
        intent_alias_map=[dict(scenario=s, action=a, raw_names=sorted({r['intent'] for r in metadata
                          if (r['scenario'], r['action']) == (s, a)}))
                          for s, a in sorted({(r['scenario'], r['action']) for r in metadata})],
        scope='TRAIN inventory, not approved semantic definitions or a Jev task bank')
    return bundles, targets, quarantine, taxonomy


def publish(clean, native, split_root, evidence, output):
    """Use the separately verified split receipt as the content-hash anchor."""
    clean, native, split_root, evidence, output = map(Path, (clean, native, split_root, evidence, output))
    if output.exists():
        raise FileExistsError('preserve immutable source release')
    approved = json.loads(evidence.read_text())
    pins = approved['identity']
    paths = dict(clean_metadata_sha256=clean/'metadata/train.jsonl',
                 alignment_sha256=clean/'ALIGNMENT-RECEIPT.json',
                 native_manifest_sha256=native/'MANIFEST.json',
                 requests_sha256=native/'train.requests.jsonl')
    for key, path in paths.items():
        if file_hash(path) != pins[key]:
            raise ValueError('authenticated input changed: ' + key)
    split_path = split_root/'source-split.json'
    if file_hash(split_path) != approved['split_sha256']:
        raise ValueError('verified source split changed')
    split = json.loads(split_path.read_text())
    if split['identity'] != pins:
        raise ValueError('source split provenance mismatch')
    metadata = [json.loads(line) for line in paths['clean_metadata_sha256'].read_text().splitlines()]
    requests = [json.loads(line) for line in paths['requests_sha256'].read_text().splitlines()]
    bundles, targets, quarantine, taxonomy = assemble(metadata, requests, split)
    if len(bundles) + len(quarantine) != approved['recordings'] or len(quarantine) != approved['quarantined']:
        raise ValueError('verified population mismatch')
    counts = Counter(b['role'] for b in bundles)
    for role in ('learning', 'selection'):
        if counts[role] != approved['counts'][role]['available']:
            raise ValueError('verified role count mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.source-stage-', dir=output.parent))
    try:
        for role in ('learning', 'selection'):
            selected = [(b, t) for b, t in zip(bundles, targets, strict=True) if b['role'] == role]
            for suffix, index in (('requests', 0), ('targets', 1)):
                path = stage/f'{role}.{suffix}.jsonl'
                with path.open('wb') as handle:
                    for pair in selected:
                        handle.write(canonical(pair[index]))
                path.chmod(0o600 if suffix == 'targets' else 0o644)
        (stage/'quarantine.json').write_bytes(canonical(quarantine))
        (stage/'taxonomy-inventory.json').write_bytes(canonical(taxonomy))
        manifest = dict(schema='slurp-clean-source-bundles-v1', counts=dict(counts),
            quarantined=len(quarantine), optimization_ready=False, test_opened=False,
            identity=dict(input_hashes=pins, split_sha256=file_hash(split_path),
                          verification_receipt_sha256=file_hash(evidence),
                          implementation_sha256=file_hash(__file__)),
            files={p.name:file_hash(p) for p in sorted(stage.iterdir())},
            source_root=str(clean), native_root=str(native),
            scope='Authenticated clean TRAIN source requests/targets only; no Jev labels, reviewed type negatives, training permission, audio byte rehash, or model execution.')
        (stage/'MANIFEST.json').write_bytes(canonical(manifest))
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return manifest
