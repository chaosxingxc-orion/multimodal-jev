"""Versioned official-clean intent admission, separate from reviewed pilot gold.

This path does not relax slurp_training.validate_admission or alter old releases.
Only TRAIN learning may optimize; TRAIN selection is an internal development set.
Unknown slot negatives and unreviewed discriminative questions are not projected.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile

from .baseline import file_hash
from .consumption import canonical
from .slurp_training import content_hash
from .slurp_intent_structure import compile_intent
from .span_candidate_intervention import intervene

SCHEMA = 'slurp-official-clean-intent-v1'
VIEWS = ('direct', 'structured', 'scenario', 'action')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def pending_groups(packet_path, audit_path, role):
    packet, audit = read(packet_path), read(audit_path)
    if file_hash(packet_path) != audit['packet_sha256']:
        raise ValueError('pending audit packet changed')
    if role == 'learning':
        indices = {int(i) for i in audit['pending']}
        approved = {int(i) for i in audit['reviewed']}
        if indices & approved or indices | approved != set(range(len(packet))):
            raise ValueError('complete learning reference audit required')
    elif role == 'selection':
        if len(audit['records']) != len(packet):
            raise ValueError('complete selection reference audit required')
        indices = set()
        for i, (record, item) in enumerate(zip(audit['records'], packet, strict=True)):
            if (record['index'] != i or record['id'] != item['request']['id']
                    or record['source_group'] != item['request']['source_group']
                    or record['record_sha256'] != content_hash(item)
                    or record['status'] not in {'pending', 'reference_consistent'}):
                raise ValueError('selection review binding differs')
            if record['status'] == 'pending':
                indices.add(i)
    else:
        raise ValueError('known source role required')
    if any(item['request']['role'] != role for item in packet):
        raise ValueError('review role differs')
    return {packet[i]['request']['source_group'] for i in indices}


def source_population(source, receipt, task_path, reviews):
    """Authenticate complete original population before applying new masks."""
    source = Path(source).resolve()
    manifest, proof, task = read(source/'MANIFEST.json'), read(receipt), read(task_path)['intent']
    if (manifest['schema'] != 'slurp-clean-source-bundles-v1'
            or proof['release'] != str(source)
            or proof['manifest_sha256'] != file_hash(source/'MANIFEST.json')
            or proof['all_native_targets_reconstructed_independently'] is not True
            or proof['cross_role_source_group_or_audio_overlap'] != 0):
        raise ValueError('independently verified clean TRAIN source required')
    # Validate complete canonical schema even for populations with missing classes.
    compile_intent('/input-validation-only.wav', task, 'Q')
    pair_to_intent = {tuple(v): k for k, v in task['candidate_pairs'].items()}
    result = {}; group_roles = {}; hash_roles = {}; ids = set()
    for role in ('learning', 'selection'):
        for suffix in ('requests', 'targets'):
            name = role+'.'+suffix+'.jsonl'
            if file_hash(source/name) != manifest['files'][name]:
                raise ValueError('source population changed')
        requests = rows(source/(role+'.requests.jsonl'))
        labels = rows(source/(role+'.targets.jsonl'))
        targets = {t['id']: t for t in labels}
        if (len(targets) != len(labels) or len(requests) != manifest['counts'][role]
                or len(labels) != len(requests) or {r['id'] for r in requests} != targets.keys()):
            raise ValueError('complete unique source population required')
        packet_path, audit_path = reviews[role]
        pending = pending_groups(packet_path, audit_path, role)
        if not pending <= {r['source_group'] for r in requests}:
            raise ValueError('pending group outside source role')
        result[role] = []
        for request in requests:
            if (set(request) != {'id', 'official_split', 'role', 'source_group', 'audio_sha256', 'input'}
                    or set(request['input']) != {'audio'} or request['official_split'] != 'train'
                    or request['role'] != role or request['id'] in ids):
                raise ValueError('unique label-free TRAIN request required')
            ids.add(request['id'])
            for value, index in ((request['source_group'], group_roles), (request['audio_sha256'], hash_roles)):
                if index.setdefault(value, role) != role:
                    raise ValueError('cross-role source/audio leakage')
            audio = Path(request['input']['audio'])
            if not audio.is_absolute() or request['id'] != 'slurp:'+audio.name:
                raise ValueError('audio identity/path mismatch')
            target = targets[request['id']]
            if (target['request_sha256'] != content_hash(request)
                    or target['annotation_status'] != 'available'):
                raise ValueError('native source target binding differs')
            native = target['native_target']
            intent = pair_to_intent[(native['scenario'], native['action'])]
            result[role].append(dict(id=request['id'], official_split='train', role=role,
                source_group=request['source_group'], audio=str(audio), audio_sha256=request['audio_sha256'],
                source_request_sha256=content_hash(request), source_target_sha256=content_hash(target),
                intent=intent, scenario=native['scenario'], action=native['action'],
                known_pending_source=request['source_group'] in pending))
    return result, task


def inspect_audio(row):
    """Technical signal checks, never semantic listening certification."""
    import hashlib
    import numpy as np
    import soundfile as sf
    if file_hash(row['audio']) != row['audio_sha256']:
        raise ValueError('waveform file differs from source')
    wave, rate = sf.read(row['audio'], dtype='float32', always_2d=True)
    if rate != 16000 or wave.shape[1] != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('finite mono 16kHz audio required')
    peak = float(np.abs(wave).max())
    if peak == 0:
        raise ValueError('all-zero audio')
    return dict(samples=len(wave), seconds=len(wave)/rate, sample_rate=rate, peak=peak,
                clipping_fraction=float(np.mean(np.abs(wave) >= .999)),
                rms=float(np.sqrt(np.mean(wave.astype(np.float64)**2))),
                waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest())


def publish(config, output, progress=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('immutable intent publication exists')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('publication dependency changed: '+path)
    expected_paths = {str(Path(config['source'])/'MANIFEST.json'), config['receipt'], config['tasks']}
    expected_paths.update(p for pair in config['reviews'].values() for p in pair)
    if not expected_paths <= config['pins'].keys():
        raise ValueError('pin every source, task and review dependency')
    population, task = source_population(config['source'], config['receipt'], config['tasks'], config['reviews'])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.official-intent-', dir=output.parent))
    counts = {}; waveform_roles = {}; quarantine = []; total = 0
    try:
        for role, original in population.items():
            admitted = []; asset_rows = []
            candidates = []
            for row in original:
                if row['known_pending_source']:
                    quarantine.append(dict(id=row['id'], role=role, source_group=row['source_group'], reason='known_pending_source'))
                else:
                    candidates.append(row)
            # Bounded threads; deterministic order and no repeated rereads per view.
            def check(row):
                try:
                    return inspect_audio(row), None
                except Exception as exc:
                    return None, repr(exc)
            with ThreadPoolExecutor(max_workers=4) as pool:
                for row, (asset, error) in zip(candidates, pool.map(check, candidates), strict=True):
                    total += 1
                    if error:
                        quarantine.append(dict(id=row['id'], role=role, source_group=row['source_group'],
                                               reason='invalid_audio_asset', error=error))
                    else:
                        h = asset['waveform_sha256']
                        if waveform_roles.setdefault(h, role) != role:
                            raise ValueError('decoded audio leaks across learning/selection')
                        admitted.append(row)
                        asset_rows.append(dict(id=row['id'], audio_sha256=row['audio_sha256'], **asset))
                    if progress and total % 500 == 0:
                        progress(dict(phase='audio_assets', role=role, checked=total, admitted=len(admitted)))
            if not admitted:
                raise ValueError('empty admitted population')
            for name, data in ((role+'.jsonl', admitted), (role+'.audio.jsonl', asset_rows)):
                with (stage/name).open('wb') as stream:
                    for row in data:
                        stream.write(canonical(row))
            counts[role] = dict(recordings=len(admitted), source_groups=len({r['source_group'] for r in admitted}),
                                classes=dict(sorted(Counter(r['intent'] for r in admitted).items())),
                                audio_hours=sum(r['seconds'] for r in asset_rows)/3600)
        shutil.copyfile(config['tasks'], stage/'tasks.json')
        (stage/'quarantine.json').write_bytes(canonical(quarantine))
        (stage/'publication-config.json').write_bytes(canonical(config))
        manifest = dict(schema=SCHEMA, source=config['source'], source_receipt=config['receipt'],
            counts=counts, quarantine=dict(Counter(r['reason'] for r in quarantine)),
            label_basis='official_clean_v7', individual_semantic_review=False, audio_semantics_reviewed=False,
            allowed_optimization_role='learning', allowed_evaluation_role='selection',
            runtime_preflight_passed=False, supported_views=list(VIEWS),
            files={p.name: file_hash(p) for p in sorted(stage.iterdir())},
            implementation_sha256=file_hash(__file__),
            scope='Official-clean intent and canonical component supervision; not reviewed slot labels, '
                  'not an official test split, not NPU/processor or efficacy acceptance.')
        (stage/'MANIFEST.json').write_bytes(canonical(manifest))
        os.rename(stage, output)
        return manifest
    except BaseException:
        shutil.rmtree(stage)
        raise


def load(root, manifest_sha256, role, *, purpose):
    """Validate once per worker lifecycle, not once per minibatch."""
    root = Path(root)
    if (role, purpose) not in {('learning', 'optimize'), ('selection', 'evaluate')}:
        raise ValueError('selection cannot optimize; explicit role/purpose required')
    if file_hash(root/'MANIFEST.json') != manifest_sha256:
        raise ValueError('release identity changed')
    manifest = read(root/'MANIFEST.json')
    if manifest['schema'] != SCHEMA or manifest['label_basis'] != 'official_clean_v7':
        raise ValueError('explicit official-clean protocol required')
    if manifest['individual_semantic_review'] is not False:
        raise ValueError('official labels cannot claim individual review')
    for name, digest in manifest['files'].items():
        if Path(name).name != name or file_hash(root/name) != digest:
            raise ValueError('release asset changed')
    config = read(root/'publication-config.json')
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('original dependency changed')
    source, task = source_population(config['source'], config['receipt'], config['tasks'], config['reviews'])
    expected = {row['id']: row for row in source[role] if not row['known_pending_source']}
    data = rows(root/(role+'.jsonl')); assets = rows(root/(role+'.audio.jsonl'))
    quarantine = read(root/'quarantine.json')
    invalid = {r['id'] for r in quarantine if r['role'] == role and r['reason'] == 'invalid_audio_asset'}
    if (len({r['id'] for r in data}) != len(data) or {r['id'] for r in data} != expected.keys()-invalid
            or len(data) != manifest['counts'][role]['recordings'] or len(assets) != len(data)):
        raise ValueError('admitted population differs from full source')
    for row, asset in zip(data, assets, strict=True):
        if row != expected[row['id']] or asset['id'] != row['id'] or asset['audio_sha256'] != row['audio_sha256']:
            raise ValueError('native labels or audio admission changed')
    if read(root/'tasks.json') != read(config['tasks']):
        raise ValueError('task definition differs')
    return data, task, manifest


def example(row, task, view, variant='native'):
    """Compile all input before mapping labels. Caller must use authenticated load.

    View is supplied by fixed experiment/epoch configuration, never gold routing.
    Current intervention changes both candidate order and symbol binding.
    """
    if view not in VIEWS:
        raise ValueError('registered instruction view required')
    request = compile_intent(row['audio'], task, 'D' if view == 'direct' else 'Q')
    if view in ('scenario', 'action'):
        selected = tuple(q for q in request.questions if q.id == view)
        request = replace(request, questions=selected, final_id=view)
    request, binding = intervene(request, variant)
    values = {k: row[k] for k in ('intent', 'scenario', 'action')}
    target = dict(id=row['id'], targets=[q.values.index(values[q.id]) for q in request.questions],
                  target_values=[values[q.id] for q in request.questions],
                  supervised=[True]*len(request.questions),
                  span_weights=[1. if q.id == request.final_id else .1 for q in request.questions],
                  request_weight=1., label_basis='official_clean_v7',
                  source_group=row['source_group'], view=view, variant=variant)
    return request, target, binding
