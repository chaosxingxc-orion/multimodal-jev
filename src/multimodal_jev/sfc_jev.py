"""SFC Jev bundles: independent Noul pairs and typed per-function call lists.

The structured call-list decision is a project extension, not an nga primitive.
Request construction depends only on published input; gold stays separate.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .baseline import file_hash
from .consumption import canonical
from .sfc import _finite, call_multiset
from .sfc_consumption import SPLITS, restore_record, validate_request
from .sfc_definitions import definitions


def request_from_native(native):
    validate_request(native)
    parsed = definitions(native['input'])
    decisions = []
    for index, function in enumerate(parsed['functions']):
        key = f'f{index:02d}'
        common = dict(function=deepcopy(function))
        for suffix, question in (
            ('present', 'At least one call to this function is required by the current request.'),
            ('absent', 'No call to this function is required by the current request.'),
        ):
            decisions.append(dict(id=key + '.' + suffix, decision_type='noul',
                question=question, true_definition=question,
                false_definition='The stated proposition is false.', **common))
        decisions.append(dict(id=key + '.calls', decision_type='typed_call_list',
            question='Return every required call to this function as a list of argument objects. '
                     'Return [] if no call is required. Preserve repeated calls, argument types, '
                     'array order and the literal missing-value string NAN. Do not invent omitted arguments.',
            value_schema=dict(type='array', items=deepcopy(function['parameters'])),
            missing_value_policy='The literal string NAN is allowed for any parameter, as in the source benchmark.',
            **common))
    return dict(id=native['id'], split=native['split'], source_group=native['source_group'],
        audio_sha256=deepcopy(native['audio_sha256']),
        input=dict(context=deepcopy(native['input']), decisions=decisions),
        protocol='sfc-jev-bundle-v1')


def native_from_request(request):
    return dict(id=request['id'], split=request['split'], source_group=request['source_group'],
        input=deepcopy(request['input']['context']), audio_sha256=deepcopy(request['audio_sha256']))


def split_bundle(native, target):
    row = restore_record(native, target)
    request = request_from_native(native)
    answers = {}
    for decision in request['input']['decisions']:
        args = [deepcopy(c['arguments']) for c in row['supervision']['calls']
                if c['name'] == decision['function']['name']]
        key = decision['id']
        answers[key] = args if key.endswith('.calls') else (bool(args) if key.endswith('.present') else not args)
    gold = dict(id=request['id'], answers=answers, native_target=deepcopy(target),
        request_sha256=hashlib.sha256(canonical(request)).hexdigest())
    restore_bundle(request, gold)
    return request, gold


def reconstruct_calls(request, answers):
    """Reconstruct call semantics without gold. No duplicate or argument flattening."""
    if set(answers) != {d['id'] for d in request['input']['decisions']}:
        raise ValueError('complete decision answers required')
    calls = []
    for decision in request['input']['decisions']:
        key = decision['id']; value = answers[key]
        if decision['decision_type'] == 'noul':
            if type(value) is not bool:
                raise ValueError('Noul answer must be a boolean')
        else:
            if type(value) is not list or any(type(v) is not dict or not _finite(v) for v in value):
                raise ValueError('typed list of finite argument objects required')
            base = key.rsplit('.', 1)[0]
            if answers[base + '.present'] != bool(value) or answers[base + '.absent'] != (not value):
                raise ValueError('Noul/call-list consistency required')
            calls.extend(dict(name=decision['function']['name'], arguments=deepcopy(v)) for v in value)
    return calls


def restore_bundle(request, gold):
    native = native_from_request(request)
    if request != request_from_native(native):
        raise ValueError('input-only decision definitions changed')
    if (set(gold) != {'id', 'answers', 'native_target', 'request_sha256'}
            or gold['id'] != request['id']
            or gold['request_sha256'] != hashlib.sha256(canonical(request)).hexdigest()):
        raise ValueError('bundle target binding changed')
    row = restore_record(native, gold['native_target'])
    if call_multiset(reconstruct_calls(request, gold['answers'])) != call_multiset(row['supervision']['calls']):
        raise ValueError('typed decisions do not reconstruct original calls')
    return row


def export(source, destination):
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    if set(manifest['splits']) != set(SPLITS):
        raise ValueError('all official SFC splits required')
    identity = dict(dataset='sfc-jev', protocol='sfc-jev-bundle-v1', source=str(source.resolve()),
        source_manifest_sha256=file_hash(source / 'MANIFEST.json'),
        code={n: file_hash(Path(__file__).with_name(n)) for n in
              ('sfc_jev.py', 'sfc_definitions.py', 'sfc_consumption.py', 'sfc.py', 'sfc_omni.py', 'consumption.py')})
    release = hashlib.sha256(canonical(identity)).hexdigest()
    final = destination / release
    if final.exists():
        raise FileExistsError('immutable SFC-Jev release exists')
    destination.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.sfc-jev-', dir=destination))
    reports = {}; audio_splits = {}
    try:
        for split in SPLITS:
            pin = manifest['splits'][split]
            for kind in ('requests', 'targets'):
                if file_hash(source / f'{split}.{kind}.jsonl') != pin[kind + '_sha256']:
                    raise ValueError('native release changed')
            ids = set(); groups = set(); decisions = 0; two_audio = 0
            with (source / f'{split}.requests.jsonl').open() as ri, (source / f'{split}.targets.jsonl').open() as ti, \
                    (stage / f'{split}.requests.jsonl').open('wb') as ro, (stage / f'{split}.targets.jsonl').open('wb') as to:
                for rline, tline in zip(ri, ti, strict=True):
                    native, target = json.loads(rline), json.loads(tline)
                    request, gold = split_bundle(native, target)
                    if request['id'] in ids or request['split'] != split:
                        raise ValueError('duplicate or wrong-split request')
                    for digest in request['audio_sha256']:
                        if digest in audio_splits and audio_splits[digest] != split:
                            raise ValueError('audio crosses official splits')
                        audio_splits[digest] = split
                    ids.add(request['id']); groups.add(request['source_group'])
                    decisions += len(request['input']['decisions'])
                    two_audio += len(native['input']['audios']) == 2
                    ro.write(canonical(request)); to.write(canonical(gold))
            if len(ids) != pin['requests'] or len(groups) != pin['source_groups']:
                raise ValueError('full split coverage required')
            reports[split] = dict(bundles=len(ids), decisions=decisions, source_groups=len(groups),
                two_audio_bundles=two_audio, requests_sha256=file_hash(stage / f'{split}.requests.jsonl'),
                targets_sha256=file_hash(stage / f'{split}.targets.jsonl'))
        receipt = dict(release_id=release, identity=identity, splits=reports,
            scope='Data conversion only; no inference. Original supplied history and ordered audio retained. '
                  'Noul pairs plus project-specific typed call lists. Source call multiset reconstructed; '
                  'original text and cross-function order retained in separate provenance. '
                  'Audio hashes inherited from native release, not waveform-rehashed.')
        (stage / 'MANIFEST.json').write_bytes(canonical(receipt))
        os.rename(stage, final)
        return final, receipt
    except BaseException:
        shutil.rmtree(stage)
        raise
