"""Native SFC input-only requests with separate, lossless scoring provenance.

This does not decompose function calls into Jev decisions. Supplied assistant
history is part of the official input; only the current target is withheld.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .baseline import file_hash
from .consumption import canonical
from .sfc import parse_calls,source_identity
from .sfc_omni import messages

SPLITS=('train_id','test_id','test_ood')


def validate_request(request):
    if set(request)!={'id','split','source_group','input','audio_sha256'}:
        raise ValueError('only request identity, input and audio hashes allowed')
    if request['split'] not in SPLITS or not request['id'] or not request['source_group']:
        raise ValueError('official request identity required')
    inputs=request['input'];hashes=request['audio_sha256']
    messages(inputs,inputs['audios'])
    if (not isinstance(hashes,list) or len(hashes)!=len(inputs['audios'])
            or any(not isinstance(d,str) or len(d)!=64 or any(c not in '0123456789abcdef' for c in d) for d in hashes)
            or any(not isinstance(p,str) or not Path(p).is_absolute() for p in inputs['audios'])):
        raise ValueError('ordered absolute audio paths and SHA256 hashes required')
    return request


def split_record(record):
    if set(record)!={'example_id','input','supervision','metadata'}:
        raise ValueError('original official record fields required')
    meta=record['metadata'];identity=source_identity(meta['relative_audios'])
    if any(meta[k]!=v for k,v in identity.items()):raise ValueError('source conversation identity changed')
    if record['example_id']!='sfc:'+meta['relative_audios'][-1]:raise ValueError('original example ID required')
    if (len(record['input']['audios'])!=identity['stage']
            or len(meta['relative_audios'])!=identity['stage']
            or any(source_identity([p])['source_group']!=identity['source_group'] for p in meta['relative_audios'])):
        raise ValueError('complete chronological conversation audio required')
    for i,(path,relative) in enumerate(zip(record['input']['audios'],meta['relative_audios'])):
        if (not path.endswith('/'+relative) or source_identity([relative])['stage']!=i+1):
            raise ValueError('ordered original audio paths required')
    supervision=record['supervision']
    if set(supervision)!={'official_solution','calls'} or parse_calls(supervision['official_solution'])!=supervision['calls']:
        raise ValueError('unchanged official target and parsed calls required')
    request=validate_request(dict(id=record['example_id'],split=meta['official_split'],
        source_group=meta['source_group'],input=record['input'],audio_sha256=meta['audio_sha256']))
    target=dict(id=request['id'],supervision=supervision,metadata=meta,
        request_sha256=hashlib.sha256(canonical(request)).hexdigest())
    return request,target


def restore_record(request,target):
    validate_request(request)
    if (set(target)!={'id','supervision','metadata','request_sha256'} or target['id']!=request['id']
            or target['request_sha256']!=hashlib.sha256(canonical(request)).hexdigest()):
        raise ValueError('paired target request binding required')
    record=dict(example_id=request['id'],input=request['input'],supervision=target['supervision'],metadata=target['metadata'])
    if split_record(record)!=(request,target):raise ValueError('lossless official record roundtrip required')
    return record


def export(source,destination):
    manifest=json.loads((source/'MANIFEST.json').read_text())
    if manifest['protocol']!='sfc-official-v1' or set(manifest['splits'])!=set(SPLITS):
        raise ValueError('complete original SFC release required')
    identity=dict(dataset='sfc-native',source_release=str(source.resolve()),source_manifest_sha256=file_hash(source/'MANIFEST.json'),
        source_files={s:file_hash(source/(s+'.jsonl')) for s in SPLITS},
        code_files={n:file_hash(Path(__file__).with_name(n)) for n in ('sfc_consumption.py','sfc.py','sfc_omni.py','sfc_audio.py','consumption.py')})
    if any(identity['source_files'][s]!=manifest['splits'][s]['sha256'] for s in SPLITS):
        raise ValueError('original SFC release changed')
    release=hashlib.sha256(canonical(identity)).hexdigest();final=destination/release
    if final.exists():raise FileExistsError('immutable native release exists')
    destination.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.sfc-native-',dir=destination))
    audio_splits={};reports={}
    try:
        for split in SPLITS:
            ids=set();groups={};count=0;two_audio=0
            with (source/(split+'.jsonl')).open() as handle,(stage/(split+'.requests.jsonl')).open('wb') as requests,(stage/(split+'.targets.jsonl')).open('wb') as targets:
                for line in handle:
                    record=json.loads(line);request,target=split_record(record)
                    if request['split']!=split or request['id'] in ids:raise ValueError('duplicate or wrong-split request')
                    if restore_record(request,target)!=record:raise ValueError('record not preserved')
                    ids.add(request['id']);meta=record['metadata'];group=request['source_group']
                    stages=groups.setdefault(group,{})
                    if meta['stage'] in stages:raise ValueError('duplicate conversation stage')
                    stages[meta['stage']]=request
                    for digest in request['audio_sha256']:
                        if digest in audio_splits and audio_splits[digest]!=split:raise ValueError('audio crosses official splits')
                        audio_splits[digest]=split
                    requests.write(canonical(request));targets.write(canonical(target));count+=1
                    two_audio+=len(request['input']['audios'])==2
            first_text_differences=0
            for group,stages in groups.items():
                expected={1,2} if group.startswith('level_3') else {1}
                if set(stages)!=expected:raise ValueError('missing complete conversation stage')
                if 2 in stages:
                    a,b=stages[1],stages[2]
                    if (a['input']['audios']!=b['input']['audios'][:1]
                            or a['audio_sha256']!=b['audio_sha256'][:1]):
                        raise ValueError('two-turn audio prefix differs')
                    # Published stage-specific function ordering can differ.
                    # Preserve each input verbatim; never normalize it to stage 1.
                    first_text_differences+=a['input']['messages']!=b['input']['messages'][:1]
            pin=manifest['splits'][split]
            if count!=pin['records'] or len(groups)!=pin['source_groups']:raise ValueError('whole official split required')
            reports[split]=dict(requests=count,source_groups=len(groups),two_audio_requests=two_audio,
                stage_specific_first_message_differences=first_text_differences,
                requests_sha256=file_hash(stage/(split+'.requests.jsonl')),targets_sha256=file_hash(stage/(split+'.targets.jsonl')))
        report=dict(release_id=release,identity=identity,splits=reports,
            audio_hashes='Inherited from verified source; cross-split isolation checked, waveform files not rehashed in this export.',
            scope='Lossless native SFC supplied-history inputs; current targets/provenance in separate files. No typed Jev decomposition, inference or interactive execution.')
        (stage/'MANIFEST.json').write_bytes(canonical(report));os.rename(stage,final)
        return final,report
    except BaseException:
        shutil.rmtree(stage);raise
