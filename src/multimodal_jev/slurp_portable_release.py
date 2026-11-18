"""Self-contained SLURP admission evidence, with immutable legacy compatibility.

The original v1 implementation remains byte-identical because existing releases
pin its source. New publications snapshot evidence and use release-relative
paths. Original logical audio paths remain metadata; audio/cache resolution is
an execution configuration concern, independent of admission evidence paths.
"""
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from .baseline import file_hash
from .consumption import canonical
from .slurp_training import content_hash
from . import slurp_official_training as legacy

SCHEMA = 'slurp-official-clean-intent-v2'


def contained(root, name):
    root=Path(root).resolve();relative=Path(name)
    if (not isinstance(name,str) or not name or relative.is_absolute()
            or '..' in relative.parts or str(relative)!=name):
        raise ValueError('canonical release-relative asset path required')
    path=root/relative
    if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in [path,*path.parents] if p!=root.parent):
        raise ValueError('publication path escapes or uses symlink')
    return path


def files(root):
    root=Path(root)
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('publication symlinks forbidden')
    return {str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p!=root/'MANIFEST.json'}


def config(root, expected_hash):
    """Resolve only verified bundle-relative evidence; never open origin paths."""
    root=Path(root);manifest=legacy.read(root/'MANIFEST.json')
    if file_hash(root/'MANIFEST.json')!=expected_hash or manifest['schema']!=SCHEMA:
        raise ValueError('explicit portable publication identity required')
    for name in manifest['files']:contained(root,name)
    if files(root)!=manifest['files']:raise ValueError('complete portable publication asset integrity differs')
    c=legacy.read(root/'publication-config.json')
    if c.get('schema')!='slurp-portable-evidence-v1':raise ValueError('relative evidence configuration required')
    for name,digest in c['pins'].items():
        if manifest['files'].get(name)!=digest or file_hash(contained(root,name))!=digest:
            raise ValueError('portable dependency changed')
    resolved={k:str(contained(root,c[k])) for k in ('source','receipt','tasks')}
    resolved['reviews']={role:[str(contained(root,name)) for name in pair] for role,pair in c['reviews'].items()}
    expected={c['source']+'/MANIFEST.json',c['receipt'],c['tasks']}
    expected.update(n for pair in c['reviews'].values() for n in pair)
    if not expected<=c['pins'].keys():raise ValueError('complete evidence pins required')
    return resolved,c,manifest


def source_population(resolved, evidence):
    """Reconstruct both roles from bundled requests, native targets and reviews."""
    source=Path(resolved['source']);manifest=legacy.read(source/'MANIFEST.json')
    proof=legacy.read(resolved['receipt']);task=legacy.read(resolved['tasks'])['intent']
    if (manifest['schema']!='slurp-clean-source-bundles-v1'
            or proof['release']!=evidence['source_origin']
            or proof['manifest_sha256']!=file_hash(source/'MANIFEST.json')
            or proof['all_native_targets_reconstructed_independently'] is not True
            or proof['cross_role_source_group_or_audio_overlap']!=0):
        raise ValueError('independently verified source provenance required')
    legacy.compile_intent('/input-validation-only.wav',task,'Q')
    pair_to_intent={tuple(v):k for k,v in task['candidate_pairs'].items()}
    result={};group_roles={};hash_roles={};ids=set()
    for role in ('learning','selection'):
        for suffix in ('requests','targets'):
            name=role+'.'+suffix+'.jsonl'
            if file_hash(source/name)!=manifest['files'][name]:raise ValueError('source population changed')
        requests=legacy.rows(source/(role+'.requests.jsonl'));labels=legacy.rows(source/(role+'.targets.jsonl'))
        targets={t['id']:t for t in labels}
        if (len(targets)!=len(labels) or len(requests)!=manifest['counts'][role]
                or len(labels)!=len(requests) or {r['id'] for r in requests}!=targets.keys()):
            raise ValueError('complete unique source population required')
        pending=legacy.pending_groups(*resolved['reviews'][role],role)
        if not pending<={r['source_group'] for r in requests}:raise ValueError('pending outside source role')
        result[role]=[]
        for request in requests:
            if (set(request)!={'id','official_split','role','source_group','audio_sha256','input'}
                    or set(request['input'])!={'audio'} or request['official_split']!='train'
                    or request['role']!=role or request['id'] in ids):
                raise ValueError('unique label-free TRAIN request required')
            ids.add(request['id'])
            for value,index in ((request['source_group'],group_roles),(request['audio_sha256'],hash_roles)):
                if index.setdefault(value,role)!=role:raise ValueError('cross-role source/audio leakage')
            audio=Path(request['input']['audio'])
            if not audio.is_absolute() or request['id']!='slurp:'+audio.name:raise ValueError('audio identity mismatch')
            target=targets[request['id']]
            if target['request_sha256']!=content_hash(request) or target['annotation_status']!='available':
                raise ValueError('native source target binding differs')
            native=target['native_target'];intent=pair_to_intent[(native['scenario'],native['action'])]
            result[role].append(dict(id=request['id'],official_split='train',role=role,
                source_group=request['source_group'],audio=str(audio),audio_sha256=request['audio_sha256'],
                source_request_sha256=content_hash(request),source_target_sha256=content_hash(target),
                intent=intent,scenario=native['scenario'],action=native['action'],
                known_pending_source=request['source_group'] in pending))
    return result,task


def load(root,manifest_sha256,role,*,purpose):
    if (role,purpose) not in {('learning','optimize'),('selection','evaluate')}:
        raise ValueError('selection cannot optimize; explicit role/purpose required')
    root=Path(root);manifest=legacy.read(root/'MANIFEST.json')
    if manifest['schema']==legacy.SCHEMA:
        return legacy.load(root,manifest_sha256,role,purpose=purpose)
    resolved,evidence,manifest=config(root,manifest_sha256)
    if manifest['label_basis']!='official_clean_v7' or manifest['individual_semantic_review'] is not False:
        raise ValueError('official-clean protocol required')
    source,task=source_population(resolved,evidence)
    expected={r['id']:r for r in source[role] if not r['known_pending_source']}
    data=legacy.rows(root/(role+'.jsonl'));assets=legacy.rows(root/(role+'.audio.jsonl'))
    quarantine=legacy.read(root/'quarantine.json')
    if len({r['id'] for r in quarantine})!=len(quarantine):raise ValueError('duplicate quarantine identity')
    if any(r['reason'] not in ('invalid_audio_asset','known_pending_source') for r in quarantine):
        raise ValueError('unregistered quarantine reason')
    pending={r['id'] for r in source[role] if r['known_pending_source']}
    if pending!={r['id'] for r in quarantine if r['role']==role and r['reason']=='known_pending_source'}:
        raise ValueError('pending source mask differs')
    invalid={r['id'] for r in quarantine if r['role']==role and r['reason']=='invalid_audio_asset'}
    if (not invalid<=expected.keys() or len({r['id'] for r in data})!=len(data)
            or {r['id'] for r in data}!=expected.keys()-invalid
            or len(data)!=manifest['counts'][role]['recordings'] or len(assets)!=len(data)):
        raise ValueError('complete admitted population differs')
    for row,asset in zip(data,assets,strict=True):
        if row!=expected[row['id']] or asset['id']!=row['id'] or asset['audio_sha256']!=row['audio_sha256']:
            raise ValueError('native labels or audio admission changed')
    if legacy.read(root/'tasks.json')!=legacy.read(resolved['tasks']):raise ValueError('task definition differs')
    counts=dict(recordings=len(data),source_groups=len({r['source_group'] for r in data}),
        classes=dict(sorted(Counter(r['intent'] for r in data).items())),audio_hours=sum(r['seconds'] for r in assets)/3600)
    reported=manifest['counts'][role]
    if (any(counts[k]!=reported[k] for k in ('recordings','source_groups','classes'))
            or not math.isclose(math.fsum(r['seconds'] for r in assets)/3600,reported['audio_hours'],rel_tol=0,abs_tol=1e-9)):
        raise ValueError('reported counts differ')
    return data,task,manifest


def migrate(source,expected_hash,output):
    """Validate v1, snapshot all evidence, preserve payload bytes, publish v2."""
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError('immutable publication exists')
    original={}
    for role,purpose in (('learning','optimize'),('selection','evaluate')):
        original[role]=legacy.load(source,expected_hash,role,purpose=purpose)
    old=legacy.read(source/'MANIFEST.json');c=legacy.read(source/'publication-config.json')
    output.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.portable-intent-',dir=output.parent))
    try:
        for name in old['files']:
            if Path(name).name!=name:raise ValueError('legacy flat publication required')
            if name!='publication-config.json':shutil.copyfile(source/name,stage/name)
        mapping={};pins={}
        def copy_asset(path,relative):
            path=Path(path);target=contained(stage,relative);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target);digest=file_hash(target)
            mapping[str(path)]=relative;pins[relative]=digest
        source_root=Path(c['source'])
        for path in sorted(source_root.iterdir()):
            if path.is_file():copy_asset(path,'evidence/source/'+path.name)
        for path,digest in c['pins'].items():
            if file_hash(path)!=digest:raise ValueError('publication dependency changed: '+path)
            if path not in mapping:copy_asset(path,'evidence/assets/'+digest+'.asset')
        evidence=dict(schema='slurp-portable-evidence-v1',source='evidence/source',source_origin=c['source'],
            receipt=mapping[c['receipt']],tasks=mapping[c['tasks']],
            reviews={role:[mapping[p] for p in pair] for role,pair in c['reviews'].items()},pins=pins)
        (stage/'publication-config.json').write_bytes(canonical(evidence))
        manifest=dict(old,schema=SCHEMA,source='evidence/source',source_receipt=evidence['receipt'],
            files=files(stage),implementation_sha256=file_hash(__file__),
            migration=dict(source_schema=legacy.SCHEMA,source_manifest_sha256=expected_hash,
                unchanged_payload_sha256={n:h for n,h in old['files'].items() if n!='publication-config.json'}))
        (stage/'MANIFEST.json').write_bytes(canonical(manifest))
        for role,purpose in (('learning','optimize'),('selection','evaluate')):
            actual=load(stage,file_hash(stage/'MANIFEST.json'),role,purpose=purpose)
            if actual[:2]!=original[role][:2]:raise ValueError('migration changed population or task')
        # mkdtemp is owner-only; a published shared dataset must be readable
        # after atomic rename by other training/collection users.
        for path in stage.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        stage.chmod(0o755)
        os.rename(stage,output);return manifest
    except BaseException:
        shutil.rmtree(stage);raise


def publish(config,output,progress=None):
    """Default publisher produces v2; v1 builder remains a frozen internal stage."""
    output=Path(output)
    if output.exists():raise FileExistsError('immutable intent publication exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.intent-admission-',dir=output.parent) as temp:
        admitted=Path(temp)/'v1';legacy.publish(config,admitted,progress)
        return migrate(admitted,file_hash(admitted/'MANIFEST.json'),output)
