"""Whole-conversation, audio-connected TRAIN partition for prompt optimization."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .baseline import file_hash
from .consumption import canonical
from .sfc import source_identity
from .sfc_jev_runtime import validate_bundle


def partition(rows, exposed_ids, selection_fraction=0.2, seed=17):
    if not 0 < selection_fraction < 1 or type(seed) is not int:
        raise ValueError('valid partition fraction and integer seed required')
    groups={};ids=set();parent={};audio_owner={}
    def find(g):
        while parent[g]!=g:
            parent[g]=parent[parent[g]];g=parent[g]
        return g
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for row in rows:
        validate_bundle(row)
        if row['split']!='train_id' or row['id'] in ids:
            raise ValueError('unique official TRAIN inputs required')
        ids.add(row['id']);g=row['source_group'];parent.setdefault(g,g)
        groups.setdefault(g,[]).append(row)
        paths=row['input']['context']['audios']
        rel=[p[p.index('audio/final_data_audio/'):] for p in paths]
        identity=source_identity(rel)
        if identity['source_group']!=g or identity['stage']!=len(paths):
            raise ValueError('source group/stage differs from audio identity')
        for h in row['audio_sha256']:
            if h in audio_owner:union(g,audio_owner[h])
            audio_owner[h]=g
    if not ids or not set(exposed_ids)<=ids:
        raise ValueError('all exposed IDs must belong to TRAIN')
    for g,items in groups.items():
        sizes=sorted(len(r['audio_sha256']) for r in items)
        if sizes!=([1,2] if g.startswith('level_3') else [1]):
            raise ValueError('complete original conversation required')
        if len(items)==2:
            a,b=sorted(items,key=lambda r:len(r['audio_sha256']))
            if (a['audio_sha256']!=b['audio_sha256'][:1]
                    or a['input']['context']['audios']!=b['input']['context']['audios'][:1]):
                raise ValueError('conversation audio prefix differs')
    components={}
    for g in sorted(groups):components.setdefault(find(g),[]).append(g)
    exposed_groups={g for g,items in groups.items() if any(r['id'] in exposed_ids for r in items)}
    forced={key for key,members in components.items() if set(members)&exposed_groups}
    totals=Counter(g.split('/')[0] for g in groups)
    targets={level:int(count*selection_fraction+0.5) for level,count in totals.items()}
    selected=set();counts=Counter()
    def rank(key):
        return hashlib.sha256(canonical(dict(seed=seed,groups=components[key]))).hexdigest()
    for key in sorted(components,key=lambda k:(rank(k),k)):
        if key in forced:continue
        contribution=Counter(g.split('/')[0] for g in components[key])
        # Never split an audio-connected component to hit a desired count.
        before=sum(abs(counts[level]-targets[level]) for level in totals)
        after=sum(abs(counts[level]+contribution[level]-targets[level]) for level in totals)
        if after<before:
            selected.add(key);counts.update(contribution)
    selection=sorted(g for k in selected for g in components[k])
    learning=sorted(set(groups)-set(selection))
    if not learning or not selection:raise ValueError('nonempty learning and selection partitions required')
    def summary(members):
        items=[r for g in members for r in groups[g]]
        return dict(source_groups=len(members),bundles=len(items),decisions=sum(len(r['input']['decisions']) for r in items),
                    by_level=dict(sorted(Counter(g.split('/')[0] for g in members).items())),
                    audio_hashes=sorted({h for r in items for h in r['audio_sha256']}))
    summaries={k:summary(v) for k,v in (('learning',learning),('selection',selection))}
    if set(summaries['learning']['audio_hashes'])&set(summaries['selection']['audio_hashes']):
        raise ValueError('audio crosses prompt partitions')
    for s in summaries.values():
        s['unique_audio_hashes']=len(s.pop('audio_hashes'))
    return dict(schema='sfc-jev-prompt-partition-v1',seed=seed,selection_fraction=selection_fraction,
        learning=learning,selection=selection,exposed_ids=sorted(exposed_ids),
        forced_learning_groups=sorted(g for k in forced for g in components[k]),
        connected_components=len(components),multi_group_components=[components[k] for k in sorted(components) if len(components[k])>1],
        selection_target_groups_by_level=dict(sorted(targets.items())),summary=summaries,
        method='Seeded SHA256 component ordering; greedily reduce L1 distance to per-level rounded selection-fraction group targets. '
               'Already exposed groups and audio-connected peers forced into learning. All TRAIN retained.',
        scope='Input-only data boundary. No teacher calls, candidate selection or model training. Official ID/OOD splits untouched.')


def build(release, preflight, exposure_reports=()):
    manifest=json.loads((release/'MANIFEST.json').read_text())
    identity=json.loads((preflight/'run/identity.json').read_text())
    result=json.loads((preflight/'run/result.json').read_text())
    if (file_hash(preflight/'run/identity.json')!=result['identity_sha256']
            or result['status']!='passed' or json.loads((preflight/'run/terminal.json').read_text())['exit_code']!=0
            or identity['manifest_sha256']!=file_hash(release/'MANIFEST.json')):
        raise ValueError('successful matching processor preflight required')
    path=release/'train_id.requests.jsonl'
    if file_hash(path)!=manifest['splits']['train_id']['requests_sha256']:
        raise ValueError('TRAIN release changed')
    rows=[json.loads(line) for line in path.open()]
    if len(rows)!=manifest['splits']['train_id']['bundles']:
        raise ValueError('complete TRAIN input population required')
    exposed=set(identity['selected_ids']);evidence={}
    for path in exposure_reports:
        value=json.loads(path.read_text())
        uid=value.get('example_id')
        if not isinstance(uid,str):raise ValueError('exposure report must name its example_id')
        exposed.add(uid);evidence[str(path.resolve())]=file_hash(path)
    report=partition(rows,exposed)
    if sum(v['source_groups'] for v in report['summary'].values())!=manifest['splits']['train_id']['source_groups']:
        raise ValueError('complete source-group coverage required')
    return dict(**report,release=str(release.resolve()),manifest_sha256=file_hash(release/'MANIFEST.json'),
        train_requests_sha256=file_hash(path),preflight_result_sha256=file_hash(preflight/'run/result.json'),
        builder_sha256=file_hash(Path(__file__)),additional_exposure_reports=evidence)
