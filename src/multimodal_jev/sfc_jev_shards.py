"""Complete SFC-Jev execution shards with indivisible supplied-history groups."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .baseline import file_hash
from .consumption import canonical
from .sfc import source_identity
from .sfc_jev_runtime import validate_bundle
from .sfc_teacher_feedback import validate_partition


def digest(rows):
    return hashlib.sha256(b''.join(map(canonical, rows))).hexdigest()


def population(release, split):
    release=Path(release)
    if split not in ('train_id','test_id','test_ood'):raise ValueError('official SFC split required')
    manifest=json.loads((release/'MANIFEST.json').read_text());path=release/f'{split}.requests.jsonl'
    if (manifest['identity']['protocol']!='sfc-jev-bundle-v1'
            or set(manifest['splits'])!={'train_id','test_id','test_ood'}
            or file_hash(path)!=manifest['splits'][split]['requests_sha256']):
        raise ValueError('unchanged complete SFC-Jev release required')
    rows=[];groups=defaultdict(dict);ids=set();count=0
    with path.open() as handle:
        for line in handle:
            row=json.loads(line);validate_bundle(row)
            if row['id'] in ids or row['split']!=split or not row['id'].startswith('sfc:'):
                raise ValueError('unique original split identity required')
            paths=row['input']['context']['audios'];identity=source_identity([row['id'][4:]])
            stage=identity['stage'];group=row['source_group']
            if (identity['source_group']!=group or identity['official_split']!=split
                    or stage!=len(paths) or not paths[-1].endswith('/'+row['id'][4:])
                    or stage in groups[group]):
                raise ValueError('original unique conversation stages required')
            ids.add(row['id']);groups[group][stage]=row;rows.append(row)
            count+=len(row['input']['decisions'])
    pin=manifest['splits'][split]
    if (len(rows),len(groups),count)!=(pin['bundles'],pin['source_groups'],pin['decisions']):
        raise ValueError('full released population required')
    for group,stages in groups.items():
        if set(stages)!=({1,2} if group.startswith('level_3') else {1}):
            raise ValueError('complete conversation required')
        if 2 in stages and (stages[1]['audio_sha256']!=stages[2]['audio_sha256'][:1]
                or stages[1]['input']['context']['audios']!=stages[2]['input']['context']['audios'][:1]):
            raise ValueError('conversation audio prefix differs')
    return rows


def make_plan(release,split,decisions_per_shard=1024,*,partition_path=None,partition_hash=None,part=None):
    if type(decisions_per_shard) is not int or decisions_per_shard<1:
        raise ValueError('positive integer shard decision target required')
    release=Path(release).resolve();all_rows=population(release,split)
    partition_identity=None;rows=all_rows
    if split=='train_id':
        if part not in ('learning','selection') or partition_path is None or partition_hash is None:
            raise ValueError('TRAIN requires a pinned learning or selection partition')
        path=Path(partition_path).resolve()
        if file_hash(path)!=partition_hash:raise ValueError('registered TRAIN partition changed')
        registered=json.loads(path.read_text());validate_partition(all_rows,registered)
        if (registered['release']!=str(release)
                or registered['manifest_sha256']!=file_hash(release/'MANIFEST.json')
                or registered['train_requests_sha256']!=file_hash(release/'train_id.requests.jsonl')):
            raise ValueError('partition release binding changed')
        wanted=set(registered[part]);rows=[r for r in all_rows if r['source_group'] in wanted]
        partition_identity=dict(path=str(path),sha256=partition_hash,part=part)
    elif any(v is not None for v in (partition_path,partition_hash,part)):
        raise ValueError('official TEST cannot be filtered by TRAIN partition')
    groups=defaultdict(list)
    for row in rows:groups[row['source_group']].append(row)
    shards=[];pending=[];count=0
    def append():
        bundles=[r for g in pending for r in sorted(groups[g],key=lambda r:len(r['audio_sha256']))]
        shards.append(dict(id=f'shard-{len(shards):04d}',source_groups=list(pending),bundles=len(bundles),
            decisions=sum(len(r['input']['decisions']) for r in bundles),ordered_bundle_ids=[r['id'] for r in bundles],
            requests_sha256=digest(bundles),decision_keys_sha256=digest(
                [[r['id'],d['id']] for r in bundles for d in r['input']['decisions']])))
    for group in sorted(groups):
        size=sum(len(r['input']['decisions']) for r in groups[group])
        if pending and count+size>decisions_per_shard:
            append();pending=[];count=0
        pending.append(group);count+=size
    if pending:append()
    return dict(schema='sfc-jev-complete-shards-v1',release=str(release),split=split,
        manifest_sha256=file_hash(release/'MANIFEST.json'),requests_sha256=file_hash(release/f'{split}.requests.jsonl'),
        partition=partition_identity,bundles=len(rows),source_groups=len(groups),
        decisions=sum(len(r['input']['decisions']) for r in rows),decisions_per_shard_target=decisions_per_shard,
        indivisible_group_exceeds_target=any(sum(len(r['input']['decisions']) for r in rs)>decisions_per_shard for rs in groups.values()),
        shards=shards,targets_read=False,inference_dispatched=False,
        scope='Complete selected population; source conversations indivisible, chronological turns, every decision retained. '
              'TRAIN learning/selection boundaries pinned; TEST always whole official split. No target, prediction or task-quality read.')


def verify_plan(plan):
    partition=plan['partition']
    kwargs={} if partition is None else dict(partition_path=partition['path'],partition_hash=partition['sha256'],part=partition['part'])
    expected=make_plan(plan['release'],plan['split'],plan['decisions_per_shard_target'],**kwargs)
    if expected!=plan:raise ValueError('complete deterministic shard plan does not reproduce')
    return expected


def load_shard(plan,shard_id):
    verify_plan(plan)
    matches=[s for s in plan['shards'] if s['id']==shard_id]
    if len(matches)!=1:raise ValueError('one registered shard required')
    selected=matches[0];rows=population(plan['release'],plan['split']);by_id={r['id']:r for r in rows}
    rows=[by_id[uid] for uid in selected['ordered_bundle_ids']]
    if digest(rows)!=selected['requests_sha256']:raise ValueError('shard request binding differs')
    return rows


def materialize(plan_path,plan_hash,shard_id,output):
    plan_path=Path(plan_path);output=Path(output)
    if file_hash(plan_path)!=plan_hash:raise ValueError('registered shard plan changed')
    plan=json.loads(plan_path.read_text());rows=load_shard(plan,shard_id)
    output.mkdir(parents=True,exist_ok=False)
    path=output/f"{plan['split']}.requests.jsonl"
    with path.open('xb') as handle:
        for row in rows:handle.write(canonical(row))
    receipt=dict(schema='sfc-jev-shard-input-v1',plan=str(plan_path.resolve()),plan_sha256=plan_hash,
        shard_id=shard_id,release=plan['release'],manifest_sha256=plan['manifest_sha256'],split=plan['split'],
        partition=plan['partition'],bundles=len(rows),source_groups=len({r['source_group'] for r in rows}),
        decisions=sum(len(r['input']['decisions']) for r in rows),requests_sha256=file_hash(path),
        targets_read=False,scope='One complete registered shard, not a new dataset split or full benchmark result.')
    with (output/'MANIFEST.json').open('xb') as handle:handle.write(canonical(receipt))
    return receipt
