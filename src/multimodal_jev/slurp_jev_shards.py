"""Plan bounded full-split execution without expanding or reading gold tasks."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .baseline import file_hash
from .consumption import canonical
from .slurp_jev_full import validate_bundle,expand_bundle


def plan_shards(release,split,recordings_per_shard=100):
    if split not in ('train','valid','test') or type(recordings_per_shard) is not int or recordings_per_shard<1:
        raise ValueError('official split and positive shard size required')
    release=Path(release);manifest=json.loads((release/'MANIFEST.json').read_text())
    bank_path=release/'task-bank.json';bank=json.loads(bank_path.read_text())
    if file_hash(bank_path)!=manifest['identity']['task_bank_sha256']:
        raise ValueError('task bank changed')
    path=release/f'{split}.bundles.jsonl';pins=manifest['splits'][split]
    if file_hash(path)!=pins['bundles_sha256']:raise ValueError('bundle source changed')
    groups=defaultdict(list);seen=set()
    for line in path.open():
        row=json.loads(line);validate_bundle(row,bank)
        if row['split']!=split or row['id'] in seen:raise ValueError('split or bundle identity changed')
        seen.add(row['id']);groups[row['source_group']].append(row)
    if len(seen)!=pins['recordings'] or len(groups)!=pins['source_groups']:
        raise ValueError('full official population required')
    tasks=len(bank['tasks'])
    if len(seen)*tasks!=pins['decisions']:raise ValueError('complete task population required')
    shards=[];pending=[];count=0
    def append():
        bundles=[r for group in pending for r in groups[group]]
        shards.append(dict(id=f'shard-{len(shards):04d}',source_groups=list(pending),
            recordings=len(bundles),decisions=len(bundles)*tasks,
            ordered_bundle_ids=[r['id'] for r in bundles],
            bundles_sha256=hashlib.sha256(b''.join(map(canonical,bundles))).hexdigest()))
    # Sorting source IDs makes execution order independent of target content.
    for group in sorted(groups):
        size=len(groups[group])
        if pending and count+size>recordings_per_shard:
            append();pending=[];count=0
        pending.append(group);count+=size
    if pending:append()
    return dict(schema='slurp-jev-full-shards-v1',release=str(release.resolve()),
        source_manifest_sha256=file_hash(release/'MANIFEST.json'),split=split,
        task_bank_sha256=file_hash(bank_path),bundles_sha256=file_hash(path),
        recordings=len(seen),source_groups=len(groups),decisions=len(seen)*tasks,
        tasks_per_recording=tasks,recordings_per_shard_target=recordings_per_shard,
        indivisible_group_exceeds_target=any(len(v)>recordings_per_shard for v in groups.values()),
        shards=shards,gold_read=False,prompt_selected=False,inference_dispatched=False,
        scope='Complete split execution plan, source groups indivisible; all tasks retained; no gold or model outputs read')


def load_shard(plan,shard_id):
    """Verify and return compact bundles; expansion remains lazy per recording."""
    matches=[s for s in plan['shards'] if s['id']==shard_id]
    if len(matches)!=1:raise ValueError('unique registered shard required')
    shard=matches[0];release=Path(plan['release'])
    if file_hash(release/'MANIFEST.json')!=plan['source_manifest_sha256']:
        raise ValueError('source manifest changed')
    path=release/(plan['split']+'.bundles.jsonl')
    if file_hash(path)!=plan['bundles_sha256']:raise ValueError('bundle source changed')
    bank=json.loads((release/'task-bank.json').read_text())
    if file_hash(release/'task-bank.json')!=plan['task_bank_sha256']:
        raise ValueError('task bank changed')
    wanted=set(shard['source_groups']);groups=defaultdict(list)
    for line in path.open():
        row=json.loads(line)
        if row['source_group'] in wanted:
            validate_bundle(row,bank);groups[row['source_group']].append(row)
    rows=[r for g in shard['source_groups'] for r in groups[g]]
    if (set(groups)!=wanted or len(wanted)!=len(shard['source_groups'])
            or len(rows)!=shard['recordings']
            or [r['id'] for r in rows]!=shard['ordered_bundle_ids']
            or hashlib.sha256(b''.join(map(canonical,rows))).hexdigest()!=shard['bundles_sha256']
            or len(rows)*len(bank['tasks'])!=shard['decisions']):
        raise ValueError('complete registered shard required')
    return rows,bank


def materialize_shard(plan_path,shard_id,output):
    """Expand one registered shard in source order, with no target dependency."""
    plan_path,output=Path(plan_path),Path(output)
    plan=json.loads(plan_path.read_text());bundles,bank=load_shard(plan,shard_id)
    output.mkdir(parents=True,exist_ok=False)
    path=output/(plan['split']+'.requests.jsonl');count=0
    with path.open('xb') as handle:
        for bundle in bundles:
            for request in expand_bundle(bundle,bank):
                handle.write(canonical(request));count+=1
    receipt=dict(source_release=plan['release'],source_manifest_sha256=plan['source_manifest_sha256'],
        split=plan['split'],source_groups=sorted({r['source_group'] for r in bundles}),
        recordings=len(bundles),requests=count,requests_sha256=file_hash(path),
        shard_id=shard_id,shard_plan=str(plan_path.resolve()),shard_plan_sha256=file_hash(plan_path),
        builder_sha256=file_hash(__file__),scope='Complete label-free registered shard; not the full split')
    with (output/'MANIFEST.json').open('xb') as handle:handle.write(canonical(receipt))
    return receipt
