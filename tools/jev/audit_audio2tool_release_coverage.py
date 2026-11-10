#!/usr/bin/env python3
"""Map all non-dialogue release records to pinned audio inventory.

Release annotations are parsed; only IDs, paths, multiplicities and differing
field NAMES are emitted. No label/transcript values or actor inputs exported.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath

from multimodal_jev.audio2tool_inventory import REVISION
from multimodal_jev.baseline import atomic_json, file_hash


def coverage(rows, tier, inventory):
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('nonempty unique release IDs required')
    expected={a['path']:a for r in inventory if r['tier']==tier for a in r['audios']}
    grouped=defaultdict(list);mapping=[]
    for row in rows:
        p=PurePosixPath(row['path'])
        if (p.is_absolute() or '..' in p.parts or len(p.parts)!=3
                or not p.parts[0].startswith(f'tier{tier}_')
                or not p.parts[0].endswith('_audios')):
            raise ValueError('release-relative tier audio path required')
        path='public/'+str(p)
        if path not in expected:raise ValueError('release references unindexed audio')
        grouped[path].append(row)
        mapping.append(dict(tier=tier,release_id=row['id'],source_query_idx=row['query_idx'],
                            audio_path=path,audio_sha256=expected[path]['sha256']))
    if set(grouped)!=set(expected):raise ValueError('inventory audio omitted from release')
    duplicates=[]
    for path,records in grouped.items():
        if len(records)>1:
            keys=set().union(*(r.keys() for r in records))
            differing=sorted(k for k in keys if any((k in r)!=(k in records[0])
                or r.get(k)!=records[0].get(k) for r in records[1:]))
            duplicates.append(dict(path=path,release_ids=[r['id'] for r in records],
                                   multiplicity=len(records),differing_fields=differing))
    return mapping,dict(release_records=len(rows),unique_audio_paths=len(grouped),
                        excess_records=len(rows)-len(grouped),duplicate_paths=duplicates,
                        path_coverage_exact=True)


def run(args):
    if file_hash(args.manifest)!=args.manifest_hash or file_hash(args.inventory)!=args.inventory_hash:
        raise ValueError('registered inventory/manifest changed')
    manifest=json.loads(args.manifest.read_text());inventory=[json.loads(l) for l in args.inventory.open()]
    prefix='datasets/audio2tool/public/'
    entries=[r for r in manifest['files'] if r.get('asset')=='audio2tool'
        and r['path'].startswith(prefix) and '_data/' in r['path'] and r['path'].endswith('.json')]
    if len(entries)!=8:raise ValueError('eight pinned release JSON files required')
    tiers={};pins={};all_mappings=[]
    for entry in sorted(entries,key=lambda r:r['path']):
        relative=entry['path'][len('datasets/audio2tool/'):]
        tier=int(PurePosixPath(relative).parts[1][4])
        if tier==7:continue  # separately audited multi-turn release
        path=args.root/relative;data=path.read_bytes()
        digest=(hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
                if entry['algorithm']=='git-sha1' else hashlib.sha256(data).hexdigest())
        if (entry['algorithm'] not in ('git-sha1','sha256') or digest!=entry['hash']
                or len(data)!=entry['size']
                or entry['url']!=f'https://huggingface.co/datasets/RVtech/Audio2Tool/resolve/{REVISION}/{relative}'):
            raise ValueError('release bytes differ from pinned manifest')
        mapping,report=coverage(json.loads(data),tier,inventory)
        if str(tier) in tiers:raise ValueError('duplicate release tier')
        tiers[str(tier)]=report;pins[relative]=hashlib.sha256(data).hexdigest();all_mappings.extend(mapping)
    if set(tiers)!=set('1234568'):raise ValueError('all seven single-audio tiers required')
    with (args.output/'release-audio-map.jsonl').open('x') as f:
        for row in all_mappings:f.write(json.dumps(row)+'\n')
    atomic_json(args.output/'result.json',dict(status='coverage_complete',tiers=tiers,
        single_audio_release_records=len(all_mappings),single_audio_unique_paths=sum(t['unique_audio_paths'] for t in tiers.values()),
        prior_tier7_dialogues=2146,combined_release_record_count=len(all_mappings)+2146,
        combined_unique_audio_or_dialogue_count=sum(t['unique_audio_paths'] for t in tiers.values())+2146,
        source_release_sha256=pins,manifest_sha256=args.manifest_hash,inventory_sha256=args.inventory_hash,
        mapping_sha256=file_hash(args.output/'release-audio-map.jsonl'),runner_sha256=file_hash(__file__),
        annotations_parsed=True,model_loaded=False,actor_inputs_created=False,
        scope='All seven non-dialogue release JSONs pinned and mapped to full audio inventory; record multiplicity retained. Tier7 counts inherited from separate full audit. No transcript/label values emitted, score or paper denominator certification.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','manifest','inventory','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('manifest','inventory'):p.add_argument('--'+k+'-hash',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))
