#!/usr/bin/env python3
"""Resolve a complete split or fixed source groups into standard audio requests."""
import argparse
import hashlib
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_jev_full import expand_bundle


def materialize(release, split, output, groups_path=None):
    manifest=json.loads((release/'MANIFEST.json').read_text())
    bank=json.loads((release/'task-bank.json').read_text())
    if file_hash(release/'task-bank.json')!=manifest['identity']['task_bank_sha256']:
        raise ValueError('task bank changed')
    bundles=release/f'{split}.bundles.jsonl'
    if file_hash(bundles)!=manifest['splits'][split]['bundles_sha256']:
        raise ValueError('bundle source changed')
    groups=None
    if groups_path is not None:
        raw=json.loads(groups_path.read_text())
        if type(raw) is not list or len(set(raw))!=len(raw) or not raw:
            raise ValueError('nonempty unique source group list required')
        groups=set(raw)
    output.mkdir(parents=True,exist_ok=False)
    count=recordings=0;found=set();seen=set()
    with (output/f'{split}.requests.jsonl').open('wb') as handle:
        for line in bundles.open():
            bundle=json.loads(line)
            if groups is not None and bundle['source_group'] not in groups:continue
            if bundle['id'] in seen or bundle['split']!=split:raise ValueError('bundle identity changed')
            seen.add(bundle['id']);found.add(bundle['source_group']);recordings+=1
            for request in expand_bundle(bundle,bank):handle.write(canonical(request));count+=1
    if not count or (groups is not None and found!=groups):raise ValueError('requested groups missing')
    receipt={'source_release':str(release),'source_manifest_sha256':file_hash(release/'MANIFEST.json'),
        'split':split,'source_groups':sorted(found),'recordings':recordings,'requests':count,
        'requests_sha256':file_hash(output/f'{split}.requests.jsonl'),
        'groups_sha256':file_hash(groups_path) if groups_path else None,
        'builder_sha256':file_hash(__file__),'scope':'label-free runtime requests; no gold file read'}
    atomic_json(output/'MANIFEST.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--split',choices=['train','valid','test'],required=True)
    p.add_argument('--groups',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(materialize(a.release,a.split,a.output,a.groups),indent=2))
