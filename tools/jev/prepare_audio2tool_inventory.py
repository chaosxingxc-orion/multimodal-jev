#!/usr/bin/env python3
"""Pin Audio2Tool registry and index manifest audio without reading TEST answers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from multimodal_jev.audio2tool_inventory import REVISION, registry, audio_inventory
from multimodal_jev.baseline import atomic_json, file_hash


def run(manifest, registry_path, output):
    data=registry_path.read_bytes(); manifest_data=manifest.read_bytes()
    original=json.loads(manifest_data)
    assets=[a for a in original['assets'] if a['asset']=='audio2tool']
    if len(assets)!=1 or assets[0]['revision']!=REVISION or assets[0]['repo']!='RVtech/Audio2Tool':
        raise ValueError('fixed Audio2Tool release required')
    files=[r for r in original['files'] if r.get('asset')=='audio2tool']
    if len(files)!=assets[0]['files']:raise ValueError('manifest count differs')
    entries=[r for r in files if r['path']=='datasets/audio2tool/tools_registry.csv']
    if len(entries)!=1:raise ValueError('one independently pinned registry required')
    entry=entries[0]
    if (entry['algorithm']!='git-sha1' or entry['size']!=len(data)
            or hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=entry['hash']):
        raise ValueError('registry differs from fixed manifest')
    tools=registry(data,hashlib.sha256(data).hexdigest()); inventory=audio_inventory(files)
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'registry.json',tools)
    with (output/'audio-inventory.jsonl').open('x') as stream:
        for row in inventory:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    tiers={}
    for tier in range(1,9):
        rows=[r for r in inventory if r['tier']==tier]
        tiers[str(tier)]=dict(queries=len(rows),audio_files=sum(len(r['audios']) for r in rows),
            audio_per_query_distribution=dict(sorted(Counter(len(r['audios']) for r in rows).items())))
    report=dict(revision=REVISION,source_manifest=str(manifest),source_manifest_sha256=hashlib.sha256(manifest_data).hexdigest(),
        source_manifest_files=len(files),registry_source=str(registry_path),registry_sha256=hashlib.sha256(data).hexdigest(),
        registry_manifest_git_sha1_verified=True,tools=len(tools),domains=dict(Counter(r['domain'] for r in tools)),
        tiers=tiers,inventory_queries=len(inventory),inventory_audio_files=sum(len(r['audios']) for r in inventory),
        artifacts={str(p):file_hash(p) for p in [output/'registry.json',output/'audio-inventory.jsonl']},
        source_files={str(p):file_hash(p) for p in [Path(__file__),Path(__file__).parents[2]/'src/multimodal_jev/audio2tool_inventory.py']},
        inference_dispatched=False,scope='Manifest-only audio index and independently pinned registry; no query metadata, transcript, reference call or waveform read. Audio hashes inherited; no current file completeness or waveform rehash claim.')
    atomic_json(output/'result.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('manifest','registry','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.manifest,args.registry,args.output),ensure_ascii=False,indent=2))
