#!/usr/bin/env python3
"""Prepare immutable offline source-selection review; no model or optimization."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_selection_review import select_review_packet


def prepare(source,output,per_stratum=6):
    manifest=json.loads((source/'MANIFEST.json').read_text())
    if manifest['schema']!='slurp-clean-source-bundles-v1':raise ValueError('clean source release required')
    inputs={}
    for name in ('learning.requests.jsonl','selection.requests.jsonl','selection.targets.jsonl'):
        path=source/name
        if file_hash(path)!=manifest['files'][name]:raise ValueError('source file changed')
        inputs[name]=[json.loads(s) for s in path.read_text().splitlines()]
    metadata=Path(manifest['source_root'])/'metadata/train.jsonl'
    if file_hash(metadata)!=manifest['identity']['input_hashes']['clean_metadata_sha256']:
        raise ValueError('clean TRAIN metadata changed')
    rows,report=select_review_packet(inputs['selection.requests.jsonl'],inputs['selection.targets.jsonl'],
        [json.loads(s) for s in metadata.read_text().splitlines()],inputs['learning.requests.jsonl'],per_stratum)
    for row in rows:
        if file_hash(row['request']['input']['audio'])!=row['request']['audio_sha256']:
            raise ValueError('selected audio bytes changed')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'packet.json',rows)
    report.update(source=str(source),source_manifest_sha256=file_hash(source/'MANIFEST.json'),
        metadata_sha256=file_hash(metadata),packet_sha256=file_hash(output/'packet.json'),
        implementation_sha256=file_hash(Path(__file__).parents[2]/'src/multimodal_jev/span_selection_review.py'))
    atomic_json(output/'receipt.json',report)
    for p in (output,*output.iterdir()):p.chmod(0o777 if p.is_dir() else 0o644)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--per-stratum',type=int,default=6);a=p.parse_args()
    report=prepare(a.source,a.output,a.per_stratum);atomic_json(a.receipt,report);print(json.dumps(report,indent=2))
