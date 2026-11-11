#!/usr/bin/env python3
"""Publish an exclusive, provenance-bound learning-only derived Span dataset."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_training import load_reviewed_bundles
from multimodal_jev.span_rule_dataset import build_rule_controls

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    bundles,manifest=load_reviewed_bundles(args.release,args.manifest_sha256,['date','time','timeofday'])
    rows,report=build_rule_controls(bundles)
    args.output.mkdir(parents=True,exist_ok=False)
    dataset=args.output/'learning.jsonl'
    dataset.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    atomic_json(args.output/'audit.json',report)
    source=Path(__file__).resolve().parents[2]/'src/multimodal_jev'
    receipt=dict(schema='jev-span-rule-controls-v1',parent_release=str(args.release),
        parent_manifest_sha256=args.manifest_sha256,parent_review_scope=manifest.get('review_scope'),
        files={p.name:file_hash(p) for p in (dataset,args.output/'audit.json')},
        source_sha256={str(p):file_hash(p) for p in [Path(__file__),source/'span_rule_dataset.py',source/'decision_rules.py',source/'span_training.py']},
        records=len(rows),training_admitted=False,evaluation_admitted=False,
        scope='Learning development fixtures only; rule distributions not balanced for training; no independent evaluation release')
    atomic_json(args.output/'MANIFEST.json',receipt)
    for path in [args.output,*args.output.iterdir()]:path.chmod(0o777 if path.is_dir() else 0o644)
    print(json.dumps(dict(output=str(args.output),records=len(rows),matched_triplets=report['matched_triplets'],excluded=len(report['excluded']),manifest_sha256=file_hash(args.output/'MANIFEST.json')),indent=2))
