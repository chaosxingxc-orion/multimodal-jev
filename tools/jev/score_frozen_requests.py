#!/usr/bin/env python3
"""Join completed inference with separate targets, checking request custody."""
import argparse
import hashlib
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.consumption_score import score

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--six-views',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve scored result')
    identity=json.loads((args.run/'identity.json').read_text())
    result=json.loads((args.run/'result.json').read_text())
    terminal=json.loads((args.run/'terminal.json').read_text())
    if terminal['exit_code']!=0 or result['status']!='passed':raise ValueError('completed inference required')
    manifest=Path(identity['config']['requests'])
    if file_hash(manifest)!=identity['requests_sha256']:raise ValueError('requests changed')
    if file_hash(args.run/'predictions.jsonl')!=result['predictions_sha256']:raise ValueError('predictions changed')
    selected=json.loads((args.run/'selection.json').read_text())['ids']
    all_requests=[json.loads(line) for line in manifest.open()]
    requests=[r for r in all_requests if r['id'] in set(selected)]
    if len(selected)!=len(set(selected)) or len(requests)!=len(selected):raise ValueError('selection changed')
    targets=[json.loads(line) for line in args.targets.open()]
    gold={r['id']:r for r in targets}
    if len(gold)!=len(targets):raise ValueError('duplicate target IDs')
    for r in requests:
        if gold[r['id']]['request_sha256']!=hashlib.sha256(canonical(r)).hexdigest():
            raise ValueError('target/request binding changed')
    predictions=[json.loads(line) for line in (args.run/'predictions.jsonl').open()]
    request_hashes={r['id']:hashlib.sha256(canonical(r)).hexdigest() for r in requests}
    for r in predictions:
        if r['request_sha256']!=request_hashes[r['example_id']]:raise ValueError('prediction/request binding changed')
    if args.six_views:
        from multimodal_jev.six_view_score import score_six_views
        report=score_six_views(requests,targets,predictions)
    else:
        report=score(requests,targets,predictions)
    report.update(split=identity['config']['split'],purpose=identity['config']['purpose'],
        run=str(args.run),targets_sha256=file_hash(args.targets),result_sha256=file_hash(args.run/'result.json'))
    atomic_json(args.output,report)
    print(json.dumps(report,indent=2))
