#!/usr/bin/env python3
"""Publish and audit paired-rule batches without modifying source controls."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_rule_balance import add_complementary_rules,balanced_rule_schedule
from verify_span_rule_controls import verify


def audit(rows,schedule):
    by_id={r['id']:r for r in rows};seen={a:set() for a in ('final_only','structured','reordered')}
    counts=[];recordings=set()
    for batch in schedule['batches']:
        if len(set(batch['source_groups']))!=4:raise ValueError('four distinct source groups required')
        orders=[]
        for arm,ids in batch['arms'].items():
            if len(ids)!=8 or len(set(ids))!=8 or seen[arm].intersection(ids):raise ValueError('repeated request')
            seen[arm].update(ids);items=[by_id[i] for i in ids]
            if any(r['arm']!=arm for r in items):raise ValueError('mixed arm batch')
            width=1 if arm=='final_only' else 3
            if any(len(r['targets'])!=width or any(type(v) is not bool for v in r['targets']) for r in items):
                raise ValueError('Boolean target width differs')
            if any(sum(r['targets'][i] for r in items)!=4 for i in range(width)):raise ValueError('per-span imbalance')
            for polarity in ('positive','negative'):
                selected=[r for r in items if r['provenance']['rule_polarity']==polarity]
                if {tuple(r['provenance']['dependency_targets']) for r in selected}!=set(itertools.product((False,True),repeat=2)):
                    raise ValueError('truth table incomplete')
            order=[(r['provenance']['recording_id'],r['provenance']['rule_polarity']) for r in items]
            orders.append(order);recordings.update(r for r,p in order)
        if orders[0]!=orders[1] or orders[0]!=orders[2]:raise ValueError('unmatched comparison arms')
        counts.append(dict(rule_pair_id=batch['rule_pair_id'],requests_per_arm=8,per_span_false=4,per_span_true=4))
    return dict(passed=True,batches=len(counts),requests_per_arm=sum(len(s) for s in seen.values())//3,
        unique_recordings=len(recordings),per_batch=counts,
        training_admitted=False,evaluation_admitted=False,
        scope='Learning-only sampling audit; individual rules not balanced, no runtime admission or held-out evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--receipt',required=True,type=Path)
    args=p.parse_args();parent=verify(args.source)
    rows=[json.loads(line) for line in (args.source/'learning.jsonl').read_text().splitlines()]
    rows=add_complementary_rules(rows);schedule=balanced_rule_schedule(rows,args.seed)
    result=audit(rows,schedule)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'learning.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    atomic_json(args.output/'schedule.json',schedule)
    source=Path(__file__).resolve().parents[2]/'src/multimodal_jev/span_rule_balance.py'
    manifest=dict(schema='jev-span-balanced-rule-controls-v1',parent_manifest_sha256=parent['manifest_sha256'],
        parent=str(args.source),seed=args.seed,rows=len(rows),training_admitted=False,evaluation_admitted=False,
        source_sha256={str(p):file_hash(p) for p in (Path(__file__),source)},
        files={p.name:file_hash(p) for p in args.output.iterdir() if p.is_file()})
    atomic_json(args.output/'MANIFEST.json',manifest)
    for path in [args.output,*args.output.iterdir()]:path.chmod(0o777 if path.is_dir() else 0o644)
    result.update(root=str(args.output),manifest_sha256=file_hash(args.output/'MANIFEST.json'),pool_rows=len(rows),
                  excluded_recording_rule_instances=len(schedule['excluded']))
    atomic_json(args.receipt,result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_batch'},indent=2))
