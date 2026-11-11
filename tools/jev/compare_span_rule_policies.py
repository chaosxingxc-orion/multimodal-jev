#!/usr/bin/env python3
"""Compare fixed learning inputs only after independently verifying each run."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from verify_span_rule_baseline import verify


def compare(roots):
    reports={name:verify(root) for name,root in roots.items()}
    predictions={name:[json.loads(line) for line in (root/'run/records.jsonl').read_text().splitlines()]
                 for name,root in roots.items()}
    names=list(roots);reference=predictions[names[0]]
    for name in names[1:]:
        if reports[name]['base_id']!=reports[names[0]]['base_id']:raise ValueError('different base model')
        for a,b in zip(reference,predictions[name],strict=True):
            if a['id']!=b['id']:raise ValueError('different selected requests')
            x,y=deepcopy(a['request_identity']),deepcopy(b['request_identity'])
            x.pop('policy_id');y.pop('policy_id')
            if x!=y:raise ValueError('audio, tokens, preprocessing, candidates or questions differ')
    table=[]
    for name,report in reports.items():
        for arm in ('final_only','structured','reordered'):
            rows=[r for r in report['metrics']['per_rule'] if r['arm']==arm]
            total=sum(r['n'] for r in rows)
            table.append(dict(policy=name,adapter_id=report['adapter'],arm=arm,requests=total,
                final_accuracy=sum(r['final_accuracy']*r['n'] for r in rows)/total,
                macro_rule_balanced_accuracy=sum(r['balanced_accuracy'] for r in rows)/len(rows),
                all_span_accuracy=sum(r['all_span_accuracy']*r['n'] for r in rows)/total,
                executor_accuracy=None if arm=='final_only' else sum(r['executor_accuracy']*r['n'] for r in rows)/total,
                rule_consistency=None if arm=='final_only' else sum(r['rule_consistency']*r['n'] for r in rows)/total,
                complement=report['metrics']['complement_response'][arm],
                paired_vs_final_only=None if arm=='final_only' else report['metrics']['paired_vs_final_only'][arm]))
    return dict(passed=True,identical_inputs_across_policies=True,table=table,
        runs={name:dict(root=str(roots[name]),files=report['files']) for name,report in reports.items()},
        scope='Same 216 learning requests across immutable policies; trained policies saw these sources; no equal-budget SFT control, no causal RL benefit or held-out generalization claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True,type=Path);p.add_argument('--sft',required=True,type=Path)
    p.add_argument('--rl2',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();report=compare(dict(base=args.base,sft=args.sft,rl2=args.rl2));atomic_json(args.output,report)
    print(json.dumps(report['table'],indent=2))
