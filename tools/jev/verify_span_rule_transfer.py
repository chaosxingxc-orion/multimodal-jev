#!/usr/bin/env python3
"""Reconstruct the prospective transfer protocol from pinned source assets."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from prepare_span_rule_transfer import prepare


def verify(path):
    saved=json.loads(path.read_text())
    current=prepare(Path(saved['learning_release']),Path(saved['selection_release']),Path(saved['initializer']['root']))
    if saved!=current:raise ValueError('transfer protocol differs from deterministic reconstruction')
    learning_groups=set(saved['training_semantic_groups'])
    if saved['heldout_semantic_group'] in learning_groups:raise ValueError('task group leakage')
    return dict(passed=True,protocol_sha256=file_hash(path),initializer_id=saved['initializer']['adapter_id'],
        seeds=saved['seeds'],updates_per_branch=saved['updates_per_branch'],
        cells={k:{key:value for key,value in v.items() if key!='request_ids'} for k,v in saved['evaluation_cells'].items()},
        training_admitted=False,scope='Source assets, semantic truth tables, balanced schedules and fact-only initializer audited; no training run or generalization result')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.protocol);atomic_json(a.output,r);print(json.dumps(r,indent=2))
