#!/usr/bin/env python3
"""Recompute complete selection metrics before summarizing an RL curve.

The input manifest binds prediction bytes to a single task run and each joint
policy. Runtime/weight provenance still requires the independent run collector.
"""
import argparse
import hashlib
import json
from pathlib import Path

from context_selection_curve import summarize
from evaluate_context_event_records import evaluate


def run(manifest):
    source=Path(manifest);data=json.loads(source.read_text())
    config=Path(data['experiment'])
    if hashlib.sha256(config.read_bytes()).hexdigest()!=data['experiment_sha256']:
        raise ValueError('experiment changed')
    experiment=json.loads(config.read_text())
    if data['stage']!='task' or data['algorithm'] not in ('grpo_v1','rloo_v1') or not data['run_id']:
        raise ValueError('explicit task run and algorithm required')
    points=[]
    for item in data['checkpoints']:
        result=evaluate(experiment['release'],experiment['release_sha256'],
            item['predictions'],item['predictions_sha256'],item['policy_id'],'selection')
        points.append(dict(step=item['step'],result=result))
    report=summarize(points,experiment,initial_policy_id=data['initial_policy_id'])
    report.update(run_id=data['run_id'],algorithm=data['algorithm'],
        experiment_sha256=data['experiment_sha256'],
        manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        metrics_recomputed=True,runtime_provenance_verified=False)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=run(args.manifest)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
