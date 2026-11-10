#!/usr/bin/env python3
"""Full target-roster synthetic scorer controls; never model performance."""
import argparse
import json
from pathlib import Path

from multimodal_jev.audio2tool_score import evaluate_projected,load_official
from multimodal_jev.baseline import atomic_json,file_hash


def run(args):
    if file_hash(args.targets)!=args.targets_hash:raise ValueError('registered targets changed')
    targets=[json.loads(l) for l in args.targets.open()]
    if len(targets)!=30733:raise ValueError('all release records required')
    official=load_official(args.official_metrics)
    # Diagnostic only: synthesized failures and primary-only oracle outputs.
    cases={}
    for mode in ('all_runtime_failures','primary_only_oracle'):
        predictions=[dict(id=t['id'],raw_text='' if mode=='all_runtime_failures' else json.dumps([t['primary']]),
                          error='synthetic_failure' if mode=='all_runtime_failures' else None) for t in targets]
        report,_=evaluate_projected(targets,predictions,official)
        if report['all']['records']!=30733 or report['all']['full_population_strict_calls_exact'] is not None:
            raise ValueError('full denominator/unknown coverage lost')
        subset=report['all']['full_arguments_subset']
        if (subset['eligible_records'],subset['unavailable_records'])!=(25969,4764):
            raise ValueError('registered target coverage changed')
        if mode=='all_runtime_failures':
            if report['all']['runtime_errors']!=30733 or any(report['all']['full_record_metrics'].values()):
                raise ValueError('failed rows gained credit or disappeared')
        elif report['all']['full_record_metrics']['strict_primary_exact']!=1:
            raise ValueError('primary identity mapping failed')
        cases[mode]=report
    return dict(status='synthetic_controls_passed',targets_sha256=args.targets_hash,
        official_metrics_sha256=file_hash(args.official_metrics),cases=cases,
        runner_sha256=file_hash(__file__),model_inference=False,
        scope='Synthetic complete-roster failures and primary-only oracle generated from targets solely to audit scoring. Not measured model performance; no prediction artifacts or actor feedback generated.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('targets','official-metrics','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--targets-hash',required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior result')
    atomic_json(a.output,run(a))
