#!/usr/bin/env python3
"""Offline diagnostics bound to accepted pair hashes; no new model execution."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.final_probability_diagnostics import diagnose
from multimodal_jev.simple_final_predictions import from_backend
from multimodal_jev.simple_selection_inputs import verify
from multimodal_jev.slurp_official_training import load, example


def run(args):
    receipt = json.loads(args.accepted_pair.read_text())
    if receipt['role'] != 'TRAIN_internal_selection_not_official_TEST' or receipt['recordings'] != 9226:
        raise ValueError('accepted complete selection pair required')
    paths = dict(inputs=args.inputs, base=args.base, candidate=args.candidate)
    for name, path in paths.items():
        if file_hash(path) != receipt[name+'_sha256']:
            raise ValueError('accepted artifact changed: '+name)
    cfg = json.loads(args.experiment.read_text())['task']
    if cfg['release_sha256'] != receipt['release_sha256']:
        raise ValueError('selection release mismatch')
    rows, task, _ = load(cfg['release'], cfg['release_sha256'], 'selection', purpose='evaluate')
    artifact = json.loads(args.inputs.read_text())
    verify(artifact, rows, task, cfg['release_sha256'])
    classes = list(example(rows[0], task, 'direct', 'native')[0].questions[0].values)
    output = {}
    for name in ('base', 'candidate'):
        raw = json.loads(paths[name].read_text())
        policy = receipt['metrics'][name]['policy_id']
        if raw['backend'] != receipt['backend']:
            raise ValueError('accepted backend mismatch')
        records = from_backend(artifact, raw, policy_id=policy)
        result = diagnose(rows, classes, records, policy_id=policy)
        for key in ('accuracy', 'macro_F1_fixed60'):
            if abs(result[key]-receipt['metrics'][name][key]) > 1e-12:
                raise ValueError('original score mismatch')
        output[name] = dict(result, policy_id=policy)
    return dict(schema='final-selection-probability-quality-v1', results=output,
        accepted_pair=str(args.accepted_pair), accepted_pair_sha256=file_hash(args.accepted_pair),
        input_hashes={k:file_hash(v) for k, v in paths.items()},
        source_hashes={str(p):file_hash(p) for p in (Path(__file__),
            Path('src/multimodal_jev/final_probability_diagnostics.py'))},
        scope='Posthoc fixed10-bin categorical probability diagnostic, no selection-driven tuning, new inference or RL efficacy claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('accepted-pair', 'inputs', 'base', 'candidate', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--experiment', type=Path, default=Path('configs/experiments/slurp-simple-final-v1.json'))
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ('per_class','reliability_bins')}
                      for k,v in result['results'].items()}, indent=2))
