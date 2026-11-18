#!/usr/bin/env python3
"""Full-selection HF/vLLM metrics only after independent terminal policy parity.

The inference root stays read-only. This separate audit writes a new result
directory and never promotes a partial run or a different training endpoint.
"""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from score_slurp_simple_final import run as score_run
from verify_slurp_simple_parity import verify


def run(root, output):
    root, output = Path(root), Path(output)
    parity = verify(root)
    if parity.get('population_mode') != 'full_selection' or parity['requests'] != 9226 or parity['updates'] != 634:
        raise ValueError('complete terminal634 full selection parity required')
    inference = json.loads((root/'config.json').read_text())
    training = json.loads((Path(parity['training_root'])/'config.json').read_text())
    metrics = {}
    for backend in ('hf', 'vllm'):
        config = dict(release=training['release'], release_sha256=training['release_sha256'],
            policy_id=inference['base_id']+':'+inference['adapter_id'], inputs=inference['inputs'],
            backend_results=str(root/'run'/backend/'result.json'),
            terminal=str(root/'run'/backend/'terminal.json'), inference_config=str(root/'config.json'),
            launch=str(root/'launch.json'), observed=str(root/'observed-final.json'))
        config['pins'] = {config[k]:file_hash(config[k]) for k in
                         ('inputs','backend_results','terminal','inference_config','launch','observed')}
        manifest = str(Path(training['release'])/'MANIFEST.json')
        config['pins'][manifest] = training['release_sha256']
        metrics[backend] = score_run(config)
        atomic_json(output/(backend+'-scoring-config.json'), config)
        atomic_json(output/(backend+'-metrics.json'), metrics[backend])
    # Small numerical differences can cross ties, even when probability parity
    # passes. Report actual argmax changes and score deltas; never force equality.
    names = ('accuracy','macro_F1_fixed60','macro_recall_supported','invalid_rate')
    result = dict(passed=True, population='TRAIN_internal_selection_not_official_TEST',
        recordings=9226, updates=634, policy_id=metrics['hf']['policy_id'], parity=parity,
        metrics=metrics, vllm_minus_hf={k:metrics['vllm'][k]-metrics['hf'][k] for k in names},
        scope='Authenticated full selection scores and independently recomputed HF/vLLM parity; no RL efficacy or official TEST claim.')
    atomic_json(output/'result.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(a.root,a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
