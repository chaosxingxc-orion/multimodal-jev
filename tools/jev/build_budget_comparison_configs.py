#!/usr/bin/env python3
"""Materialize registered cap/LR controls; never launch or tune on results."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.budget_control import load_budget_controller
from multimodal_jev.comparison_schedule import budget_comparison_configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--templates', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--export-root', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('retain immutable existing configuration directory')
    terminal = args.profile.parent/'terminal.json'
    if json.loads(terminal.read_text())['exit_code'] != 0:
        raise ValueError('successful processor profile required')
    paths = sorted(args.templates.glob('*.json'))
    configs = budget_comparison_configs(
        {path.name:json.loads(path.read_text()) for path in paths},
        args.profile.resolve(),file_hash(args.profile),args.export_root)
    # Validate each distinct schedule/input mode before materializing any files.
    # LR/reward variants share this identity and are checked by their launch preflight.
    verified = {}
    for seed in (17,29,43):
        for arm in ('sft','pair-grpo-standard'):
            name = f'{seed}-{arm}-lr-1e-5.json';config = configs[name]
            controller = load_budget_controller(config,sft=arm=='sft')
            first = config['batches'][0]
            verified[name] = {'first_batch_admitted': controller.admit(first),
                             'first_batch_reservation': controller.reservation(first)[0]}
    args.output.mkdir(parents=True)
    for name,config in configs.items():
        atomic_json(args.output/name,config)
    receipt = {'configs':len(configs),'output':str(args.output.resolve()),
        'profile_path':str(args.profile.resolve()),'profile_sha256':file_hash(args.profile),
        'template_sha256':{p.name:file_hash(p) for p in paths},
        'config_sha256':{name:file_hash(args.output/name) for name in configs},
        'validated_schedule_modes':verified,
        'scope':'static cap/LR configs only; real execution gates and per-arm preflight required'}
    atomic_json(args.output/'PREPARATION.json',receipt)
    print(json.dumps({'configs':len(configs),'output':str(args.output),'scope':receipt['scope']}))


if __name__=='__main__':
    main()
