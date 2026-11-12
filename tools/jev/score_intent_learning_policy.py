#!/usr/bin/env python3
"""Read a completed pinned D/Q gate and report learning diagnostics only."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_policy_metrics import score


def run(config):
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('scoring evidence changed')
    root=Path(config['gate_root']);read=lambda p:json.loads(p.read_text())
    required=[root/'config.json',root/'run/READY.json',root/'run/terminal.json',root/'run/vllm/result.json',root/'run/vllm/terminal.json']
    if any(config['pins'].get(str(p))!=file_hash(p) for p in required):raise ValueError('pinned completed gate required')
    if read(root/'run/terminal.json')['exit_code']!=0 or read(root/'run/vllm/terminal.json')['exit_code']!=0:raise ValueError('gate failed')
    ready=read(root/'run/READY.json');result=read(root/'run/vllm/result.json')
    if not ready['passed'] or not result['passed'] or ready['result_sha256']!=file_hash(root/'run/vllm/result.json'):raise ValueError('gate not ready')
    if ready['base_id']!=result['base_id'] or ready['adapter_id']!=result['adapter_id']:raise ValueError('policy differs')
    reports={};expected=set()
    for arm in ('D','Q'):
        items,admission=load_pilot(config['pilot'],config['manifest_sha256'],arm)
        wanted={i['id']+'::'+arm for i in items};expected|=wanted
        records=[r for r in result['records'] if r['id'] in wanted]
        reports[arm]=dict(metrics=score(items,records,arm),admission=admission)
    if len(result['records'])!=len(expected) or {r['id'] for r in result['records']}!=expected:raise ValueError('full D/Q population required')
    return dict(base_id=result['base_id'],adapter_id=result['adapter_id'],role='learning',reports=reports,
        scope='89-source reference-reviewed learning diagnostics only. Structured template means Q/P shared inputs, not evidence of P training. No development/test accuracy or RL improvement claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('immutable score already exists')
    atomic_json(args.output,run(json.loads(args.config.read_text())))
