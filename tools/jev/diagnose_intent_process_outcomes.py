#!/usr/bin/env python3
"""Authenticate full gates before describing structured decision outcomes."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_process_outcomes import summarize,transitions
from queue_span_native_actor_npu import read,completed
from queue_span_dp4_resume import check_pins
from queue_intent_finite_npu import verify_gate
from score_intent_learning_policy import run as score


def run(config):
    check_pins(config['pins']);reports=[];evidence=dict(config['pins']);initial=None
    for entry in config['gates']:
        root=Path(entry['root']);launch=read(root/'launch.json')
        if launch['image']!=config['image'] or not completed(launch['container_id'],config['image']):
            raise ValueError('actual complete gate required')
        audit=verify_gate(root,config['pilot'],config['manifest_sha256'])
        pins={str(root/p):file_hash(root/p) for p in ('config.json','launch.json','run/READY.json','run/terminal.json','run/vllm/result.json','run/vllm/terminal.json')}
        scored=score(dict(gate_root=str(root),pilot=config['pilot'],manifest_sha256=config['manifest_sha256'],pins=pins))
        rows=scored['reports']['Q']['metrics']['records']
        if initial is None:initial=rows
        reports.append(dict(update=entry['update'],adapter_id=scored['adapter_id'],gate=audit,
                            outcomes=summarize(rows),from_first=transitions(initial,rows)))
        evidence.update(pins)
    for p in (Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_process_outcomes.py'):
        evidence[str(p)]=file_hash(p)
    return dict(reports=reports,files=evidence,scope='Structured-template learning diagnostics. Gate identity is verified; supplied update labels require external lineage evidence. Not evidence of Q/P training, held-out gain or causal reasoning.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('immutable outcome report exists')
    atomic_json(args.output,run(read(args.config)))
