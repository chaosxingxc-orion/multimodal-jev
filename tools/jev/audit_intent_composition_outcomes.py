#!/usr/bin/env python3
"""Compare fixed initial/D12/Q6 ontology slices without new model execution."""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_composition_metrics import compare
from queue_span_native_actor_npu import read,completed
from queue_intent_finite_npu import verify_gate
from compare_intent_learning_update import run as compare_update
from score_intent_learning_policy import run as score


def run(base):
    base=Path(base);initial=base/'slurp-intent-dqp-policy-v2'
    # D12 roots are supplied by the completed original D controller, not guessed.
    controller=read(base/'intent-finite-d12-pilot-controller-v3/config.json')
    roots=[('D12',Path(controller['steps'][-1]['training_root']),Path(controller['steps'][-1]['suite_root'])),
           ('Q6',base/'intent-q12-update06-v1',base/'intent-q12-suite06-v1')]
    cfg=read(roots[1][1]/'config.json');pilot=Path(cfg['pilot']);manifest=cfg['manifest_sha256']
    launch=read(initial/'launch.json')
    if not completed(launch['container_id'],launch['image']):raise ValueError('initial gate not completed')
    verify_gate(initial,pilot,manifest)
    pins={str(initial/n):file_hash(initial/n) for n in ('config.json','run/READY.json','run/terminal.json','run/vllm/result.json','run/vllm/terminal.json')}
    start=score(dict(gate_root=str(initial),pilot=str(pilot),manifest_sha256=manifest,pins=pins))
    task=Path(read(pilot/'MANIFEST.json')['release'])/'tasks.json';pairs=read(task)['intent']['candidate_pairs']
    output={}
    for key,training,suite in roots:
        result=compare_update(training,suite);pins.update(result['pins']);after=result['full_after']
        if after['base_id']!=start['base_id']:raise ValueError('base differs')
        output[key]={arm:compare(start['reports'][arm]['metrics']['records'],after['reports'][arm]['metrics']['records'],pairs) for arm in ('D','Q')}
    for p in (task,pilot/'MANIFEST.json',Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_composition_metrics.py'):
        pins[str(p)]=file_hash(p)
    return dict(results=output,files=pins,role='learning',scope='Exploratory post-observation ontology slices; no held-out, process-reward or multi-hop efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable analysis exists')
    atomic_json(a.output,run(a.base))
