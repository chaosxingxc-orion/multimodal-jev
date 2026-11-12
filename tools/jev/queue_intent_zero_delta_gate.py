#!/usr/bin/env python3
"""Evaluate the frozen zero-delta control only after existing residency replay."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_native_actor_npu import read,completed,locks,launch,wait_exact
from queue_span_dp4_resume import check_pins
from queue_intent_prepared_acceptance import wait_resources
from queue_intent_residency_replay import accepted
from queue_intent_finite_npu import verify_gate
from score_intent_learning_policy import run as score


def await_predecessor(config,output):
    root=Path(config['predecessor_controller'])
    for name in ('config.json','launch.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('pinned replay predecessor required')
    receipt=read(root/'launch.json')
    while True:
        check_pins(config['pins']);terminal=root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('replay predecessor failed')
            prior=read(root/'config.json');plan=Path(prior['plan_root'])
            result=accepted(plan/'training',plan/'suite',config['image'])
            atomic_json(output/'predecessor-accepted.json',result);return
        try:actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('predecessor missing without terminal')
        if actual!=receipt['argv']:raise ValueError('predecessor PID reused')
        atomic_json(output/'progress.json',dict(stage='waiting_real_residency_replay',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def run(config,output):
    check_pins(config['pins']);root=Path(config['gate_root']);plan=read(root/'prepared.json')
    for name in ('config.json','prepared.json','source-manifest.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('complete frozen gate required')
    if plan['image']!=config['image'] or plan['physical_devices']!=[4]:raise ValueError('TP1 fixed image required')
    if any((root/n).exists() for n in ('launch.json','create-attempt.json','run')):raise ValueError('gate already claimed')
    await_predecessor(config,output);wait_resources(config,output)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);check_pins(read(root/'config.json')['pins'])
        receipt=launch(root,plan,plan['create_argv']);atomic_json(output/'dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_zero_delta_gate',time.time()+600)
    if not completed(receipt['container_id'],config['image']):raise ValueError('actual gate completion required')
    audit=verify_gate(root,config['pilot'],config['manifest_sha256']);atomic_json(output/'gate-verified.json',audit)
    pins={str(root/p):file_hash(root/p) for p in ('config.json','run/READY.json','run/terminal.json','run/vllm/result.json','run/vllm/terminal.json')}
    atomic_json(output/'learning-score.json',score(dict(gate_root=str(root),pilot=config['pilot'],manifest_sha256=config['manifest_sha256'],pins=pins)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
