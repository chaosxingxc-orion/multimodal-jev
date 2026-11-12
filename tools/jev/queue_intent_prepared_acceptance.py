#!/usr/bin/env python3
"""Dispatch an unclaimed frozen acceptance after resource availability.

A busy resource is a wait before creation, never a reason to restart training.
Unexpected errors or ambiguous create/start outcomes terminate without retries.
"""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_dp4_resume import check_pins,check_devices
from queue_span_native_actor_npu import read,locks,launch,wait_exact,completed
from queue_intent_finite_npu import verify_final


def wait_resources(config,output):
    while True:
        check_pins(config['pins'])
        try:check_devices(config['image'])
        except ValueError as exc:
            if not str(exc).startswith('physical cards reserved by container: '):raise
            atomic_json(output/'progress.json',dict(stage='waiting_resource_before_first_create',reason=str(exc),updated_unix=time.time()))
            time.sleep(600)
        else:return


def run(config,output):
    check_pins(config['pins']);suite=Path(config['suite_root']);plan=read(suite/'prepared.json');cfg=read(suite/'config.json')
    for name in ('config.json','prepared.json','source-manifest.json'):
        if config['pins'].get(str(suite/name))!=file_hash(suite/name):raise ValueError('frozen suite must be pinned')
    if plan['image']!=config['image'] or plan['physical_devices']!=[4] or plan['root']!=str(suite):raise ValueError('suite identity differs')
    parent=Path(cfg['training_root']);receipt=read(parent/'launch.json')
    if not completed(receipt['container_id'],config['image']):raise ValueError('training must already have exited zero')
    if any((suite/p).exists() for p in ('create-attempt.json','launch.json','run')):raise ValueError('suite already claimed; inspect exact existing attempt')
    check_pins(cfg['pins']);check_pins({str(suite/'source'/p):h for p,h in read(suite/'source-manifest.json').items()})
    wait_resources(config,output)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);check_pins(cfg['pins'])
        receipt=launch(suite,plan,plan['create_argv'])
        atomic_json(output/'acceptance-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_independent_acceptance',time.time()+600)
    atomic_json(output/'accepted.json',verify_final(suite))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
