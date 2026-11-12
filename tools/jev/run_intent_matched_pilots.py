#!/usr/bin/env python3
"""Run frozen Q then P pilots only after the exact D12 controller succeeds."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_resume import validate_lineage
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read,completed,locks,launch,wait_exact
from queue_intent_finite_npu import freeze_training,verify_final,verify_gate
from freeze_intent_finite_acceptance import freeze as freeze_suite
from queue_intent_prepared_acceptance import wait_resources
from queue_intent_continuation import run as continue_step
from compare_intent_learning_update import run as compare


def validate(config):
    if [a['arm'] for a in config['arms']]!=['Q','P']:raise ValueError('fixed Q then P order required')
    previous_configs=[];all_roots=set()
    for arm in config['arms']:
        if len(arm['steps'])!=12:raise ValueError('twelve matched updates per arm required')
        prior=None
        for index,step in enumerate(arm['steps']):
            root,suite=Path(step['training_root']),Path(step['suite_root'])
            for base in (root,suite):
                if str(base) in all_roots:raise ValueError('distinct immutable roots required')
                all_roots.add(str(base))
                for name in ('prepared.json','config-template.json','source-manifest.json'):
                    if config['pins'].get(str(base/name))!=file_hash(base/name):raise ValueError('all plans must be pinned')
            cfg=read(root/'config-template.json')
            if cfg['arm']!=arm['arm'] or cfg['step']!=index or cfg['seed']!=317 or cfg['lr']!=1e-4:
                raise ValueError('matched fixed arm/step/seed/rate required')
            if read(suite/'config-template.json')['training_root']!=str(root):raise ValueError('suite pairing differs')
            if prior:
                validate_lineage(prior,cfg)
                before=arm['steps'][index-1]
                if cfg['previous_root']!=before['training_root'] or cfg['previous_suite']!=before['suite_root']:
                    raise ValueError('noncontiguous matched chain')
            else:
                if cfg['initial_adapter']!=config['initial_adapter'] or cfg['acceptance_root']!=config['initial_gate']:
                    raise ValueError('same fixed initialization required')
                previous_configs.append(cfg)
            prior=cfg
    a,b=previous_configs
    if {k:v for k,v in a.items() if k not in ('arm','pins')}!={k:v for k,v in b.items() if k not in ('arm','pins')}:
        raise ValueError('Q/P non-reward training configuration differs')


def await_d12(config,output):
    root=Path(config['predecessor_controller']);receipt=read(root/'launch.json')
    if config['pins'].get(str(root/'launch.json'))!=file_hash(root/'launch.json'):
        raise ValueError('pinned exact D controller required')
    while True:
        check_pins(config['pins'])
        terminal=root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('D12 predecessor failed; no matched pilot dispatch')
            progress=read(root/'run/progress.json')
            if progress['completed_updates']!=12 or progress['total_updates']!=12 or progress['accepted_suite']!=config['predecessor_suite']:
                raise ValueError('complete D12 terminal required')
            suite=Path(config['predecessor_suite']);launch_record=read(suite/'launch.json')
            if launch_record['image']!=config['image'] or not completed(launch_record['container_id'],config['image']):
                raise ValueError('actual D12 acceptance exit required')
            accepted=verify_final(suite);verified=read(suite/'verified.json')
            if verified['updates']!=12 or verified['arm']!='D':raise ValueError('D12 acceptance differs')
            atomic_json(output/'predecessor-accepted.json',accepted)
            return
        path=Path('/proc')/str(receipt['pid'])/'cmdline'
        try:actual=path.read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('D12 controller missing without terminal')
        if actual!=receipt['argv']:raise ValueError('D12 controller PID reused')
        atomic_json(output/'progress.json',dict(stage='waiting_exact_D12',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def first_step(config,step,output):
    root=Path(step['training_root']);suite=Path(step['suite_root']);cfg=read(root/'config-template.json')
    initial=Path(config['initial_gate']);receipt=read(initial/'launch.json')
    if receipt['image']!=config['image'] or not completed(receipt['container_id'],config['image']):
        raise ValueError('actual initialized policy gate required')
    verify_gate(initial,cfg['pilot'],cfg['manifest_sha256'])
    wait_resources(config,output)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);plan=read(root/'prepared.json');freeze_training(plan)
        if plan['image']!=config['image'] or plan['physical_devices']!=[4,5,6,7]:raise ValueError('DP4 required')
        receipt=launch(root,plan,plan['create_argv']);atomic_json(output/'training-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_matched_first_update',time.time()+600)
    wait_resources(config,output)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);plan=read(suite/'prepared.json');freeze_suite(suite)
        if plan['image']!=config['image'] or plan['physical_devices']!=[4]:raise ValueError('TP1 gate required')
        receipt=launch(suite,plan,plan['create_argv']);atomic_json(output/'acceptance-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_matched_first_acceptance',time.time()+600)
    atomic_json(output/'accepted.json',verify_final(suite))


def run(config,output):
    check_pins(config['pins']);validate(config);await_d12(config,output)
    for arm in config['arms']:
        for index,step in enumerate(arm['steps']):
            stage=output/(arm['arm']+f'-update-{index+1:04d}');stage.mkdir(exist_ok=False)
            if index==0:first_step(config,step,stage)
            else:
                previous=Path(arm['steps'][index-1]['suite_root']);receipt=read(previous/'launch.json')
                c=dict(step,previous_suite=str(previous),previous_container=receipt['container_id'],first_poll_after=0.,
                    image=config['image'],lock_directory=config['lock_directory'],pins=dict(config['pins']))
                c['pins'][str(previous/'launch.json')]=file_hash(previous/'launch.json')
                atomic_json(stage/'config.json',c);continue_step(c,stage)
            atomic_json(stage/'learning-comparison.json',compare(step['training_root'],step['suite_root']))
            atomic_json(output/'progress.json',dict(arm=arm['arm'],completed_updates=index+1,total_updates=12,
                accepted_suite=step['suite_root'],updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time(),scope='Matched learning pilots only; not independent efficacy.'))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
