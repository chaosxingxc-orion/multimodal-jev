#!/usr/bin/env python3
"""Execute a predeclared finite update budget with every independent audit intact."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_resume import validate_lineage
from queue_span_native_actor_npu import read
from queue_span_dp4_resume import check_pins
from queue_intent_continuation import run as run_step
from compare_intent_learning_update import run as compare


def validate(config):
    steps=config['steps']
    if not steps or type(config['total_updates'])!=int:raise ValueError('fixed nonempty update budget required')
    previous_root=config['previous_training'];previous_suite=config['previous_suite'];seen=set()
    previous=read(Path(previous_root)/'config-template.json')
    if (type(config['initial_updates'])!=int or config['initial_updates']!=previous['step']+1
            or config['total_updates']!=config['initial_updates']+len(steps)):
        raise ValueError('initial cursor and bounded budget must match actual lineage')
    for step in steps:
        root=Path(step['training_root']);suite=Path(step['suite_root'])
        if root==suite or str(root) in seen or str(suite) in seen:raise ValueError('distinct update roots required')
        seen.update((str(root),str(suite)))
        for base in (root,suite):
            for name in ('prepared.json','config-template.json','source-manifest.json'):
                if config['pins'].get(str(base/name))!=file_hash(base/name):raise ValueError('all future plans must be pinned')
        cfg=read(root/'config-template.json');validate_lineage(previous,cfg)
        if cfg['previous_root']!=previous_root or cfg['previous_suite']!=previous_suite:
            raise ValueError('noncontiguous predecessor chain')
        if read(suite/'config-template.json')['training_root']!=str(root):raise ValueError('suite pair differs')
        previous_root=str(root);previous_suite=str(suite);previous=cfg
    if previous['step']+1!=config['total_updates']:raise ValueError('final update budget differs')


def run(config,output):
    check_pins(config['pins']);validate(config)
    previous_suite=config['previous_suite']
    for i,step in enumerate(config['steps']):
        check_pins(config['pins'])
        stage=output/f'update-{i+config["initial_updates"]+1:04d}';stage.mkdir(exist_ok=False)
        c=dict(step,previous_suite=previous_suite,image=config['image'],lock_directory=config['lock_directory'],pins=dict(config['pins']))
        if i==0:c['previous_queue']=config['previous_queue']
        else:
            launch=Path(previous_suite)/'launch.json';receipt=read(launch)
            c.update(previous_container=receipt['container_id'],first_poll_after=0.)
            c['pins'][str(launch)]=file_hash(launch)
        atomic_json(stage/'config.json',c)
        run_step(c,stage)
        atomic_json(stage/'learning-comparison.json',compare(step['training_root'],step['suite_root']))
        atomic_json(output/'progress.json',dict(completed_updates=i+config['initial_updates']+1,
            total_updates=config['total_updates'],accepted_suite=step['suite_root'],updated_unix=time.time()))
        previous_suite=step['suite_root']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time(),scope='Bounded learning pilot only; independent development/transfer efficacy remains unproven.'))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
