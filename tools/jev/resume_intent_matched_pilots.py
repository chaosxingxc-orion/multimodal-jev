#!/usr/bin/env python3
"""Resume a superseded Q/P scheduler at a trained, unclaimed suite boundary.

Existing accepted pairs are independently authenticated. No training attempt is
repeated, no old output is rewritten, and future claimed work rejects takeover.
"""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read,completed,locks,launch,wait_exact
from queue_intent_finite_npu import verify_final
from freeze_intent_finite_acceptance import freeze
from queue_intent_prepared_acceptance import wait_resources
from queue_intent_continuation import run as continue_step
from run_intent_matched_pilots import validate,first_step
from compare_intent_learning_update import run as compare


def unclaimed(root):
    root=Path(root)
    if any((root/n).exists() for n in ('config.json','create-attempt.json','launch.json','run')):
        raise ValueError('future workload already claimed: '+str(root))


def accepted_pair(step,image,arm,index):
    for key in ('training_root','suite_root'):
        root=Path(step[key]);receipt=read(root/'launch.json')
        if receipt['image']!=image or not completed(receipt['container_id'],image):
            raise ValueError('actual completed prior pair required')
        check_pins({str(root/'config.json'):receipt['config_sha256'],
                    str(root/'source-manifest.json'):receipt['source_manifest_sha256']})
    suite=Path(step['suite_root']);audit=verify_final(suite);proof=read(suite/'verified.json')
    if (proof['root']!=step['training_root'] or proof['suite']!=str(suite)
            or proof['arm']!=arm or proof['updates']!=index+1):
        raise ValueError('prior accepted pair identity differs')
    return audit


def boundary(config,*,require_stopped=True):
    check_pins(config['pins']);validate(config)
    old=Path(config['resume_from_controller']);previous=read(old/'config.json')
    for name in ('config.json','launch.json','source-manifest.json'):
        if config['pins'].get(str(old/name))!=file_hash(old/name):
            raise ValueError('exact superseded controller required')
    if {k:v for k,v in config.items() if k not in ('pins','resume_from_controller')}!={k:v for k,v in previous.items() if k not in ('pins','resume_from_controller')}:
        raise ValueError('frozen matched protocol changed')
    check_pins(previous['pins'])
    if (old/'run/terminal.json').exists():raise ValueError('original controller already terminal')
    progress=read(old/'run/progress.json')
    if (progress['arm']!='Q' or type(progress['completed_updates'])!=int
            or not 1<=progress['completed_updates']<12 or progress['total_updates']!=12):
        raise ValueError('supported partial Q boundary required')
    count=progress['completed_updates'];steps=config['arms'][0]['steps']
    if progress['accepted_suite']!=steps[count-1]['suite_root']:raise ValueError('accepted prefix differs')
    receipt=read(old/'launch.json');path=Path('/proc')/str(receipt['pid'])/'cmdline'
    live=path.exists() and path.read_bytes().rstrip(b'\0').decode().split('\0')==receipt['argv']
    if require_stopped:
        proof=read(old/'superseded.json')
        if live or proof.get('observed_process_exit') is not True or proof.get('pid')!=receipt['pid']:
            raise ValueError('original exact supervisor must be stopped before takeover')
    elif not live:raise ValueError('preflight requires exact original live supervisor')
    stage=old/'run'/f'Q-update-{count+1:04d}'
    if read(stage/'progress.json')['stage']!='waiting_resource_before_first_create':
        raise ValueError('original controller is not waiting for resources')
    step=steps[count];training=Path(step['training_root']);tr=read(training/'launch.json')
    origin=old;origin_cfg=previous;visited=set()
    while not (origin/'run'/stage.name/'training-dispatched.json').exists():
        if str(origin) in visited or 'resume_from_controller' not in origin_cfg:
            raise ValueError('missing original training dispatch provenance')
        visited.add(str(origin));ancestor=Path(origin_cfg['resume_from_controller'])
        for name in ('config.json','launch.json','superseded.json'):
            if origin_cfg['pins'].get(str(ancestor/name))!=file_hash(ancestor/name):
                raise ValueError('pinned superseded ancestor required')
        if read(ancestor/'superseded.json').get('observed_process_exit') is not True:
            raise ValueError('ancestor not superseded')
        ancestor_cfg=read(ancestor/'config.json')
        if ancestor_cfg['arms']!=config['arms']:raise ValueError('ancestor workload differs')
        origin,origin_cfg=ancestor,ancestor_cfg
    if (tr!=read(origin/'run'/stage.name/'training-dispatched.json') or tr['image']!=config['image']
            or tr['physical_devices']!=[4,5,6,7] or not completed(tr['container_id'],config['image'])):
        raise ValueError('exact pending training must have exited zero')
    if (stage/'acceptance-dispatched.json').exists():raise ValueError('acceptance already dispatched')
    unclaimed(step['suite_root'])
    for future in steps[count+1:]+config['arms'][1]['steps']:
        for key in ('training_root','suite_root'):unclaimed(future[key])
    return count


def run(config,out):
    count=boundary(config)
    for arm in config['arms']:
        for index,step in enumerate(arm['steps']):
            stage=out/(arm['arm']+f'-update-{index+1:04d}');stage.mkdir(exist_ok=False)
            if arm['arm']=='Q' and index<count:
                atomic_json(stage/'inherited-accepted.json',accepted_pair(step,config['image'],arm['arm'],index))
            elif arm['arm']=='Q' and index==count:
                suite=Path(step['suite_root']);plan=read(suite/'prepared.json')
                wait_resources(config,stage)
                with locks(Path(config['lock_directory'])):
                    boundary(config);freeze(suite)
                    receipt=launch(suite,plan,plan['create_argv'])
                    atomic_json(stage/'acceptance-dispatched.json',receipt)
                wait_exact(receipt['container_id'],config['image'],stage,'waiting_resumed_acceptance',time.time()+600)
                atomic_json(stage/'accepted.json',accepted_pair(step,config['image'],arm['arm'],index))
            elif index==0:first_step(config,step,stage)
            else:
                previous=Path(arm['steps'][index-1]['suite_root']);receipt=read(previous/'launch.json')
                cfg=dict(step,previous_suite=str(previous),previous_container=receipt['container_id'],first_poll_after=0.,
                    image=config['image'],lock_directory=config['lock_directory'],pins=dict(config['pins']))
                cfg['pins'][str(previous/'launch.json')]=file_hash(previous/'launch.json')
                atomic_json(stage/'config.json',cfg);continue_step(cfg,stage)
            atomic_json(stage/'learning-comparison.json',compare(step['training_root'],step['suite_root']))
            atomic_json(out/'progress.json',dict(arm=arm['arm'],completed_updates=index+1,total_updates=12,
                accepted_suite=step['suite_root'],updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
