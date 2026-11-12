#!/usr/bin/env python3
"""Three batch32 updates after the exact residency probe; no automatic retry."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_resume import validate_lineage
from multimodal_jev.slurp_intent_pilot import load_pilot
from multimodal_jev.intent_actor_inputs import schedule
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read,completed
from queue_span_residency import verify_results
from run_intent_matched_pilots import first_step
from queue_intent_continuation import run as continue_step
from compare_intent_learning_update import run as compare


def validate(config):
    protocol=read(config['protocol'])
    if (config['pins'].get(config['protocol'])!=file_hash(config['protocol'])
            or protocol['schema']!='intent-finite-batch32-control-v1' or protocol['total_updates']!=3
            or protocol['global_requests_per_update']!=32 or protocol['total_requests']!=96
            or len(config['steps'])!=3):raise ValueError('fixed three-step source budget required')
    prior=None;manifests={};seen=set()
    for index,step in enumerate(config['steps']):
        for kind in ('training','suite'):
            root=Path(step[kind+'_root'])
            if str(root) in seen:raise ValueError('distinct immutable roots required')
            seen.add(str(root))
            for name in ('config-template.json','prepared.json','source-manifest.json'):
                if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('fixed plan required')
            manifest=read(root/'source-manifest.json')
            if index and manifest!=manifests[kind]:raise ValueError('source changed between updates')
            manifests[kind]=manifest
        train=Path(step['training_root']);cfg=read(train/'config-template.json')
        if (cfg['arm']!='D' or cfg['seed']!=317 or cfg['lr']!=1e-4 or cfg['step']!=index
                or cfg['global_requests']!=32 or cfg.get('adam_betas',[.9,.999])!=[.9,.999]
                or read(Path(step['suite_root'])/'config-template.json')['training_root']!=str(train)):
            raise ValueError('matched coverage configuration required')
        if prior is None:
            if (cfg['initial_adapter']!=config['initial_adapter'] or cfg['acceptance_root']!=config['initial_gate']
                    or cfg.get('previous_root') or cfg.get('initial_optimizer_checkpoint')):
                raise ValueError('original initialization required')
            items,_=load_pilot(cfg['pilot'],cfg['manifest_sha256'],'D')
            expected=[[r['id'] for r in schedule(items,317,i,32)] for i in range(3)]
            if protocol['schedule']!=expected:raise ValueError('fixed source order differs')
        else:
            validate_lineage(prior,cfg);previous=config['steps'][index-1]
            if cfg['previous_root']!=previous['training_root'] or cfg['previous_suite']!=previous['suite_root']:
                raise ValueError('contiguous accepted-policy chain required')
        prior=cfg


def await_predecessor(config,output):
    root=Path(config['predecessor_controller'])
    for name in ('launch.json','config.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('exact predecessor required')
    receipt=read(root/'launch.json')
    while True:
        check_pins(config['pins']);terminal=root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('residency predecessor failed')
            cfg=read(root/'config.json');probe=Path(cfg['probe_root']);launched=read(probe/'launch.json')
            if launched['image']!=config['image'] or not completed(launched['container_id'],config['image']):
                raise ValueError('actual predecessor container must complete')
            check_pins({str(probe/'config.json'):launched['config_sha256'],
                str(probe/'source-manifest.json'):launched['source_manifest_sha256']})
            check_pins(read(probe/'config.json')['pins'])
            result=verify_results(probe)
            if result!=read(root/'run/verified.json'):raise ValueError('predecessor evidence differs')
            atomic_json(output/'predecessor-accepted.json',result);return
        try:actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('predecessor missing without terminal')
        if actual!=receipt['argv']:raise ValueError('predecessor PID reused')
        atomic_json(output/'progress.json',dict(stage='waiting_residency_probe',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def run(config,output):
    check_pins(config['pins']);validate(config);await_predecessor(config,output)
    for index,step in enumerate(config['steps']):
        stage=output/f'update-{index+1:04d}';stage.mkdir(exist_ok=False)
        if index==0:first_step(config,step,stage)
        else:
            prior=Path(config['steps'][index-1]['suite_root']);launched=read(prior/'launch.json')
            c=dict(step,previous_suite=str(prior),previous_container=launched['container_id'],first_poll_after=0.,
                image=config['image'],lock_directory=config['lock_directory'],pins=dict(config['pins']))
            c['pins'][str(prior/'launch.json')]=file_hash(prior/'launch.json')
            atomic_json(stage/'config.json',c);continue_step(c,stage)
        atomic_json(stage/'learning-comparison.json',compare(step['training_root'],step['suite_root']))
        atomic_json(output/'progress.json',dict(completed_updates=index+1,total_updates=3,
            source_uses=(index+1)*32,accepted_suite=step['suite_root'],updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
