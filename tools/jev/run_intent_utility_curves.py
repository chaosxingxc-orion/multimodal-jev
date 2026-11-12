#!/usr/bin/env python3
"""Continue every registered objective arm after actual engineering acceptance."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_resume import validate_lineage
from multimodal_jev.intent_actor_inputs import schedule
from multimodal_jev.slurp_intent_pilot import load_pilot
from queue_span_native_actor_npu import read,completed,wait_exact,launch
from queue_span_dp4_resume import check_pins
from queue_intent_finite_npu import verify_final
from queue_intent_continuation import run as continue_step
from compare_intent_learning_update import run as compare
from prepare_intent_utility_curves import validate_protocol
from wait_intent_priority_tail import wait_tail


def validate(config):
    check_pins(config['pins']);root=Path(config['plan_root']);protocol=read(root/'protocol.json')
    for p in (root/'protocol.json',root/'prepared.json',Path(protocol['pilot_protocol'])):
        if config['pins'].get(str(p))!=file_hash(p):raise ValueError('pinned complete curve protocol required')
    pilot=read(protocol['pilot_protocol']);validate_protocol(protocol,pilot)
    check_pins(read(root/'prepared.json')['pins'])
    items,_=load_pilot(protocol['pilot'],protocol['manifest_sha256'],'D')
    if protocol['schedule']!=[[r['id'] for r in schedule(items,protocol['seed'],i,8)] for i in range(12)]:
        raise ValueError('full fixed source schedule differs')
    names=set();source=None
    for arm in protocol['arms']:
        previous=None;prior=None
        for index,step in enumerate(arm['steps']):
            for key,devices in (('training_root',[4,5,6,7]),('suite_root',[4])):
                p=Path(step[key]);plan=read(p/'prepared.json');cfg=read(p/'config-template.json')
                if plan['image']!=config['image'] or plan['physical_devices']!=devices or plan['root']!=str(p):
                    raise ValueError('fixed image and topology required')
                check_pins({str(p/'config-template.json'):plan['config_template_sha256'],str(p/'source-manifest.json'):plan['source_manifest_sha256']})
                check_pins(cfg['pins']);check_pins({str(p/'source'/n):h for n,h in read(p/'source-manifest.json').items()})
                if source is None:source=plan['source_manifest_sha256']
                if plan['source_manifest_sha256']!=source:raise ValueError('matched frozen computation required')
                name=plan['create_argv'][plan['create_argv'].index('--name')+1]
                if name in names:raise ValueError('distinct containers required')
                names.add(name)
                if index>=2 and any((p/n).exists() for n in ('config.json','launch.json','create-attempt.json','run')):
                    raise ValueError('extension already claimed; inspect original attempt')
            cfg=read(Path(step['training_root'])/'config-template.json')
            if cfg['step']!=index or cfg['objective']!=arm['objective']:raise ValueError('objective/cursor differs')
            if previous is not None:
                validate_lineage(previous,cfg)
                if cfg['previous_root']!=prior['training_root'] or cfg['previous_suite']!=prior['suite_root']:
                    raise ValueError('noncontiguous arm')
            if read(Path(step['suite_root'])/'config-template.json')['training_root']!=step['training_root']:
                raise ValueError('suite pairing differs')
            previous,prior=cfg,step
    return protocol


def authenticate_pair(step,image,updates):
    for key in ('training_root','suite_root'):
        root=Path(step[key]);receipt=read(root/'launch.json')
        if receipt['image']!=image or not completed(receipt['container_id'],image):
            raise ValueError('actual successful training and suite required')
        check_pins({str(root/'config.json'):receipt['config_sha256'],str(root/'source-manifest.json'):receipt['source_manifest_sha256']})
    suite=Path(step['suite_root']);accepted=verify_final(suite);verified=read(suite/'verified.json')
    if verified['root']!=step['training_root'] or verified['suite']!=str(suite) or verified['updates']!=updates or verified['arm']!='D':
        raise ValueError('accepted update identity differs')
    return accepted


def verify_replay(config,out):
    root=Path(config['replay_root']);plan=read(root/'prepared.json')
    if (plan['image']!=config['image'] or plan['physical_devices']!=[]
            or plan['root']!=str(root) or '--device' in plan['create_argv']):
        raise ValueError('fixed no-device replay verifier required')
    check_pins({str(root/'config.json'):plan['config_sha256'],str(root/'source-manifest.json'):plan['source_manifest_sha256']})
    check_pins(read(root/'config.json')['pins'])
    check_pins({str(root/'source'/n):h for n,h in read(root/'source-manifest.json').items()})
    receipt=launch(root,plan,plan['create_argv']);atomic_json(out/'replay-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],out,'waiting_exact_correctness_replay',time.time()+600)
    result=read(root/'run/verified.json')
    replay_config=read(root/'config.json')
    if (read(root/'run/terminal.json')['exit_code']!=0 or result.get('passed') is not True
            or result['reference']!=replay_config['reference'] or result['replay']!=replay_config['replay']
            or [r['rank'] for r in result['ranks']]!=list(range(4))
            or any(r['updates']!=2 or r['exact_master_and_adam'] is not True for r in result['ranks'])):
        raise ValueError('complete independent correctness replay required')
    check_pins(result['files']);atomic_json(out/'replay-accepted.json',dict(result_sha256=file_hash(root/'run/verified.json')))


def run(config,out):
    protocol=validate(config)
    wait_tail(config,out)
    for arm in protocol['arms']:
        for i,step in enumerate(arm['steps'][:2]):authenticate_pair(step,config['image'],i+1)
    verify_replay(config,out)
    curves={}
    for arm in protocol['arms']:
        points=[]
        for index,step in enumerate(arm['steps']):
            stage=out/(arm['name']+f'-update-{index+1:02d}');stage.mkdir(exist_ok=False)
            if index>=2:
                prior=Path(arm['steps'][index-1]['suite_root']);receipt=read(prior/'launch.json')
                c=dict(step,previous_suite=str(prior),previous_container=receipt['container_id'],
                    first_poll_after=0.,image=config['image'],lock_directory=config['lock_directory'],pins=dict(config['pins']))
                c['pins'][str(prior/'launch.json')]=file_hash(prior/'launch.json')
                atomic_json(stage/'config.json',c);continue_step(c,stage)
            authenticate_pair(step,config['image'],index+1)
            result=compare(step['training_root'],step['suite_root'])
            atomic_json(stage/'learning-comparison.json',result)
            if index==0:points.append(dict(update=0,metrics=result['full_before']['reports']['D']['metrics']))
            points.append(dict(update=index+1,metrics=result['full_after']['reports']['D']['metrics']))
            atomic_json(out/'progress.json',dict(arm=arm['name'],completed_updates=index+1,total_updates=12,
                accepted_suite=step['suite_root'],updated_unix=time.time()))
        curves[arm['name']]=points
        atomic_json(out/(arm['name']+'-curve.json'),dict(points=points,role='learning'))
    atomic_json(out/'curves.json',dict(protocol_sha256=file_hash(Path(config['plan_root'])/'protocol.json'),
        curves=curves,role='learning',scope='Complete learning curves only; no independent efficacy or transfer claim.'))


def authenticate_complete(config,root):
    """Full successful curve dependency for FSC; never select by scores."""
    prior=read(root/'config.json');check_pins(prior['pins'])
    if prior['image']!=config['image']:raise ValueError('curve image differs')
    path=Path(prior['plan_root'])/'protocol.json';protocol=read(path)
    validate_protocol(protocol,read(protocol['pilot_protocol']))
    curves=read(root/'run/curves.json')
    if (curves['protocol_sha256']!=file_hash(path) or curves['role']!='learning'
            or set(curves['curves'])!={a['name'] for a in protocol['arms']}
            or any([p['update'] for p in points]!=list(range(13)) for points in curves['curves'].values())):
        raise ValueError('complete all-arm learning curves required')
    accepted=[]
    for arm in protocol['arms']:
        for index,step in enumerate(arm['steps']):
            accepted.append(authenticate_pair(step,config['image'],index+1))
    replay=Path(prior['replay_root']);receipt=read(replay/'launch.json')
    if receipt!=read(root/'run/replay-dispatched.json') or not completed(receipt['container_id'],config['image']):
        raise ValueError('actual correctness replay completion required')
    saved=read(root/'run/replay-accepted.json')
    if saved['result_sha256']!=file_hash(replay/'run/verified.json'):raise ValueError('correctness replay changed')
    check_pins(read(replay/'run/verified.json')['files'])
    return dict(passed=True,accepted_pairs=len(accepted),curves_sha256=file_hash(root/'run/curves.json'),
                scope='All36 actual training/suite pairs and replay verified; no score-based selection.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
