#!/usr/bin/env python3
"""Run registered two-update objective admissions after the zero-delta gate."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_resume import validate_lineage
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read,completed
from queue_intent_finite_npu import verify_gate
from run_intent_matched_pilots import first_step
from queue_intent_continuation import run as continue_step
from compare_intent_learning_update import run as compare
from prepare_intent_utility_pilot import OBJECTIVES,audit


def validate(config):
    plan=Path(config['plan_root']);protocol=read(plan/'protocol.json')
    for p in (plan/'protocol.json',plan/'prepared.json'):
        if config['pins'].get(str(p))!=file_hash(p):raise ValueError('pinned objective protocol required')
    if (protocol['updates_per_arm']!=2 or protocol['initial_gate']!=config['initial_gate']
            or protocol['initial_adapter']!=config['initial_adapter']
            or [a['objective'] for a in protocol['arms']]!=OBJECTIVES):
        raise ValueError('fixed matched objective protocol required')
    reference=read(Path(protocol['reference_training'])/'config-template.json')
    if file_hash(Path(protocol['reference_training'])/'config-template.json')!=protocol['reference_config_sha256']:
        raise ValueError('reference training changed')
    from multimodal_jev.slurp_intent_pilot import load_pilot
    from multimodal_jev.intent_actor_inputs import schedule
    population,_=load_pilot(reference['pilot'],reference['manifest_sha256'],'D')
    expected=[[r['id'] for r in schedule(population,reference['seed'],i,8)] for i in range(2)]
    if expected!=protocol['schedule']:raise ValueError('registered source schedule differs')
    roots=set();source_digest=None;container_names=set()
    for arm in protocol['arms']:
        if len(arm['steps'])!=2:raise ValueError('exactly two updates required')
        previous=None;prior_step=None
        for index,step in enumerate(arm['steps']):
            train,suite=Path(step['training_root']),Path(step['suite_root'])
            for root in (train,suite):
                if str(root) in roots:raise ValueError('distinct output roots required')
                roots.add(str(root))
                for name in ('config-template.json','prepared.json','source-manifest.json'):
                    if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('pinned complete plan required')
                manifest_hash=file_hash(root/'source-manifest.json')
                if source_digest is None:source_digest=manifest_hash
                elif manifest_hash!=source_digest:raise ValueError('matched current source required')
                plan_row=read(root/'prepared.json');argv=plan_row['create_argv'];name=argv[argv.index('--name')+1]
                if name in container_names:raise ValueError('distinct container names required')
                container_names.add(name)
                if plan_row['image']!=config['image']:raise ValueError('fixed image required')
                if any((root/p).exists() for p in ('config.json','launch.json','create-attempt.json','run')):
                    raise ValueError('plan already claimed; inspect existing attempt')
            cfg=read(train/'config-template.json')
            if cfg['objective']!=arm['objective'] or cfg['step']!=index:
                raise ValueError('objective or update index differs')
            if read(suite/'config-template.json')['training_root']!=str(train):raise ValueError('suite pairing differs')
            if previous is None:audit(reference,cfg)
            else:
                validate_lineage(previous,cfg)
                if cfg['previous_root']!=prior_step['training_root'] or cfg['previous_suite']!=prior_step['suite_root']:
                    raise ValueError('noncontiguous objective lineage')
            previous,prior_step=cfg,step
    return protocol


def await_predecessor(config,output):
    root=Path(config['predecessor_controller'])
    for name in ('config.json','launch.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('pinned zero-delta predecessor required')
    receipt=read(root/'launch.json')
    while True:
        check_pins(config['pins']);terminal=root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('zero-delta predecessor failed')
            prior=read(root/'config.json');gate=Path(config['predecessor_gate'])
            if prior['gate_root']!=str(gate):raise ValueError('zero-delta gate differs')
            launch=read(gate/'launch.json')
            if (read(root/'run/dispatched.json')!=launch or launch['image']!=config['image']
                    or not completed(launch['container_id'],config['image'])):
                raise ValueError('actual zero-delta completion required')
            result=verify_gate(gate,prior['pilot'],prior['manifest_sha256'])
            atomic_json(output/'predecessor-accepted.json',result);return
        try:actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('predecessor missing without terminal')
        if actual!=receipt['argv']:raise ValueError('predecessor PID reused')
        atomic_json(output/'progress.json',dict(stage='waiting_zero_delta_completion',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def run(config,output):
    check_pins(config['pins']);protocol=validate(config);await_predecessor(config,output)
    for arm in protocol['arms']:
        for index,step in enumerate(arm['steps']):
            stage=output/(arm['name']+f'-update-{index+1:02d}');stage.mkdir(exist_ok=False)
            if index==0:first_step(config,step,stage)
            else:
                prior=Path(arm['steps'][index-1]['suite_root']);receipt=read(prior/'launch.json')
                continuation=dict(step,previous_suite=str(prior),previous_container=receipt['container_id'],
                    first_poll_after=0.,image=config['image'],lock_directory=config['lock_directory'],pins=dict(config['pins']))
                continuation['pins'][str(prior/'launch.json')]=file_hash(prior/'launch.json')
                atomic_json(stage/'config.json',continuation);continue_step(continuation,stage)
            atomic_json(stage/'learning-comparison.json',compare(step['training_root'],step['suite_root']))
            atomic_json(output/'progress.json',dict(arm=arm['name'],completed_updates=index+1,
                updates_per_arm=2,accepted_suite=step['suite_root'],updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
