#!/usr/bin/env python3
"""Exact input gate -> first finite intent update -> independent acceptance."""
import argparse
import json
import math
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.container_pins import validate_bind_pins
from multimodal_jev.intent_gate_identity import verify_population
from multimodal_jev.slurp_intent_pilot import load_pilot
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import wait_exact,locks,launch,read
from queue_intent_runtime_admission import verify_suite_terminal
from freeze_intent_finite_acceptance import freeze


def verify_gate(root,pilot,manifest):
    root=Path(root);receipt=read(root/'launch.json');cfg=read(root/'config.json')
    check_pins({str(root/'config.json'):receipt['config_sha256'],str(root/'source-manifest.json'):receipt['source_manifest_sha256']})
    check_pins(cfg['pins']);check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()})
    for name in ('terminal.json','hf/terminal.json','vllm/terminal.json'):
        if read(root/'run'/name)['exit_code']!=0:raise ValueError('input gate phase failed')
    ready=read(root/'run/READY.json');hf=read(root/'run/hf/result.json');vl=read(root/'run/vllm/result.json')
    if (not ready['passed'] or ready['requests']!=178 or ready['base_id']!=cfg['base_id']
            or ready['adapter_id']!=cfg['adapter_id'] or ready['result_sha256']!=file_hash(root/'run/vllm/result.json')):
        raise ValueError('full intent gate readiness differs')
    artifact=read(Path(cfg['inputs']))
    if cfg['pins'].get(cfg['inputs'])!=file_hash(cfg['inputs']):raise ValueError('processor input must be pinned')
    expected={}
    for arm in ('D','Q'):
        items,_=load_pilot(pilot,manifest,arm)
        expected.update({r['id']+'::'+arm:r['request'].model_input() for r in items})
    if len(expected)!=178:raise ValueError('fixed full 89-source pilot required')
    for result in (hf,vl):
        if not result['passed'] or result['base_id']!=cfg['base_id'] or result['adapter_id']!=cfg['adapter_id']:
            raise ValueError('input policy differs')
        verify_population(artifact,expected,result['records'],cfg['base_id']+':'+cfg['adapter_id'])
    refs={r['id']:r for r in hf['records']};worst=0.;low=1.;high=1.
    for row in vl['records']:
        ref=refs[row['id']]
        if row['request_id']!=ref['request_id'] or row['request_identity']!=ref['request_identity']:
            raise ValueError('backend identities differ')
        lower=upper=0.
        for a,b in zip(ref['candidate_logprobs'],row['candidate_logprobs'],strict=True):
            delta=[x-y for x,y in zip(a,b,strict=True)]
            worst=max(worst,max(map(abs,delta)));lower+=min(delta);upper+=max(delta)
        if worst>.02 or lower<math.log(.98) or upper>math.log(1.02):raise ValueError('full candidate parity failed')
        low=min(low,math.exp(lower));high=max(high,math.exp(upper))
    return dict(passed=True,requests=len(expected),maximum_logprob_error=worst,joint_ratio_range=[low,high],
        scope='Full pilot input/policy parity, not held-out efficacy.')


def freeze_training(plan):
    root=Path(plan['root'])
    if any((root/p).exists() for p in ('config.json','launch.json','create-attempt.json','run')):
        raise ValueError('training already claimed; never overwrite')
    check_pins({str(root/'config-template.json'):plan['config_template_sha256'],str(root/'source-manifest.json'):plan['source_manifest_sha256']})
    cfg=read(root/'config-template.json');check_pins(cfg['pins'])
    check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()})
    for path in plan['pending_pins']:cfg['pins'][path]=file_hash(path)
    atomic_json(root/'bind-pins.json',validate_bind_pins(plan['create_argv'],cfg['pins']))
    atomic_json(root/'config.json',cfg)
    return cfg


def verify_final(root):
    root=Path(root)
    if read(root/'controller-terminal.json')['exit_code']!=0:raise ValueError('finite acceptance controller failed')
    accepted=read(root/'ACCEPTED.json');verified=read(root/'verified.json')
    if (not accepted['passed'] or accepted['verified_sha256']!=file_hash(root/'verified.json')
            or not verified['passed'] or not verified['exact_export_reload'] or verified['full_gate_requests']!=178):
        raise ValueError('complete finite acceptance required')
    check_pins(verified['files']);check_pins(read(root/'config.json')['pins'])
    return accepted


def await_gate_dispatch(config,output):
    """Follow the authenticated existing queue; never invent a replacement."""
    queue=Path(config['gate_queue']);launch_record=read(queue/'launch.json')
    if config['pins'].get(str(queue/'launch.json'))!=file_hash(queue/'launch.json'):
        raise ValueError('pinned gate queue identity required')
    while True:
        terminal=queue/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('gate dispatch queue failed')
            receipt=read(queue/'run/dispatched.json');gate=Path(config['gate_root'])
            if receipt!=read(gate/'launch.json') or receipt['root']!=str(gate) or receipt['image']!=config['image']:
                raise ValueError('gate dispatch receipt differs')
            atomic_json(output/'gate-dispatch-observed.json',receipt)
            return receipt
        try:
            actual=(Path('/proc')/str(launch_record['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('gate queue missing without terminal')
        if actual!=launch_record['argv']:raise ValueError('gate queue PID reused')
        atomic_json(output/'progress.json',dict(stage='waiting_existing_gate_queue',pid=launch_record['pid'],updated_unix=time.time()))
        time.sleep(600)


def run(config,output):
    check_pins(config['pins']);gate=Path(config['gate_root'])
    if config.get('gate_queue'):
        receipt=await_gate_dispatch(config,output);first_poll=time.time()+600
    else:
        receipt=read(gate/'launch.json');first_poll=config['first_poll_after']
        if receipt['container_id']!=config['gate_container']:raise ValueError('exact gate container differs')
    wait_exact(receipt['container_id'],config['image'],output,'waiting_intent_gate',first_poll)
    verify_suite_terminal(Path(config['native_suite']))
    plan=read(config['training_plan']);parent=read(Path(plan['root'])/'config-template.json')
    audit=verify_gate(gate,parent['pilot'],parent['manifest_sha256']);atomic_json(output/'input-gate-verified.json',audit)
    from score_intent_learning_policy import run as score
    score_config=dict(gate_root=str(gate),pilot=parent['pilot'],manifest_sha256=parent['manifest_sha256'],
        pins={str(gate/p):file_hash(gate/p) for p in ('config.json','run/READY.json','run/terminal.json','run/vllm/result.json','run/vllm/terminal.json')})
    atomic_json(output/'initial-learning-score-config.json',score_config)
    atomic_json(output/'initial-learning-score.json',score(score_config))
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins'])
        if plan['image']!=config['image'] or plan['physical_devices']!=[4,5,6,7]:raise ValueError('training topology differs')
        freeze_training(plan);root=Path(plan['root']);receipt=launch(root,plan,plan['create_argv'])
        atomic_json(output/'training-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_finite_update',time.time()+600)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);suite=Path(config['suite_root']);suite_plan=read(suite/'prepared.json')
        if suite_plan['image']!=config['image'] or suite_plan['physical_devices']!=[4]:raise ValueError('suite topology differs')
        freeze(suite);receipt=launch(suite,suite_plan,suite_plan['create_argv'])
        atomic_json(output/'acceptance-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_finite_acceptance',time.time()+600)
    atomic_json(output/'accepted.json',verify_final(suite))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time(),scope='First finite intent update accepted, not continuous training or held-out improvement.'))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
