#!/usr/bin/env python3
"""Dispatch the prepared native actor and independent gate once, polling at 10m.

Exact containers only. A queue success means acceptance dispatched, not accepted.
Uncertain creation/start states never cause a replacement container.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import math
from pathlib import Path
import re
import time

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from queue_span_dp4_resume import remote,inspect,check_pins,check_devices,CARDS


def read(path):return json.loads(Path(path).read_text())


def completed(cid,image):
    row=inspect([cid])[cid]
    if row['Image']!=image:raise ValueError('container image differs')
    state=row['State']
    if state['Status']=='running':return False
    if state['Status']!='exited' or state['ExitCode']!=0:raise ValueError('predecessor failed or unexpected state: '+cid)
    return True


def wait_exact(cid,image,output,stage,not_before=0.):
    if time.time()<not_before:time.sleep(not_before-time.time())
    while True:
        done=completed(cid,image)
        atomic_json(output/'progress.json',dict(stage=stage,container_id=cid,terminal=done,updated_unix=time.time()))
        if done:return
        time.sleep(600)


def verify_native_gate(root,launch):
    check_pins({str(root/'config.json'):launch['config_sha256'],str(root/'source-manifest.json'):launch['source_manifest_sha256']})
    config=read(root/'config.json');check_pins(config['pins'])
    check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()})
    for p in ('terminal.json','hf/terminal.json','vllm/terminal.json'):
        if read(root/'run'/p)['exit_code']!=0:raise ValueError('native gate phase failed')
    ready=read(root/'run/READY.json');hf=read(root/'run/hf/result.json');vl=read(root/'run/vllm/result.json')
    if not ready['passed'] or ready['requests']!=44 or ready['result_sha256']!=file_hash(root/'run/vllm/result.json'):
        raise ValueError('native gate readiness differs')
    for result in (hf,vl):
        if not result['passed'] or result['base_id']!=config['base_id'] or result['adapter_id']!=config['adapter_id']:
            raise ValueError('native gate policy differs')
    references={r['id']:r for r in hf['records']}
    if len(references)!=44 or len(vl['records'])!=44 or {r['id'] for r in vl['records']}!=set(references):
        raise ValueError('complete native gate population required')
    worst=0.;low=1.;high=1.
    for row in vl['records']:
        ref=references[row['id']]
        if row['request_id']!=ref['request_id'] or row['request_identity']!=ref['request_identity']:
            raise ValueError('native gate exact request differs')
        a,b=ref['candidate_logprobs'],row['candidate_logprobs']
        if not a or len(a)!=len(b):raise ValueError('candidate support differs')
        lower=upper=0.
        for left,right in zip(a,b,strict=True):
            if len(left)!=len(right) or len(left)<2 or not all(math.isfinite(x) for x in left+right):
                raise ValueError('finite candidate support required')
            if any(abs(math.log(sum(math.exp(x) for x in values)))>1e-6 for values in (left,right)):
                raise ValueError('candidate probabilities not normalized')
            delta=[x-y for x,y in zip(left,right,strict=True)];worst=max(worst,max(map(abs,delta)))
            lower+=min(delta);upper+=max(delta)
        if worst>.02 or lower<math.log(.98) or upper>math.log(1.02):raise ValueError('native candidate parity failed')
        low=min(low,math.exp(lower));high=max(high,math.exp(upper))
    return dict(passed=True,requests=44,maximum_logprob_error=worst,joint_ratio_range=[low,high],
        scope='Exact frozen 44 native-input requests only; no native task training or accuracy claim.')


def locks(base):
    stack=ExitStack()
    try:
        for card in CARDS:
            handle=stack.enter_context((base/f'.physical-card{card}-launch.lock').open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:stack.close();raise
    return stack


def launch(root,plan,argv):
    if any((root/p).exists() for p in ('launch.json','create-attempt.json','run')):
        raise ValueError('already claimed; inspect existing attempt instead of restarting')
    check_devices(plan['image'])
    atomic_json(root/'create-attempt.json',dict(argv=argv,time=time.time()))
    result=remote(argv);atomic_json(root/'create.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode or not re.fullmatch('[a-f0-9]{64}',result.stdout.strip()):raise ValueError('creation failed/ambiguous; no retry')
    cid=result.stdout.strip()
    receipt=dict(root=str(root),image=plan['image'],physical_devices=plan['physical_devices'],argv=argv,
        container_id=cid,launch_exit_code=0,config_sha256=file_hash(root/'config.json'),source_manifest_sha256=file_hash(root/'source-manifest.json'))
    atomic_json(root/'launch.json',receipt)
    check_devices(plan['image'])
    result=remote(['docker','start',cid]);atomic_json(root/'start.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr));result.check_returncode()
    completed(cid,plan['image']) # Authoritative launch acceptance; running or exited zero.
    return receipt


def freeze_suite(config):
    root=Path(config['native_root']);suite=Path(config['suite_root']);parent=read(root/'config.json');receipt=read(root/'launch.json')
    if any((suite/p).exists() for p in ('config.json','launch.json','create-attempt.json','run')):raise ValueError('suite already claimed')
    check_pins({str(root/'config.json'):receipt['config_sha256'],str(root/'source-manifest.json'):receipt['source_manifest_sha256']})
    check_pins(parent['pins']);check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()})
    prepared=read(suite/'prepared.json');manifest=read(suite/'source-manifest.json')
    if file_hash(suite/'source-manifest.json')!=prepared['source_manifest_sha256']:raise ValueError('suite source manifest changed')
    pins={str(suite/'source'/p):h for p,h in manifest.items()};check_pins(pins)
    adapter_id,files=checkpoint_identity(root/'run/adapter');pins.update({str(root/'run/adapter'/p):h for p,h in files.items()})
    for rank in range(4):
        path=root/'run'/f'rank-{rank}';result=read(path/'result.json')
        if (read(path/'terminal.json')['exit_code']!=0 or result['status']!='passed' or result['rank']!=rank
                or result['world_size']!=4 or result['optimizer_updates']!=1 or result['adapter_id']!=adapter_id):raise ValueError('successful native ranks required')
        for name in ('result.json','terminal.json','selection.json','optimizer-local.pt','after.json','native-actor.json'):
            pins[str(path/name)]=file_hash(path/name)
    for path in (root/'launch.json',root/'config.json',root/'source-manifest.json',Path(prepared['identity'])):pins[str(path)]=file_hash(path)
    atomic_json(suite/'config.json',dict(model=parent['model_path'],adapter=str(root/'run/adapter'),
        identity=prepared['identity'],runtime_profile=parent['runtime_profile'],native_root=str(root),reference_root=config['reference_root'],pins=pins))
    return prepared


def run(config,output):
    check_pins(config['pins'])
    gate=Path(config['gate_root']);gate_launch=read(gate/'launch.json')
    if gate_launch['container_id']!=config['gate_container']:raise ValueError('gate identity differs')
    wait_exact(config['gate_container'],config['image'],output,'waiting_native_input_gate',config['first_poll_after'])
    audit=verify_native_gate(gate,gate_launch);atomic_json(output/'native-gate-verified.json',audit)
    resume=Path(config['resume_suite'])
    if read(resume/'controller-terminal.json')['exit_code']!=0:raise ValueError('recovery acceptance failed')
    accepted=read(resume/'ACCEPTED.json')
    if not accepted['passed'] or accepted['verified_sha256']!=file_hash(resume/'verified.json'):raise ValueError('recovery acceptance changed')
    check_pins(read(resume/'verified.json')['files'])
    plan=read(config['native_plan']);root=Path(config['native_root'])
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins'])
        if plan['root']!=str(root) or plan['image']!=config['image'] or plan['physical_devices']!=list(CARDS):raise ValueError('native plan differs')
        check_pins({str(root/'config.json'):plan['config_sha256'],str(root/'source-manifest.json'):plan['source_manifest_sha256']})
        check_pins(read(root/'config.json')['pins']);check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()})
        if plan['argv'][:3]!=['docker','run','-d']:raise ValueError('prepared launch form differs')
        native=launch(root,plan,['docker','create',*plan['argv'][3:]])
        atomic_json(output/'native-dispatched.json',native)
    wait_exact(native['container_id'],config['image'],output,'waiting_native_actor',time.time()+600)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);check_devices(config['image'])
        suite_plan=freeze_suite(config);suite=Path(config['suite_root'])
        if suite_plan['image']!=config['image'] or suite_plan['physical_devices']!=[4]:raise ValueError('suite plan differs')
        receipt=launch(suite,suite_plan,suite_plan['create_argv'])
        atomic_json(output/'acceptance-dispatched.json',receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    try:
        run(read(args.config),args.output)
        atomic_json(args.output/'terminal.json',dict(exit_code=0,finished_unix=time.time(),scope='Independent acceptance dispatched, not yet accepted.'))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
