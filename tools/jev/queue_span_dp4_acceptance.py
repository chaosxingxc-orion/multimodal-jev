#!/usr/bin/env python3
"""Wait exact DP4 resume, freeze exported assets, dispatch independent gate.

The existing drain queue owns training dispatch. This queue never restarts it.
Successful terminal means gate dispatched; ACCEPTED is written by its auditor.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import re
import time

from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_dp4_resume import remote,inspect,check_pins,check_devices,CARDS


def predecessor(config):
    queue=Path(config['resume_queue']);root=Path(config['resume_root'])
    check_pins(config['pins'])
    terminal=queue/'run/terminal.json'
    if not terminal.exists():
        launch=json.loads((queue/'launch.json').read_text())
        try:argv=(Path('/proc')/str(launch['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():return None
            raise ValueError('resume queue absent without terminal')
        if argv!=launch['argv']:raise ValueError('resume queue PID replaced')
        return None
    if json.loads(terminal.read_text())['exit_code']!=0:raise ValueError('DP4 dispatch queue failed')
    dispatched=json.loads((queue/'run/dispatched.json').read_text())
    if dispatched['root']!=str(root):raise ValueError('unexpected resume root')
    launch=json.loads((root/'launch.json').read_text());cid=dispatched['container_id']
    if launch['container_id']!=cid or launch['launch_exit_code']!=0:raise ValueError('resume identity mismatch')
    state=inspect([cid])[cid]
    if state['Image']!=config['image']:raise ValueError('resume image mismatch')
    if state['State']['Status']=='running':return None
    if state['State']['Status']!='exited' or state['State']['ExitCode']!=0:raise ValueError('resume failed or unexpected state')
    return cid


def freeze_inputs(config):
    from multimodal_jev.policy_identity import checkpoint_identity
    root=Path(config['resume_root']);suite=Path(config['suite_root'])
    if any((suite/name).exists() for name in ('config.json','launch.json','run','create-attempt.json')):
        raise ValueError('acceptance already claimed; inspect existing artifacts')
    prepared=json.loads((suite/'prepared.json').read_text())
    manifest=json.loads((suite/'source-manifest.json').read_text())
    check_pins({str(suite/'source'/p):h for p,h in manifest.items()})
    parent=json.loads((root/'config.json').read_text());launch=json.loads((root/'launch.json').read_text())
    if (file_hash(root/'config.json')!=launch['config_sha256']
            or file_hash(root/'source-manifest.json')!=launch['source_manifest_sha256']):
        raise ValueError('resume provenance changed')
    check_pins(parent['pins'])
    adapter_id,files=checkpoint_identity(root/'run/adapter')
    pins={str(suite/'source'/p):h for p,h in manifest.items()}
    pins.update({str(root/'run/adapter'/p):h for p,h in files.items()})
    for rank in range(4):
        path=root/'run'/f'rank-{rank}'
        terminal=json.loads((path/'terminal.json').read_text());result=json.loads((path/'result.json').read_text())
        if (terminal['exit_code']!=0 or result['status']!='passed' or result['optimizer_updates']!=2
                or result['rank']!=rank or result['world_size']!=4 or result['adapter_id']!=adapter_id):
            raise ValueError('four successful step-two shards required')
        for name in ('terminal.json','result.json','optimizer-local.pt','restored-local.pt','optimizer-resume.json','selection.json','after.json'):
            pins[str(path/name)]=file_hash(path/name)
    for path in (root/'launch.json',root/'config.json',root/'source-manifest.json',Path(prepared['identity'])):
        pins[str(path)]=file_hash(path)
    result=dict(model=parent['model_path'],adapter=str(root/'run/adapter'),identity=prepared['identity'],
        runtime_profile=parent['runtime_profile'],resume_root=str(root),pins=pins)
    atomic_json(suite/'config.json',result)
    return result


def dispatch(config,output):
    suite=Path(config['suite_root'])
    with ExitStack() as stack:
        for card in CARDS:
            lock=stack.enter_context((Path(config['lock_directory'])/f'.physical-card{card}-launch.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cid=predecessor(config)
        if cid is None:raise ValueError('resume no longer ready')
        check_devices(config['image'])
        freeze_inputs(config)
        plan=json.loads((suite/'prepared.json').read_text());argv=plan['create_argv']
        if argv[:2]!=['docker','create'] or plan['image']!=config['image'] or plan['physical_devices']!=[4]:
            raise ValueError('fixed TP1 acceptance plan required')
        atomic_json(suite/'create-attempt.json',dict(argv=argv,parent_container=cid,time=time.time()))
        result=remote(argv)
        atomic_json(suite/'create.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode!=0 or not re.fullmatch('[0-9a-f]{64}',result.stdout.strip()):
            raise ValueError('create failed or ambiguous; never auto-recreate')
        child=result.stdout.strip()
        atomic_json(suite/'launch.json',dict(root=str(suite),container_id=child,image=config['image'],
            argv=argv,physical_devices=[4],launch_exit_code=0,parent_container=cid,
            config_sha256=file_hash(suite/'config.json'),source_manifest_sha256=file_hash(suite/'source-manifest.json')))
        check_devices(config['image'])
        result=remote(['docker','start',child]);atomic_json(suite/'start.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        result.check_returncode()
        state=inspect([child])[child]
        if state['Image']!=config['image'] or state['State']['Status'] not in ('running','exited'):
            raise ValueError('acceptance identity/state differs')
        if state['State']['Status']=='exited' and state['State']['ExitCode']!=0:raise ValueError('acceptance failed at start')
        atomic_json(output/'dispatched.json',dict(root=str(suite),container_id=child,
            scope='Independent gate dispatched; wait controller-terminal and ACCEPTED, not a training or parity success.'))


def run(config,output):
    while True:
        cid=predecessor(config)
        atomic_json(output/'progress.json',dict(stage='waiting_exact_resume' if cid is None else 'dispatching_acceptance',
            resume_container_id=cid,updated_unix=time.time()))
        if cid is not None:
            dispatch(config,output);return
        time.sleep(600)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(json.loads(a.config.read_text()),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
