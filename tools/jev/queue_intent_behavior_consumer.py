#!/usr/bin/env python3
"""Consume actual resident behavior on CPU after its independent host gate.

No accelerator mapping, optimizer update, retry or dependency on target scores.
"""
import argparse
from pathlib import Path
import re
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_native_actor_npu import read,wait_exact
from queue_span_dp4_resume import check_pins,remote
from wait_intent_priority_tail import wait_tail
from bind_intent_resident_behavior import bind


def cpu_argv(root,image,external):
    root=Path(root)
    argv=['docker','create','--pull=never','--network=none','--name','multimodal-jev-'+root.name,
        '--cpus=4','--memory=8g','--shm-size=1g','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
        '-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2',
        '-e','PYTHONPATH='+str(root/'source/src')+':'+str(root/'source/tools/jev'),
        '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
        '-v','/mnt/cephfs/xingchao:/mnt/cephfs/xingchao:ro',
        '-v',str(root)+':'+str(root)]
    for entry in external:argv+=['-v',entry['copy']+':'+entry['original']+':ro']
    return argv+[image,'python3','-B',str(root/'source/tools/jev/probe_intent_behavior_consumption_cpu.py'),
        '--config',str(root/'consumer.json'),'--output',str(root/'consumer-run')]


def run(config,out):
    check_pins(config['pins']);root=Path(config['root'])
    if config['priority_kind']!='resident_policy_suite':raise ValueError('actual resident suite dependency required')
    if any((root/n).exists() for n in ('consumer.json','consumer-run','create-attempt.json','container-launch.json')):
        raise ValueError('consumer already claimed; inspect original attempt')
    for p in (config['reference'],str(root/'source-manifest.json')):
        if config['pins'].get(p)!=file_hash(p):raise ValueError('pinned source/reference required')
    check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()})
    for entry in config['external']:
        if file_hash(entry['copy'])!=entry['sha256'] or config['pins'].get(entry['copy'])!=entry['sha256']:
            raise ValueError('fixed external asset required')
    wait_tail(config,out)
    audit=Path(config['priority_predecessor'])/'run/accepted.json'
    bound,proof=bind(config['reference'],audit,'initial')
    # Every host-only evidence dependency needs a fixed read-only container copy.
    external={e['original']:e['sha256'] for e in config['external']}
    if {p:h for p,h in bound['pins'].items() if not p.startswith('/mnt/cephfs/xingchao/')}!=external:
        raise ValueError('unmounted external consumer evidence')
    candidate=root/'bound-input.json';atomic_json(candidate,bound);atomic_json(root/'binding.json',proof)
    atomic_json(root/'consumer.json',dict(reference=config['reference'],candidate=str(candidate),mode='resident',
        pins={config['reference']:file_hash(config['reference']),str(candidate):file_hash(candidate)}))
    argv=cpu_argv(root,config['image'],config['external'])
    atomic_json(root/'create-attempt.json',dict(argv=argv,created_unix=time.time()))
    result=remote(argv);atomic_json(root/'create.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    if result.returncode or not re.fullmatch('[a-f0-9]{64}',result.stdout.strip()):raise ValueError('ambiguous CPU creation; no retry')
    cid=result.stdout.strip();atomic_json(root/'container-launch.json',dict(container_id=cid,image=config['image'],physical_devices=[],argv=argv))
    result=remote(['docker','start',cid]);atomic_json(root/'start.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr));result.check_returncode()
    wait_exact(cid,config['image'],out,'waiting_actual_cpu_consumer',time.time()+600)
    report=read(root/'consumer-run/result.json')
    if (read(root/'consumer-run/terminal.json')['exit_code']!=0 or report.get('passed') is not True
            or report['mode']!='resident' or report['recordings']!=89 or report['source_uses']!=96
            or len(report['batches'])!=12 or report['policy_id']!=proof['policy_id']):
        raise ValueError('complete same-policy resident consumption required')
    atomic_json(out/'accepted.json',dict(passed=True,container_id=cid,result_sha256=file_hash(root/'consumer-run/result.json'),
        binding_sha256=file_hash(root/'binding.json'),scope='Real resident behavior consumed by CPU actor input compiler; no optimizer update.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
