#!/usr/bin/env python3
"""Run the frozen multi-policy suite after all registered utility curves."""
import argparse
import inspect
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read,locks,launch,wait_exact
from queue_intent_prepared_acceptance import wait_resources
from verify_span_resident_policy_suite import verify


def validate(config):
    check_pins(config['pins']);root=Path(config['suite_root'])
    plan=read(root/'prepared.json');cfg=read(root/'config.json')
    for name in ('config.json','prepared.json','source-manifest.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('pinned resident plan required')
    if (plan['root']!=str(root) or plan['image']!=config['image'] or plan['physical_devices']!=[4]
            or plan['config_sha256']!=file_hash(root/'config.json')
            or plan['source_manifest_sha256']!=file_hash(root/'source-manifest.json')
            or cfg['schema']!='jev-resident-policy-suite-v1'
            or [p['key'] for p in cfg['policies']]!=['initial','d12']):
        raise ValueError('fixed two-policy TP1 suite required')
    check_pins(cfg['pins']);check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()})
    if any((root/n).exists() for n in ('create-attempt.json','launch.json','run')):
        raise ValueError('resident suite already claimed; inspect original attempt')
    if config['priority_kind']!='utility_curves12':raise ValueError('registered utility curves must precede suite')
    return root,plan


def authenticate_complete(config,root):
    """Downstream verifies actual inference and complete saved host evidence."""
    root=Path(root);prior=read(root/'config.json');check_pins(prior['pins'])
    if prior['image']!=config['image'] or read(root/'run/terminal.json')['exit_code']!=0:
        raise ValueError('successful fixed-image resident controller required')
    from wait_intent_priority_tail import authenticate
    if (prior['priority_kind']!='utility_curves12'
            or read(root/'run/priority-accepted.json')!=authenticate(prior,Path(prior['priority_predecessor']))):
        raise ValueError('complete utility predecessor acceptance required')
    suite=Path(prior['suite_root'])
    if read(root/'run/dispatched.json')!=read(suite/'launch.json'):
        raise ValueError('resident dispatch identity differs')
    audit=verify(suite,prior['pilot'],prior['manifest_sha256'],prior['image'])
    saved=read(root/'run/accepted.json')
    # The downstream controller has an identical frozen verifier at a different
    # path. Authenticate both implementations, then compare workload evidence.
    old_verifier=root/'source/tools/jev/verify_span_resident_policy_suite.py'
    new_verifier=Path(inspect.getfile(verify))
    if (prior['pins'].get(str(old_verifier))!=file_hash(old_verifier)
            or file_hash(old_verifier)!=file_hash(new_verifier)
            or saved['files'].get(str(old_verifier))!=file_hash(old_verifier)
            or audit['files'].get(str(new_verifier))!=file_hash(new_verifier)):
        raise ValueError('identical pinned host verifier required')
    normalized=dict(saved,files=dict(saved['files']));normalized['files'].pop(str(old_verifier))
    normalized['files'][str(new_verifier)]=file_hash(new_verifier)
    check_pins(saved['files'])
    if normalized!=audit:raise ValueError('resident acceptance differs')
    return dict(root=str(root),suite=str(suite),accepted_sha256=file_hash(root/'run/accepted.json'),
        policies=audit['policies'],scope='Resident inference accepted; no optimizer update or efficacy claim.')


def run(config,out):
    from wait_intent_priority_tail import wait_tail
    root,plan=validate(config);wait_tail(config,out)
    wait_resources(config,out)
    with locks(Path(config['lock_directory'])):
        validate(config);receipt=launch(root,plan,plan['create_argv'])
        atomic_json(out/'dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],out,'waiting_resident_policy_suite',time.time()+600)
    atomic_json(out/'accepted.json',verify(root,config['pilot'],config['manifest_sha256'],config['image']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
