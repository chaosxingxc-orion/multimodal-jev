#!/usr/bin/env python3
"""One real-model residency replay after existing binding queue, no retries."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_native_actor_npu import read,completed,locks,launch,wait_exact
from queue_span_dp4_resume import check_pins
from queue_intent_prepared_acceptance import wait_resources
from queue_intent_finite_npu import freeze_training,verify_final
from freeze_intent_finite_acceptance import freeze as freeze_suite
from queue_intent_binding_first_step import verify_final as verify_binding
from queue_span_residency import verify_results
from probe_intent_residency_replay_npu import validate as validate_replay


def validate(config):
    check_pins(config['pins']);root=Path(config['plan_root'])
    for kind in ('training','suite'):
        out=root/kind
        for name in ('prepared.json','config-template.json','source-manifest.json'):
            if config['pins'].get(str(out/name))!=file_hash(out/name):raise ValueError('complete pinned replay plan required')
        plan=read(out/'prepared.json');cfg=read(out/'config-template.json')
        if plan['root']!=str(out) or plan['image']!=config['image'] or plan['physical_devices']!=([4,5,6,7] if kind=='training' else [4]):
            raise ValueError('replay plan topology differs')
        check_pins(cfg['pins']);check_pins({str(out/'source'/p):h for p,h in read(out/'source-manifest.json').items()})
    validate_replay(read(root/'training/config-template.json'))
    if read(root/'suite/config-template.json')['training_root']!=str(root/'training'):
        raise ValueError('replay suite pairing differs')


def await_predecessor(config,output):
    root=Path(config['predecessor_controller'])
    for name in ('config.json','launch.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('exact binding predecessor required')
    receipt=read(root/'launch.json')
    while True:
        check_pins(config['pins']);terminal=root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('binding predecessor failed')
            prior=read(root/'config.json');plan=Path(prior['plan_root'])
            accepted=verify_binding(plan/'suite',plan/'training',config['image'])
            probe=Path(config['residency_probe']);launch_record=read(probe/'launch.json')
            if launch_record['image']!=config['image'] or not completed(launch_record['container_id'],config['image']):
                raise ValueError('actual synthetic residency completion required')
            if config['pins'].get(str(probe/'prepared.json'))!=file_hash(probe/'prepared.json'):
                raise ValueError('pinned residency probe required')
            prepared=read(probe/'prepared.json')
            if prepared['root']!=str(probe) or prepared['image']!=config['image'] or '--residency-roundtrip' not in prepared['create_argv']:
                raise ValueError('wrong synthetic residency probe')
            for name,key in (('config.json','config_sha256'),('source-manifest.json','source_manifest_sha256')):
                if key in prepared and file_hash(probe/name)!=prepared[key]:raise ValueError('synthetic probe changed')
            check_pins({str(probe/'source'/p):h for p,h in read(probe/'source-manifest.json').items()})
            check_pins(read(probe/'config.json')['pins'])
            synthetic=verify_results(probe)
            atomic_json(output/'predecessor-accepted.json',dict(binding=accepted,residency=synthetic));return
        try:actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('predecessor missing without terminal')
        if actual!=receipt['argv']:raise ValueError('predecessor PID reused')
        atomic_json(output/'progress.json',dict(stage='waiting_binding_first_acceptance',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def accepted(training,suite,image):
    training,suite=map(Path,(training,suite))
    for root in (training,suite):
        receipt=read(root/'launch.json')
        if receipt['image']!=image or not completed(receipt['container_id'],image):raise ValueError('actual replay phases must exit zero')
    ordinary=verify_final(suite)
    if read(suite/'residency-terminal.json')['exit_code']!=0:raise ValueError('replay comparison failed')
    receipt=read(suite/'RESIDENCY_ACCEPTED.json');report=read(suite/'residency-verified.json')
    if (receipt['passed'] is not True or receipt['verified_sha256']!=file_hash(suite/'residency-verified.json')
            or report['passed'] is not True or report['replay']!=str(training)
            or [r['rank'] for r in report['ranks']]!=list(range(4))
            or any(r['updates']!=2 or r['exact_master_and_adam'] is not True for r in report['ranks'])):
        raise ValueError('complete exact replay evidence required')
    check_pins(report['files'])
    return dict(passed=True,ordinary=ordinary,residency=receipt)


def run(config,output):
    validate(config);root=Path(config['plan_root']);training=root/'training';suite=root/'suite'
    if any((p/n).exists() for p in (training,suite) for n in ('config.json','launch.json','create-attempt.json','run')):
        raise ValueError('replay already claimed; inspect existing task')
    await_predecessor(config,output);wait_resources(config,output)
    with locks(Path(config['lock_directory'])):
        validate(config);plan=read(training/'prepared.json');freeze_training(plan)
        receipt=launch(training,plan,plan['create_argv']);atomic_json(output/'training-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_replay_training',time.time()+600)
    wait_resources(config,output)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);freeze_suite(suite);plan=read(suite/'prepared.json')
        receipt=launch(suite,plan,plan['create_argv']);atomic_json(output/'acceptance-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_replay_acceptance',time.time()+600)
    atomic_json(output/'accepted.json',accepted(training,suite,config['image']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
