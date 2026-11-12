#!/usr/bin/env python3
"""One accepted intent update -> exact state continuation -> independent audit.

Follows the existing controller, never duplicates a pending acceptance. All
training/suite launches are single attempts; unknown outcomes require inspection.
"""
import argparse
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read,locks,launch,wait_exact,completed
from queue_intent_finite_npu import verify_final
from freeze_intent_continuation import freeze as freeze_training
from freeze_intent_finite_acceptance import freeze as freeze_suite


def await_predecessor(config,output):
    if config.get('previous_container'):
        suite=Path(config['previous_suite']);receipt=read(suite/'launch.json')
        if (config['pins'].get(str(suite/'launch.json'))!=file_hash(suite/'launch.json')
                or receipt['container_id']!=config['previous_container']
                or receipt['root']!=str(suite) or receipt['image']!=config['image']
                or receipt['physical_devices']!=[4]):
            raise ValueError('exact independently dispatched suite receipt required')
        wait_exact(receipt['container_id'],config['image'],output,'waiting_exact_predecessor_acceptance',config['first_poll_after'])
        check_pins(config['pins'])
        atomic_json(output/'predecessor-accepted.json',verify_final(suite))
        return
    queue=Path(config['previous_queue']);suite=Path(config['previous_suite'])
    receipt=read(queue/'launch.json')
    if config['pins'].get(str(queue/'launch.json'))!=file_hash(queue/'launch.json'):
        raise ValueError('exact predecessor queue receipt must be pinned')
    while True:
        check_pins(config['pins'])
        terminal=queue/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('predecessor queue failed')
            accepted=verify_final(suite)
            if read(queue/'run/accepted.json')!=accepted:raise ValueError('predecessor acceptance differs')
            launched=read(suite/'launch.json')
            if (read(queue/'run/acceptance-dispatched.json')!=launched
                    or launched['root']!=str(suite) or launched['image']!=config['image']
                    or launched['physical_devices']!=[4]):
                raise ValueError('predecessor dispatch identity differs')
            if not completed(launched['container_id'],config['image']):
                raise ValueError('predecessor controller finished before its container')
            atomic_json(output/'predecessor-accepted.json',accepted)
            return
        try:
            actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('predecessor controller missing without terminal')
        if actual!=receipt['argv']:raise ValueError('predecessor controller PID reused')
        atomic_json(output/'progress.json',dict(stage='waiting_existing_acceptance_controller',
            pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def run(config,output):
    check_pins(config['pins'])
    training=Path(config['training_root']);suite=Path(config['suite_root'])
    plan=read(training/'prepared.json');suite_plan=read(suite/'prepared.json')
    parent=read(training/'config-template.json');acceptance=read(suite/'config-template.json')
    if (plan['root']!=str(training) or suite_plan['root']!=str(suite)
            or plan['image']!=config['image'] or suite_plan['image']!=config['image']
            or plan['physical_devices']!=[4,5,6,7] or suite_plan['physical_devices']!=[4]
            or parent['previous_suite']!=config['previous_suite']
            or acceptance['training_root']!=str(training)):
        raise ValueError('continuation topology or lineage differs')
    await_predecessor(config,output)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);freeze_training(training)
        receipt=launch(training,plan,plan['create_argv'])
        atomic_json(output/'training-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_continuation',time.time()+600)
    with locks(Path(config['lock_directory'])):
        check_pins(config['pins']);freeze_suite(suite)
        receipt=launch(suite,suite_plan,suite_plan['create_argv'])
        atomic_json(output/'acceptance-dispatched.json',receipt)
    wait_exact(receipt['container_id'],config['image'],output,'waiting_continuation_acceptance',time.time()+600)
    atomic_json(output/'accepted.json',verify_final(suite))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time(),
            scope='One exact intent RL continuation and independent acceptance. No held-out or transfer efficacy claim.'))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
