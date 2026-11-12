#!/usr/bin/env python3
"""Run a full fixed FSC matrix after the existing training queue drains.

Every create is attempted once. Failures preserve artifacts, and scores are
computed only after all seven members have independently passed validation.
"""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read, completed, locks, launch, wait_exact
from queue_intent_finite_npu import verify_final
from queue_intent_prepared_acceptance import wait_resources
from prepare_fsc_transfer_matrix import prepare
from score_fsc_span_transfer import authenticate_gate, run as score


def await_training(config, output):
    root=Path(config['predecessor_controller'])
    for name in ('launch.json','config.json'):
        if config['pins'].get(str(root/name))!=file_hash(root/name):raise ValueError('exact training controller must be pinned')
    receipt=read(root/'launch.json')
    while True:
        check_pins(config['pins']);terminal=root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code']!=0:raise ValueError('training predecessor failed')
            progress=read(root/'run/progress.json')
            if (progress['completed_updates']!=12 or progress['total_updates']!=12
                    or progress['accepted_suite']!=config['predecessor_suite']):
                raise ValueError('complete training budget required')
            suite=Path(config['predecessor_suite']);record=read(suite/'launch.json')
            if record['image']!=config['image'] or not completed(record['container_id'],config['image']):
                raise ValueError('actual completed predecessor gate required')
            accepted=verify_final(suite);verified=read(suite/'verified.json')
            if verified['updates']!=12 or verified['arm']!='D':raise ValueError('final lower-LR D12 acceptance required')
            atomic_json(output/'predecessor-accepted.json',accepted);return
        try:actual=(Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists():continue
            raise ValueError('training controller missing without terminal')
        if actual!=receipt['argv']:raise ValueError('training controller PID reused')
        atomic_json(output/'progress.json',dict(stage='waiting_training_queue',pid=receipt['pid'],updated_unix=time.time()))
        time.sleep(600)


def run_matrix(config, output):
    root=Path(config['matrix_root']);path=root/'protocol.json';digest=file_hash(path);protocol=read(path)
    expected=['step0']+[f'{a}-step{s}' for a in ('D','Q','P') for s in (6,12)]
    if [e['key'] for e in protocol['checkpoints']]!=expected or protocol['image']!=config['image']:
        raise ValueError('complete fixed seven-policy matrix required')
    task=read(protocol['task']);population=read(protocol['population'])
    pins=dict(config['pins']);pins[str(path)]=digest
    for entry in protocol['checkpoints']:
        gate=Path(entry['root'])
        if any((gate/n).exists() for n in ('launch.json','create-attempt.json','run')):
            raise ValueError('matrix member already claimed; never recreate')
        for name in ('config.json','prepared.json','source-manifest.json'):pins[str(gate/name)]=file_hash(gate/name)
    atomic_json(output/'matrix-pins.json',pins)
    for entry in protocol['checkpoints']:
        gate=Path(entry['root']);plan=read(gate/'prepared.json');cfg=read(gate/'config.json')
        if (plan['image']!=config['image'] or plan['physical_devices']!=[4] or plan['root']!=str(gate)
                or plan['config_sha256']!=file_hash(gate/'config.json')
                or plan['source_manifest_sha256']!=file_hash(gate/'source-manifest.json')
                or cfg['pins'].get(str(path))!=digest):raise ValueError('frozen matrix member identity differs')
        stage=output/entry['key'];stage.mkdir(exist_ok=False)
        wait_resources(dict(image=config['image'],pins=pins),stage)
        with locks(Path(config['lock_directory'])):
            check_pins(pins);check_pins(cfg['pins'])
            check_pins({str(gate/'source'/p):h for p,h in read(gate/'source-manifest.json').items()})
            receipt=launch(gate,plan,plan['create_argv']);atomic_json(stage/'dispatched.json',receipt)
        wait_exact(receipt['container_id'],config['image'],stage,'waiting_full_fsc_policy',time.time()+600)
        normalized,evidence=authenticate_gate(entry,path,digest,protocol,population,task)
        atomic_json(stage/'backend-accepted.json',dict(audit=normalized['audit'],files=evidence))
        del normalized
        atomic_json(output/'progress.json',dict(stage='accepted_matrix_member',key=entry['key'],updated_unix=time.time()))
    # Score's entry point authenticates the entire matrix again before labels.
    atomic_json(output/'comparison.json',score(path,digest))


def run(config, output):
    check_pins(config['pins'])
    if config['pins'].get(config['builder_config'])!=file_hash(config['builder_config']):
        raise ValueError('fixed matrix builder required')
    await_training(config,output)
    atomic_json(output/'prepared-matrix.json',prepare(read(config['builder_config']),config['matrix_root']))
    run_matrix(config,output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time(),scope='Completed fixed exploratory transfer evaluation; efficacy depends on reported scores.'))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
