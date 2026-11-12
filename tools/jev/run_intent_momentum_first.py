#!/usr/bin/env python3
"""Dispatch one reviewed momentum-control step after exact binding diagnostics."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read
from prepare_intent_lr_control import audit_momentum_control
from run_intent_matched_pilots import first_step
from score_intent_binding_matrix import run as score_binding
from compare_intent_learning_update import run as compare


def validate(config):
    root = Path(config['control_root'])
    for path in (root/'protocol.json', root/'source-audit.json', Path(config['reference_training'])/'config-template.json'):
        if config['pins'].get(str(path)) != file_hash(path): raise ValueError('control evidence must be pinned')
    protocol = read(root/'protocol.json')
    if (protocol['schema'] != 'intent-finite-d12-momentum-control-v1'
            or protocol['adam_betas'] != [0., .999] or protocol['lr'] != 1e-4
            or protocol['total_updates'] != 12 or protocol['checkpoints'] != list(range(13))):
        raise ValueError('registered momentum protocol required')
    for name in ('training','suite'):
        target = root/name
        for filename in ('prepared.json','config-template.json','source-manifest.json'):
            path = target/filename
            if config['pins'].get(str(path)) != file_hash(path): raise ValueError('exact prepared pair required')
        manifest = read(target/'source-manifest.json')
        check_pins({str(target/'source'/p):h for p,h in manifest.items()})
        if any((target/p).exists() for p in ('config.json','launch.json','create-attempt.json','run')):
            raise ValueError('already claimed; no retries')
    training = read(root/'training/config-template.json')
    audit_momentum_control(read(Path(config['reference_training'])/'config-template.json'), training)
    if (training['initial_adapter'] != config['initial_adapter'] or training['acceptance_root'] != config['initial_gate']
            or read(root/'suite/config-template.json')['training_root'] != str(root/'training')):
        raise ValueError('original policy and exact acceptance pairing required')


def await_binding(config, output):
    root = Path(config['predecessor_controller'])
    for name in ('launch.json','config.json'):
        if config['pins'].get(str(root/name)) != file_hash(root/name): raise ValueError('exact predecessor required')
    receipt = read(root/'launch.json')
    while True:
        check_pins(config['pins']); terminal = root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code'] != 0: raise ValueError('binding predecessor failed')
            predecessor = read(root/'config.json')
            if predecessor['image'] != config['image']: raise ValueError('predecessor image differs')
            # Reauthenticate actual Docker exits and every backend record. No
            # decision based on whether the binding diagnostic looks favorable.
            result = score_binding(predecessor['protocol'], predecessor['protocol_sha256'])
            if result != read(root/'run/comparison.json'): raise ValueError('binding comparison differs')
            atomic_json(output/'predecessor-accepted.json', dict(protocol_sha256=result['protocol_sha256'],
                comparison_sha256=file_hash(root/'run/comparison.json')))
            return
        try: actual = (Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists(): continue
            raise ValueError('predecessor missing without terminal')
        if actual != receipt['argv']: raise ValueError('predecessor PID reused')
        atomic_json(output/'progress.json', dict(stage='waiting_exact_binding_matrix', pid=receipt['pid'], updated_unix=time.time()))
        time.sleep(600)


def run(config, output):
    check_pins(config['pins']); validate(config); await_binding(config, output)
    validate(config)
    root = Path(config['control_root']); step = dict(training_root=str(root/'training'), suite_root=str(root/'suite'))
    stage = output/'update-0001'; stage.mkdir(exist_ok=False)
    first_step(config, step, stage)
    atomic_json(output/'learning-comparison.json', compare(step['training_root'], step['suite_root']))
    atomic_json(output/'progress.json', dict(completed_updates=1, accepted_suite=step['suite_root'],
        planned_protocol_updates=12, scope='First step accepted; remaining steps require separate preparation.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
