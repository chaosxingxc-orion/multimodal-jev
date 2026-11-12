#!/usr/bin/env python3
"""Finish the fixed beta1=0 learning curve after the queued batch32 control."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.intent_actor_inputs import schedule
from multimodal_jev.slurp_intent_pilot import load_pilot
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read, completed
from queue_intent_finite_npu import verify_final
from run_intent_bounded_continuation import validate as validate_chain
from run_intent_batch32_control import validate as validate_batch
from prepare_intent_lr_control import audit_momentum_control
from queue_intent_continuation import run as continue_step
from compare_intent_learning_update import run as compare


def validate(config):
    validate_chain(config)
    protocol = read(config['protocol'])
    if (config['pins'].get(config['protocol']) != file_hash(config['protocol'])
            or protocol['schema'] != 'intent-finite-d12-momentum-control-v1'
            or protocol['adam_betas'] != [0., .999] or protocol['total_updates'] != 12
            or config['initial_updates'] != 1 or config['total_updates'] != 12
            or protocol['checkpoints'] != list(range(13))):
        raise ValueError('fixed full momentum protocol required')
    initial = Path(config['previous_training'])
    cfg = read(initial/'config-template.json')
    reference = Path(config['reference_training'])/'config-template.json'
    if config['pins'].get(str(reference)) != file_hash(reference): raise ValueError('pinned reference required')
    audit_momentum_control(read(reference), cfg)
    if cfg['global_requests'] != 8 or cfg['seed'] != 317 or protocol['lr'] != cfg['lr']:
        raise ValueError('matched original source budget required')
    items, _ = load_pilot(cfg['pilot'], cfg['manifest_sha256'], 'D')
    expected = [[r['id'] for r in schedule(items, 317, i, 8)] for i in range(12)]
    if protocol['schedule'] != expected: raise ValueError('fixed source schedule differs')
    first = dict(training_root=str(initial), suite_root=config['previous_suite'])
    sources = {}
    for step in [first, *config['steps']]:
        for kind in ('training', 'suite'):
            root = Path(step[kind+'_root'])
            for name in ('config-template.json', 'prepared.json', 'source-manifest.json'):
                if config['pins'].get(str(root/name)) != file_hash(root/name):
                    raise ValueError('all original and future plans must be pinned')
            manifest = read(root/'source-manifest.json')
            if kind in sources and manifest != sources[kind]: raise ValueError('frozen continuation source differs')
            sources[kind] = manifest
            check_pins({str(root/'source'/p): h for p,h in manifest.items()})
    if read(Path(config['previous_suite'])/'config-template.json')['training_root'] != str(initial):
        raise ValueError('first acceptance pairing differs')


def authenticate_pair(step, image, updates):
    training, suite = Path(step['training_root']), Path(step['suite_root'])
    for root in (training, suite):
        receipt = read(root/'launch.json')
        if receipt['image'] != image or not completed(receipt['container_id'], image):
            raise ValueError('actual completed training and acceptance required')
        check_pins({str(root/'config.json'): receipt['config_sha256'],
                    str(root/'source-manifest.json'): receipt['source_manifest_sha256']})
    accepted = verify_final(suite)
    verified = read(suite/'verified.json')
    if verified['updates'] != updates or verified['root'] != str(training) or verified['arm'] != 'D':
        raise ValueError('accepted update identity differs')
    return accepted


def await_predecessor(config, output):
    root = Path(config['predecessor_controller'])
    for name in ('launch.json', 'config.json'):
        if config['pins'].get(str(root/name)) != file_hash(root/name): raise ValueError('exact predecessor required')
    receipt = read(root/'launch.json')
    while True:
        check_pins(config['pins']); terminal = root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code'] != 0: raise ValueError('batch32 predecessor failed')
            predecessor = read(root/'config.json')
            check_pins(predecessor['pins']); validate_batch(predecessor)
            if predecessor['image'] != config['image']: raise ValueError('predecessor image differs')
            progress = read(root/'run/progress.json')
            if (progress['completed_updates'] != 3 or progress['total_updates'] != 3
                    or progress['accepted_suite'] != predecessor['steps'][-1]['suite_root']):
                raise ValueError('complete batch32 budget required')
            receipts = [authenticate_pair(step, config['image'], i+1)
                        for i,step in enumerate(predecessor['steps'])]
            first = authenticate_pair(dict(training_root=config['previous_training'],
                suite_root=config['previous_suite']), config['image'], 1)
            atomic_json(output/'predecessor-accepted.json', dict(batch32=receipts, momentum_first=first))
            return
        try: actual = (Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists(): continue
            raise ValueError('predecessor missing without terminal')
        if actual != receipt['argv']: raise ValueError('predecessor PID reused')
        atomic_json(output/'progress.json', dict(stage='waiting_batch32_full_control', pid=receipt['pid'], updated_unix=time.time()))
        time.sleep(600)


def run(config, output):
    check_pins(config['pins']); validate(config); await_predecessor(config, output)
    previous = Path(config['previous_suite'])
    for update, step in enumerate(config['steps'], 2):
        stage = output/f'update-{update:04d}'; stage.mkdir(exist_ok=False)
        launch = read(previous/'launch.json')
        c = dict(step, previous_suite=str(previous), previous_container=launch['container_id'], first_poll_after=0.,
            image=config['image'], lock_directory=config['lock_directory'], pins=dict(config['pins']))
        c['pins'][str(previous/'launch.json')] = file_hash(previous/'launch.json')
        atomic_json(stage/'config.json', c); continue_step(c, stage)
        atomic_json(stage/'learning-comparison.json', compare(step['training_root'], step['suite_root']))
        atomic_json(output/'progress.json', dict(completed_updates=update, total_updates=12,
            accepted_suite=step['suite_root'], updated_unix=time.time()))
        previous = Path(step['suite_root'])


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
