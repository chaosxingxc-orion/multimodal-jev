#!/usr/bin/env python3
"""Run the fixed lower-LR pilot after the exact Q/P controller completes."""
import argparse
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.intent_resume import validate_lineage
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read, completed
from queue_intent_finite_npu import verify_final
from run_intent_matched_pilots import first_step
from queue_intent_continuation import run as continue_step
from compare_intent_learning_update import run as compare


def validate(config):
    if len(config['steps']) != 12: raise ValueError('fixed twelve-update LR control required')
    protocol = read(config['protocol'])
    if (config['pins'].get(config['protocol']) != file_hash(config['protocol'])
            or protocol['lr'] != 2e-5 or protocol['seed'] != 317 or protocol['total_updates'] != 12):
        raise ValueError('registered lower-LR protocol required')
    prior = None; previous = None; roots = set()
    for index, step in enumerate(config['steps']):
        train, suite = Path(step['training_root']), Path(step['suite_root'])
        for root in (train, suite):
            if str(root) in roots: raise ValueError('distinct immutable roots required')
            roots.add(str(root))
            for name in ('prepared.json', 'config-template.json', 'source-manifest.json'):
                if config['pins'].get(str(root/name)) != file_hash(root/name): raise ValueError('all plans must be pinned')
        cfg = read(train/'config-template.json')
        if (cfg['lr'] != 2e-5 or cfg['seed'] != 317 or cfg['arm'] != 'D' or cfg['step'] != index
                or cfg['global_requests'] != 8 or read(suite/'config-template.json')['training_root'] != str(train)):
            raise ValueError('fixed matched LR control configuration required')
        if prior is None:
            if (cfg['initial_adapter'] != config['initial_adapter'] or cfg['acceptance_root'] != config['initial_gate']
                    or cfg.get('previous_root') or cfg.get('initial_optimizer_checkpoint')):
                raise ValueError('original initialization and fresh optimizer required')
        else:
            validate_lineage(prior, cfg)
            if cfg['previous_root'] != previous['training_root'] or cfg['previous_suite'] != previous['suite_root']:
                raise ValueError('contiguous lower-LR lineage required')
        prior, previous = cfg, step


def await_qp(config, output):
    root = Path(config['predecessor_controller']); receipt = read(root/'launch.json')
    for name in ('launch.json', 'config.json'):
        if config['pins'].get(str(root/name)) != file_hash(root/name): raise ValueError('exact Q/P controller must be pinned')
    while True:
        check_pins(config['pins']); terminal = root/'run/terminal.json'
        if terminal.exists():
            if read(terminal)['exit_code'] != 0: raise ValueError('Q/P predecessor failed')
            progress = read(root/'run/progress.json')
            if (progress['arm'] != 'P' or progress['completed_updates'] != 12 or progress['total_updates'] != 12
                    or progress['accepted_suite'] != config['predecessor_suite']):
                raise ValueError('complete Q/P budget required')
            suite = Path(config['predecessor_suite']); launch = read(suite/'launch.json')
            if launch['image'] != config['image'] or not completed(launch['container_id'], config['image']):
                raise ValueError('actual completed P12 gate required')
            accepted = verify_final(suite); verified = read(suite/'verified.json')
            if verified['arm'] != 'P' or verified['updates'] != 12: raise ValueError('accepted P12 required')
            atomic_json(output/'predecessor-accepted.json', accepted); return
        try: actual = (Path('/proc')/str(receipt['pid'])/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        except FileNotFoundError:
            if terminal.exists(): continue
            raise ValueError('Q/P controller missing without terminal')
        if actual != receipt['argv']: raise ValueError('Q/P controller PID reused')
        atomic_json(output/'progress.json', dict(stage='waiting_exact_QP12', pid=receipt['pid'], updated_unix=time.time()))
        time.sleep(600)


def run(config, output):
    check_pins(config['pins']); validate(config); await_qp(config, output)
    for index, step in enumerate(config['steps']):
        stage = output/f'update-{index+1:04d}'; stage.mkdir(exist_ok=False)
        if index == 0: first_step(config, step, stage)
        else:
            previous = Path(config['steps'][index-1]['suite_root']); receipt = read(previous/'launch.json')
            c = dict(step, previous_suite=str(previous), previous_container=receipt['container_id'], first_poll_after=0.,
                     image=config['image'], lock_directory=config['lock_directory'], pins=dict(config['pins']))
            c['pins'][str(previous/'launch.json')] = file_hash(previous/'launch.json')
            atomic_json(stage/'config.json', c); continue_step(c, stage)
        atomic_json(stage/'learning-comparison.json', compare(step['training_root'], step['suite_root']))
        atomic_json(output/'progress.json', dict(completed_updates=index+1, total_updates=12,
            accepted_suite=step['suite_root'], updated_unix=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(exist_ok=False)
    try:
        run(read(a.config), a.output); atomic_json(a.output/'terminal.json', dict(exit_code=0, finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json', dict(exit_code=1, error=repr(exc), finished_unix=time.time())); raise
