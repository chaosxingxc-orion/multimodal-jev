#!/usr/bin/env python3
"""Bind the prospective seven-policy FSC selection to accepted source artifacts.

No target audio, labels, predictions, model loading or dispatch are consumed.
"""
import argparse
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read, completed
from queue_intent_finite_npu import verify_final, verify_gate


def validate_rule(rule):
    expected = [('step0', 'shared_initial', 0)] + [
        (f'{arm}-step{step}', arm, step) for arm in ('D', 'Q', 'P') for step in (6, 12)]
    if (rule['schema'] != 'fsc-dqp-transfer-prospective-selection-v1'
            or rule['target_feedback_used_for_selection'] is not False
            or [(r['key'], r['arm'], r['update']) for r in rule['checkpoints']] != expected
            or len({r['source_gate'] for r in rule['checkpoints']}) != 7):
        raise ValueError('exact prospective seven-policy selection required')


def controller_complete(root, expected_suite, arm, evidence):
    root = Path(root)
    if read(root/'run/terminal.json')['exit_code'] != 0:
        raise ValueError('source controller failed')
    progress = read(root/'run/progress.json')
    if (progress['completed_updates'] != 12 or progress['total_updates'] != 12
            or progress['accepted_suite'] != expected_suite
            or (arm == 'P' and progress['arm'] != 'P')):
        raise ValueError('all source pilots must complete before selection resolution')
    for name in ('launch.json', 'config.json', 'run/terminal.json', 'run/progress.json'):
        evidence[str(root/name)] = file_hash(root/name)
    launch = read(root/'launch.json')
    check_pins({str(root/'config.json'): launch['config_sha256'],
                str(root/'source-manifest.json'): launch['source_manifest_sha256']})
    check_pins(read(root/'config.json')['pins'])


def resolve(config):
    check_pins(config['pins'])
    rule_path = Path(config['rule'])
    if config['pins'].get(str(rule_path)) != file_hash(rule_path):
        raise ValueError('prospective rule must be pinned')
    rule = read(rule_path); validate_rule(rule); check_pins(rule['pins'])
    for key in ('d_controller', 'qp_controller'):
        for name in ('launch.json', 'config.json'):
            path = str(Path(config[key])/name)
            if config['pins'].get(path) != file_hash(path):
                raise ValueError('exact source controllers must be pinned')
    evidence = dict(config['pins']); evidence.update(rule['pins'])
    sources = rule['checkpoints']
    controller_complete(config['d_controller'], sources[2]['source_gate'], 'D', evidence)
    controller_complete(config['qp_controller'], sources[-1]['source_gate'], 'P', evidence)
    results = []; common = None
    for item in sources:
        gate = Path(item['source_gate']); launch = read(gate/'launch.json'); cfg = read(gate/'config.json')
        if launch['image'] != config['image'] or not completed(launch['container_id'], config['image']):
            raise ValueError('actual completed source gate required')
        verify_gate(gate, config['pilot'], config['manifest_sha256'])
        identity = (cfg['base_id'], cfg['runtime_profile'])
        if identity != (config['base_id'], config['runtime_profile']):
            raise ValueError('registered base and runtime required')
        if common is None: common = identity
        if identity != common: raise ValueError('same base and runtime required')
        if item['update']:
            verify_final(gate); verified = read(gate/'verified.json'); train = Path(item['source_training'])
            training = read(train/'config.json'); tl = read(train/'launch.json')
            if (verified['updates'] != item['update'] or verified['arm'] != item['arm']
                    or verified['root'] != str(train) or verified['suite'] != str(gate)
                    or verified['adapter_id'] != cfg['adapter_id'] or cfg['training_root'] != str(train)
                    or training['step'] + 1 != item['update'] or training['arm'] != item['arm']
                    or training['seed'] != 317 or training['lr'] != 1e-4
                    or cfg['adapter'] != str(train/'run/adapter')
                    or tl['image'] != config['image'] or not completed(tl['container_id'], config['image'])):
                raise ValueError('selected source arm/update/export differs')
            check_pins({str(train/'config.json'): tl['config_sha256']})
            for path in (gate/'verified.json', gate/'ACCEPTED.json', train/'config.json', train/'launch.json'):
                evidence[str(path)] = file_hash(path)
        adapter_id, files = checkpoint_identity(cfg['adapter'])
        if adapter_id != cfg['adapter_id']: raise ValueError('source adapter content changed')
        evidence.update({str(Path(cfg['adapter'])/p): h for p, h in files.items()})
        for name in ('config.json', 'launch.json', 'source-manifest.json', 'run/READY.json'):
            evidence[str(gate/name)] = file_hash(gate/name)
        results.append(dict(item, adapter_id=adapter_id, adapter=cfg['adapter']))
    return dict(schema='fsc-dqp-resolved-source-selection-v1',
        checkpoints=[dict(key=r['key'], adapter_id=r['adapter_id']) for r in results],
        resolved_sources=results, base_id=common[0], runtime_profile=common[1], image=config['image'],
        target_feedback_used=False, prospective_rule_sha256=file_hash(rule_path), files=evidence,
        scope='Actual accepted source identities only. No FSC model inference or transfer efficacy; '
              'historical target exposure remains disclosed in the prospective rule.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('immutable selection already exists')
    atomic_json(args.output, resolve(read(args.config)))
