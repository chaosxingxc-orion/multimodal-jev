#!/usr/bin/env python3
"""Score an entirely completed, prospectively pinned FSC checkpoint matrix.

No dispatch, target-driven checkpoint selection, partial scores or template edits.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.fsc_backend_records import normalize_gate
from multimodal_jev.fsc_transfer_population import scoring_targets
from multimodal_jev.fsc_transfer_metrics import score_policy, compare_checkpoints, speaker_bootstrap
from queue_span_dp4_resume import inspect, check_pins


def read(path): return json.loads(Path(path).read_text())


def authenticate_gate(entry, protocol_path, protocol_sha256, protocol, population, task):
    root = Path(entry['root']); launch = read(root/'launch.json'); cfg = read(root/'config.json')
    if (launch['image'] != protocol['image'] or launch['physical_devices'] != [4]
            or launch['config_sha256'] != file_hash(root/'config.json')
            or launch['source_manifest_sha256'] != file_hash(root/'source-manifest.json')):
        raise ValueError('launched gate identity differs')
    # Protocol must be included in the config authenticated at Docker creation.
    # It cannot be registered after viewing this run's target-domain results.
    required = {str(protocol_path): protocol_sha256, protocol['task']: protocol['task_sha256'],
                protocol['population']: protocol['population_sha256']}
    if any(cfg['pins'].get(p) != h for p, h in required.items()):
        raise ValueError('prospective protocol/task/population pins required')
    if (cfg['base_id'] != protocol['base_id'] or cfg['adapter_id'] != entry['adapter_id']
            or cfg['runtime_profile'] != protocol['runtime_profile']
            or cfg['population'] != {'recordings': population['recordings'], 'requests': 6*population['recordings']}):
        raise ValueError('frozen policy, runtime or complete population differs')
    check_pins(cfg['pins'])
    check_pins({str(root/'source'/p): h for p, h in read(root/'source-manifest.json').items()})
    input_path = Path(cfg['inputs'])
    if cfg['pins'].get(str(input_path)) != file_hash(input_path):
        raise ValueError('pinned CPU inputs required')
    state = inspect([launch['container_id']])[launch['container_id']]
    if (state['Image'] != protocol['image'] or state['State']['Status'] != 'exited'
            or state['State']['ExitCode'] != 0):
        raise ValueError('exact successfully completed gate required')
    for name in ('terminal.json', 'hf/terminal.json', 'vllm/terminal.json'):
        if read(root/'run'/name)['exit_code'] != 0: raise ValueError('gate phase failed')
    ready = read(root/'run/READY.json'); hf = read(root/'run/hf/result.json'); vl = read(root/'run/vllm/result.json')
    if (ready['passed'] is not True or ready['base_id'] != protocol['base_id']
            or ready['adapter_id'] != entry['adapter_id'] or ready['requests'] != 6*population['recordings']
            or ready['result_sha256'] != file_hash(root/'run/vllm/result.json')):
        raise ValueError('complete policy readiness required')
    normalized = normalize_gate(task, population['records'], read(input_path), hf, vl,
                                base_id=protocol['base_id'], adapter_id=entry['adapter_id'], seed=protocol['binding_seed'])
    evidence = {str(root/p): file_hash(root/p) for p in (
        'config.json', 'source-manifest.json', 'launch.json', 'run/READY.json', 'run/terminal.json',
        'run/hf/result.json', 'run/hf/terminal.json', 'run/vllm/result.json', 'run/vllm/terminal.json')}
    evidence[str(input_path)] = file_hash(input_path)
    return normalized, evidence


def run(protocol_path, expected_sha256):
    protocol_path = Path(protocol_path).resolve()
    if file_hash(protocol_path) != expected_sha256: raise ValueError('protocol identity differs')
    protocol = read(protocol_path)
    if (protocol['schema'] != 'fsc-span-transfer-evaluation-v1'
            or protocol['selection_rule'] != 'source_only_frozen_before_target_inference'
            or protocol['target_feedback_used_for_selection'] is not False
            or protocol['binding_seed'] != 317 or protocol['bootstrap_seed'] != 941
            or protocol['bootstrap_replicates'] != 2000):
        raise ValueError('registered target-blind comparison contract required')
    entries = protocol['checkpoints']; keys = [e['key'] for e in entries]
    if (len(entries) < 3 or len(set(keys)) != len(keys) or keys[0] != 'step0'
            or len({e['root'] for e in entries}) != len(entries)):
        raise ValueError('initial and at least two separately evaluated subsequent checkpoints required')
    pins = {protocol['task']: protocol['task_sha256'], protocol['population']: protocol['population_sha256'],
            protocol['source_selection']: protocol['source_selection_sha256']}
    check_pins(pins); population = read(protocol['population']); task = read(protocol['task'])
    selection = read(protocol['source_selection'])
    if (selection['checkpoints'] != [{'key': e['key'], 'adapter_id': e['adapter_id']} for e in entries]
            or selection['target_feedback_used'] is not False):
        raise ValueError('source-only selection receipt differs')
    results = {}; audits = {}; evidence = dict(pins); normalized = {}
    # Authenticate the whole matrix before extracting or scoring target labels.
    for entry in entries:
        normalized[entry['key']], files = authenticate_gate(entry, protocol_path, expected_sha256, protocol, population, task)
        evidence.update(files)
    targets = scoring_targets(population, task)
    for entry in entries:
        key = entry['key']; value = normalized[key]; audits[key] = value['audit']
        results[key] = score_policy(task, population['records'], targets, value['records'],
            policy_id=protocol['base_id']+':'+entry['adapter_id'], seed=protocol['binding_seed'])
    comparisons = {key: dict(point_estimates=compare_checkpoints(results['step0'], results[key]),
        uncertainty=speaker_bootstrap(results['step0'], results[key], seed=941, replicates=2000)) for key in keys[1:]}
    evidence[str(protocol_path)] = expected_sha256
    repo = Path(__file__).resolve().parents[2]
    for p in [Path(__file__), *[repo/'src/multimodal_jev'/n for n in (
            'fsc_backend_records.py', 'fsc_transfer_population.py', 'fsc_transfer_metrics.py', 'fsc_span_transfer.py')]]:
        evidence[str(p)] = file_hash(p)
    return dict(protocol_sha256=expected_sha256, split=population['split'], policies=results,
        comparisons=comparisons, backend_audits=audits, files=evidence,
        scope='Complete fixed FSC matrix; source checkpoint selection and historical target exposure must be disclosed. '
              'Speaker intervals exclude training-seed variability and repeated-command dependence. '
              'A passed evaluation execution does not imply positive transfer or reasoning efficacy.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('immutable comparison already exists')
    atomic_json(a.output, run(a.protocol, a.protocol_sha256))
