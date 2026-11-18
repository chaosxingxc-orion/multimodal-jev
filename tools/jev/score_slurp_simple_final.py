#!/usr/bin/env python3
"""Score a terminal single-Final run against the authenticated full selection."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.slurp_official_training import load, example
from multimodal_jev.simple_final_metrics import score


def run(config):
    for path, digest in config['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('scoring dependency changed: '+path)
    backend_mode = 'backend_results' in config
    evidence_keys = ('inputs', 'backend_results', 'terminal') if backend_mode else ('predictions', 'terminal')
    for key in evidence_keys:
        if config['pins'].get(config[key]) != file_hash(config[key]):
            raise ValueError('pinned prediction/terminal evidence required')
    terminal = json.loads(Path(config['terminal']).read_text())
    if terminal.get('exit_code') != 0 or (not backend_mode and terminal.get('policy_id') != config['policy_id']):
        raise ValueError('completed same-policy inference required')
    if backend_mode:
        for key in ('inference_config', 'launch', 'observed'):
            if config['pins'].get(config[key]) != file_hash(config[key]):
                raise ValueError('pinned actual inference execution evidence required')
        inference = json.loads(Path(config['inference_config']).read_text())
        launch = json.loads(Path(config['launch']).read_text())
        observed = json.loads(Path(config['observed']).read_text())
        if (observed['container_id'] != launch['container_id'] or observed['image'] != launch['image']
                or observed['state']['Status'] != 'exited' or observed['state']['ExitCode'] != 0
                or launch['config_sha256'] != file_hash(config['inference_config'])
                or inference['inputs'] != config['inputs']
                or inference['pins'].get(config['inputs']) != file_hash(config['inputs'])
                or inference['base_id'] + ':' + inference['adapter_id'] != config['policy_id']):
            raise ValueError('actual inference lineage differs')
        for path, digest in inference['pins'].items():
            if file_hash(path) != digest:
                raise ValueError('inference dependency changed: ' + path)
        if Path(config['backend_results']).parent != Path(config['terminal']).parent:
            raise ValueError('backend terminal must belong to exact output directory')
    rows, task, _ = load(config['release'], config['release_sha256'], 'selection', purpose='evaluate')
    classes = list(example(rows[0], task, 'direct', 'native')[0].questions[0].values)
    if backend_mode:
        from multimodal_jev.simple_final_predictions import from_backend
        from multimodal_jev.simple_selection_inputs import verify as verify_inputs
        artifact = json.loads(Path(config['inputs']).read_text())
        if artifact.get('release_sha256') != config['release_sha256']:
            raise ValueError('input release differs from scoring release')
        verify_inputs(artifact, rows, task, config['release_sha256'])
        records = from_backend(artifact, json.loads(Path(config['backend_results']).read_text()),
                               policy_id=config['policy_id'])
    else:
        records = [json.loads(line) for line in Path(config['predictions']).read_text().splitlines()]
    result = score(rows, classes, records, policy_id=config['policy_id'])
    result.update(release_sha256=config['release_sha256'],
                  prediction_evidence_sha256=file_hash(config['backend_results'] if backend_mode else config['predictions']),
                  policy_id=config['policy_id'], role='selection',
                  provenance_limit='Requires separately authenticated inference receipt; file hashes alone do not certify model execution.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    atomic_json(a.output, run(json.loads(a.config.read_text())))
