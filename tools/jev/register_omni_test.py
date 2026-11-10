#!/usr/bin/env python3
"""Reproduce complete TRAIN selection before freezing the official TEST roster."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.omni_test_contract import REQUEST_SHA, TARGET_SHA, test_actor_config
from select_omni_prompt import select


def register(selection_root, output):
    if json.loads((selection_root/'evaluation-terminal.json').read_text())['exit_code'] != 0:
        raise ValueError('successful complete selection queue required')
    prior_path = selection_root/'selection-protocol.json'
    report_path = selection_root/'selection-result.json'
    prior = json.loads(prior_path.read_text())
    report = json.loads(report_path.read_text())
    reproduced = select(prior_path, {k:selection_root/k/'run' for k in prior['prompts']})
    if reproduced != report:
        raise ValueError('selection result does not reproduce')
    parent = json.loads((selection_root/'parent/run/identity.json').read_text())
    cfg = test_actor_config(parent['config'])
    for key in ('partition_sha256', 'source_groups'):
        cfg.pop(key, None)
    cfg.update(split='test', evaluation_phase='test',
        requests=str(Path(cfg['requests']).with_name('test.requests.jsonl')),
        purpose='frozen-qwen3-omni-official-fsc-test', wall_seconds=86400)
    # Read and hash held-out inputs only; no targets or prior TEST outputs.
    if file_hash(cfg['requests']) != REQUEST_SHA:
        raise ValueError('official TEST requests changed')
    prompts = {'parent':prior['prompts']['parent']}
    winner = report['selected_id']
    if winner != 'parent':
        prompts[winner] = prior['prompts'][winner]
    protocol = dict(schema='jev-omni-test-protocol-v1', actor_config=cfg, prompts=prompts,
        selection_result=str(report_path.resolve()), selection_result_sha256=file_hash(report_path),
        selection_protocol=str(prior_path.resolve()), selection_protocol_sha256=file_hash(prior_path),
        asset_audit_sha256=parent['asset_audit_sha256'], requests_sha256=REQUEST_SHA,
        targets_sha256=TARGET_SHA, test_feedback_used=False,
        actor_identity={k:parent[k] for k in ('runner_sha256','source_sha256','versions','model_files')},
        image=json.loads((selection_root/'parent/launch.json').read_text())['image'],
        scope='full official TEST registration after complete TRAIN selection; no inference dispatched')
    with output.open('xb') as handle:
        handle.write(canonical(protocol))
    return protocol


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--selection-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(register(a.selection_root, a.output), indent=2))
