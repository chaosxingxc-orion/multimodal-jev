#!/usr/bin/env python3
"""Freeze the entire reviewed prompt roster before any selection inference."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.omni_selection_contract import actor_config
from compare_omni_waveform import compare, load


def register(controls, review_path, candidates, output):
    if json.loads((controls/'evaluation-terminal.json').read_text())['exit_code'] != 0:
        raise ValueError('successful complete controls required')
    for name in ('actual', 'zeros'):
        launch = json.loads((controls/name/'launch.json').read_text())
        terminal = json.loads((controls/name/'docker-terminal.json').read_text())
        if (terminal['container_id'] != launch['container_id']
                or terminal['state']['Status'] != 'exited' or terminal['state']['ExitCode'] != 0
                or json.loads((controls/name/'evaluation-terminal.json').read_text())['exit_code'] != 0):
            raise ValueError('successful exact control containers required')
    comparison = compare(controls/'actual/run', controls/'zeros/run')
    identity, _, _ = load(controls/'actual/run', 'actual')
    cfg = dict(identity['config'])
    if file_hash(cfg['asset_audit']) != identity['asset_audit_sha256']:
        raise ValueError('asset audit changed')
    review = json.loads(review_path.read_text())
    if (review['parent_prompt_sha256'] != identity['instruction_sha256']
            or review['selection_or_test_items_read_for_review'] is not False
            or review['prompt_selected'] is not False
            or not 1 <= len(candidates) <= 2
            or set(candidates) != set(review['evaluation_eligible'])
            or not set(candidates) <= {'p1', 'p2'}):
        raise ValueError('complete reviewed TRAIN-only prompt roster required')
    prompts = {'parent': identity['instruction_sha256']}
    for key, path in candidates.items():
        entries = [r for r in review['candidates'] if r['id'] == key]
        if (len(entries) != 1 or entries[0]['decision'] != 'eligible_for_selection_evaluation'
                or entries[0]['instruction_sha256'] != file_hash(path)
                or not 0 < len(Path(path).read_text()) <= 1500):
            raise ValueError('reviewed candidate identity required')
        prompts[key] = file_hash(path)
    if len(set(prompts.values())) != len(prompts):
        raise ValueError('duplicate prompt')
    partition = json.loads(Path(cfg['partition']).read_text())
    if len(partition['selection']) != 256 or len(set(partition['selection'])) != 256:
        raise ValueError('exact selection256 required')
    cfg.update(source_groups=partition['selection'], evaluation_phase='selection',
               purpose='frozen-qwen3-omni-reviewed-selection256')
    report = dict(schema='jev-omni-selection-protocol-v1', actor_config=actor_config(cfg),
        prompts=prompts, review_sha256=file_hash(review_path),
        requests_sha256=identity['requests_sha256'], partition_sha256=cfg['partition_sha256'],
        asset_audit_sha256=identity['asset_audit_sha256'],
        controls_actual_result_sha256=comparison['actual_result_sha256'],
        controls_zeros_result_sha256=comparison['zeros_result_sha256'],
        rule='all-six correct, then sum four paired correct counts, then sum six view correct counts; ties parent, shorter, ID',
        require_all_candidates_complete=True, test_feedback_used=False,
        scope='selection registration only; no inference dispatch or efficacy claim')
    with output.open('xb') as handle:
        handle.write(canonical(report))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('controls', 'review', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--candidate', action='append', default=[])
    a = p.parse_args()
    candidates = dict(item.split('=', 1) for item in a.candidate)
    if len(candidates) != len(a.candidate):
        raise ValueError('duplicate candidate ID')
    print(json.dumps(register(a.controls, a.review, candidates, a.output), indent=2))
