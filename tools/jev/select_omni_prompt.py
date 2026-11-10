#!/usr/bin/env python3
"""Select all registered Omni prompts with strict free-generation scoring."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.generative_teacher_feedback import prediction_status
from multimodal_jev.omni_selection_contract import select_registered
from multimodal_jev.six_view_score import score_six_views
from select_direct_prompt import rank, mcnemar_p


def load(root, protocol_path, prompt_id):
    identity = json.loads((root/'identity.json').read_text())
    cfg = identity['config']
    protocol = json.loads(protocol_path.read_text())
    result = json.loads((root/'result.json').read_text())
    launch = json.loads((root.parent/'launch.json').read_text())
    terminal = json.loads((root.parent/'docker-terminal.json').read_text())
    if (terminal['container_id'] != launch['container_id']
            or terminal['state']['Status'] != 'exited' or terminal['state']['ExitCode'] != 0
            or json.loads((root.parent/'evaluation-terminal.json').read_text())['exit_code'] != 0
            or json.loads((root/'terminal.json').read_text())['exit_code'] != 0
            or result['status'] != 'passed' or result['weight_updates'] != 0
            or result['model_files_unchanged'] is not True or result['audio_condition'] != 'actual'):
        raise ValueError('complete successful frozen selection run required')
    if (file_hash(protocol_path) != cfg['selection_protocol_sha256']
            or identity['selection_protocol_sha256'] != cfg['selection_protocol_sha256']
            or protocol['prompts'][prompt_id] != identity['instruction_sha256']
            or file_hash(cfg['instruction']) != identity['instruction_sha256']
            or file_hash(cfg['requests']) != identity['requests_sha256']
            or file_hash(cfg['asset_audit']) != identity['asset_audit_sha256']
            or file_hash(root/'predictions.jsonl') != result['predictions_sha256']):
        raise ValueError('selection input/output identity changed')
    rows = select_registered(cfg, [json.loads(line) for line in Path(cfg['requests']).open()])
    predictions = [json.loads(line) for line in (root/'predictions.jsonl').open()]
    indexed = {p['example_id']:p for p in predictions}
    ids = json.loads((root/'selection.json').read_text())['ids']
    if (len(predictions) != 1536 or len(indexed) != 1536
            or len(ids) != 1536 or set(ids) != set(indexed)
            or set(indexed) != {r['id'] for r in rows}
            or result['requests'] != 1536 or result['predictions'] != 1536
            or any(p['replicate'] != 0 or p['audio_condition'] != 'actual'
                   or p['split'] != 'train' for p in predictions)):
        raise ValueError('complete unique selection predictions required')
    return identity, rows, indexed, launch['image']


def paired(requests, targets, parent, candidate):
    gold = {r['id']:r for r in targets}
    views, groups = {}, {}
    def cell():
        return dict(both_correct=0, both_wrong=0, fixed=0, regressed=0)
    def count(c, a, b):
        c['both_correct' if a and b else 'both_wrong' if not a and not b
          else 'fixed' if b else 'regressed'] += 1
    for request in requests:
        uid = request['id']; target = gold[uid]
        a = prediction_status(request, parent[uid], target)['correct']
        b = prediction_status(request, candidate[uid], target)['correct']
        count(views.setdefault(target['metadata']['variant'], cell()), a, b)
        groups.setdefault(request['source_group'], []).append((a, b))
    whole = cell()
    for outcomes in groups.values():
        count(whole, all(a for a,b in outcomes), all(b for a,b in outcomes))
    for c in [*views.values(), whole]:
        c['mcnemar_exact_two_sided_p'] = mcnemar_p(c['fixed'], c['regressed'])
    return dict(by_view=views, all_six_by_source_group=whole,
                scope='descriptive selection diagnostics; correlated views not pooled; no TEST claim')


def select(protocol_path, runs):
    protocol = json.loads(protocol_path.read_text())
    if set(runs) != set(protocol['prompts']):
        raise ValueError('every registered prompt must complete before selection')
    parent, rows, predictions, image = load(runs['parent'], protocol_path, 'parent')
    targets_path = Path(parent['config']['requests']).with_name('train.targets.jsonl')
    targets = [json.loads(line) for line in targets_path.open()]
    reports = []
    for key, root in sorted(runs.items()):
        identity, candidate_rows, indexed, candidate_image = load(root, protocol_path, key)
        if candidate_image != image or candidate_rows != rows:
            raise ValueError('runtime image or evaluation population differs')
        for field in ('requests_sha256','asset_audit_sha256','model_files','runner_sha256',
                      'source_sha256','versions','population_selector_sha256'):
            if identity[field] != parent[field]:
                raise ValueError('actor identity differs: ' + field)
        scored = score_six_views(rows, targets, list(indexed.values()))
        if scored['source_groups'] != 256 or len(scored['replicates']) != 1:
            raise ValueError('complete single selection replicate required')
        instruction = Path(identity['config']['instruction'])
        reports.append(dict(id=key, metrics=scored['replicates'][0],
            instruction_characters=len(instruction.read_text()), instruction=str(instruction),
            instruction_sha256=file_hash(instruction), result_sha256=file_hash(root/'result.json'),
            run=str(root), paired_to_parent=paired(rows, targets, predictions, indexed)))
    winner = min(reports, key=rank)
    return dict(schema='jev-omni-prompt-selection-v1', selected_id=winner['id'],
        selected_instruction=winner['instruction'], selected_instruction_sha256=winner['instruction_sha256'],
        comparisons=reports, protocol_sha256=file_hash(protocol_path),
        targets_sha256=file_hash(targets_path), rule=protocol['rule'], test_feedback_used=False,
        weight_updates=0, scope='disjoint TRAIN selection only; no official TEST efficacy claim')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--run', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs = dict(item.split('=', 1) for item in a.run)
    if len(runs) != len(a.run):
        raise ValueError('duplicate run ID')
    report = select(a.protocol, {k:Path(v) for k,v in runs.items()})
    with a.output.open('xb') as handle:
        handle.write(canonical(report))
    print(json.dumps(report, indent=2))
