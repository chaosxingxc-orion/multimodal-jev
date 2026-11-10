#!/usr/bin/env python3
"""Select a reviewed prompt using only the registered TRAIN selection population."""
import argparse
import hashlib
import json
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.consumption_score import score
from multimodal_jev.frozen_evidence import load_evidence
from compare_frozen_control import compare


def select(original, control, candidates, review_path, targets_path, split_path):
    original, control = Path(original), Path(control)
    if not compare(original, control)['passed']:
        raise ValueError('ASR reuse control must match original exactly')
    review = json.loads(Path(review_path).read_text())
    if set(candidates) != set(review['evaluation_eligible']) or not candidates:
        raise ValueError('evaluate every eligible candidate before selection')
    split = json.loads(Path(split_path).read_text())
    if set(split['learning']) & set(split['selection']):
        raise ValueError('learning/selection overlap')
    parent = json.loads((control / 'identity.json').read_text())
    cfg = parent['config']
    if (cfg['split'] != 'train' or set(cfg['source_groups']) != set(split['selection'])
            or cfg['sampling'] != {'temperature': 0, 'n': 1}):
        raise ValueError('registered greedy TRAIN selection population required')
    if file_hash(cfg['requests']) != parent['requests_sha256']:
        raise ValueError('requests changed')
    requests = [json.loads(line) for line in Path(cfg['requests']).read_text().splitlines()]
    requests = [r for r in requests if r['source_group'] in set(split['selection'])]
    if len(requests) != 256 or any(r['split'] != 'train' for r in requests):
        raise ValueError('complete selection256 required')
    if Path(targets_path).resolve() != Path(cfg['requests']).with_name('train.targets.jsonl').resolve():
        raise ValueError('paired TRAIN targets required')
    targets = list(map(json.loads, Path(targets_path).read_text().splitlines()))
    gold = {r['id']: r for r in targets}
    if len(gold) != len(targets): raise ValueError('duplicate target IDs')
    request_hashes = {r['id']: hashlib.sha256(canonical(r)).hexdigest() for r in requests}
    if any(gold[k]['request_sha256'] != v for k, v in request_hashes.items()):
        raise ValueError('gold request custody changed')
    comparisons = []
    prompt_paths = {}
    configs_same = ('route', 'parameters_frozen', 'requests', 'split', 'source_groups',
                    'dtype', 'device', 'seed', 'sampling', 'max_context_tokens',
                    'max_new_tokens', 'asr_max_new_tokens', 'asr_evidence')
    for candidate_id, root in [('parent', control), *sorted(candidates.items())]:
        run = Path(root)
        identity = json.loads((run / 'identity.json').read_text())
        result = json.loads((run / 'result.json').read_text())
        terminal = json.loads((run / 'terminal.json').read_text())
        if (terminal['exit_code'] != 0 or result['status'] != 'passed'
                or result['weight_updates'] != 0 or not result['model_files_unchanged']):
            raise ValueError('completed frozen candidate required')
        if any(identity['config'][k] != cfg[k] for k in configs_same):
            raise ValueError('candidate evaluation condition changed')
        if identity['models'] != parent['models'] or identity['requests_sha256'] != parent['requests_sha256']:
            raise ValueError('candidate model or requests changed')
        load_evidence(cfg['asr_evidence'], requests, cfg, identity['models'])
        path = Path(identity['config']['instruction'])
        if file_hash(path) != identity['instruction_sha256']: raise ValueError('instruction changed')
        if candidate_id != 'parent':
            reviewed = next(r for r in review['candidates'] if r['id'] == candidate_id)
            if reviewed['decision'] != 'eligible_for_selection_evaluation' or reviewed['instruction_sha256'] != file_hash(path):
                raise ValueError('candidate differs from reviewed instruction')
        if file_hash(run / 'predictions.jsonl') != result['predictions_sha256']:
            raise ValueError('candidate predictions changed')
        predictions = list(map(json.loads, (run / 'predictions.jsonl').read_text().splitlines()))
        if any(r['request_sha256'] != request_hashes.get(r['example_id']) for r in predictions):
            raise ValueError('candidate request binding changed')
        if len(predictions) != 256 or any(r['replicate'] != 0 for r in predictions):
            raise ValueError('single greedy evaluation required')
        report = score(requests, targets, predictions)['replicates'][0]
        prompt_paths[candidate_id] = path
        comparisons.append({'id': candidate_id, **report, 'instruction_sha256': file_hash(path),
                            'instruction_characters': len(path.read_text()), 'run': str(run),
                            'result_sha256': file_hash(run / 'result.json'),
                            'input_tokens': sum(r['input_tokens'] for r in predictions),
                            'output_tokens': sum(len(r['token_ids']) for r in predictions),
                            'elapsed_seconds': result['elapsed_seconds']})
    winner = min(comparisons, key=lambda r: (-r['correct'], r['id'] != 'parent',
                                           r['instruction_characters'], r['id']))
    return {'schema': 'jev-frozen-prompt-selection-v1', 'split': 'train',
            'purpose': 'selection, not official test performance', 'selected_id': winner['id'],
            'selected_instruction': str(prompt_paths[winner['id']]),
            'selected_instruction_sha256': winner['instruction_sha256'],
            'comparisons': comparisons, 'review_sha256': file_hash(review_path),
            'split_sha256': file_hash(split_path), 'targets_sha256': file_hash(targets_path),
            'rule': 'highest joint accuracy; tie parent, then shortest instruction, then ID',
            'test_feedback_used': False, 'weight_updates': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('original', 'control', 'review', 'targets', 'split', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--candidate', action='append', required=True, help='id=/absolute/run')
    args = p.parse_args()
    candidates = dict(item.split('=', 1) for item in args.candidate)
    if len(candidates) != len(args.candidate): raise ValueError('duplicate candidates')
    report = select(args.original, args.control, candidates, args.review, args.targets, args.split)
    with args.output.open('xb') as handle: handle.write(canonical(report))
    print(json.dumps(report, indent=2))
