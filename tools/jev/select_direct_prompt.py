#!/usr/bin/env python3
"""Select reviewed direct audio prompts on disjoint TRAIN recording groups."""
import argparse
import json
import math
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.end_to_end import require_audio_task_route
from multimodal_jev.six_view_score import score_six_views
from compare_direct_readout_controls import load_run


def paired_outcomes(requests, targets, parent, candidate):
    """Paired error transitions; views are correlated, not pooled samples."""
    gold={r['id']:r for r in targets};views={};groups={}
    for request in requests:
        uid=request['id'];target=gold[uid];key=target['metadata']['variant']
        def correct(pred):
            value=pred[uid]['value']
            return type(value) is type(target['target']) and value==target['target']
        a,b=correct(parent),correct(candidate)
        cell=views.setdefault(key,{'both_correct':0,'both_wrong':0,'fixed':0,'regressed':0})
        cell['both_correct' if a and b else 'both_wrong' if not a and not b else 'fixed' if b else 'regressed']+=1
        groups.setdefault(request['source_group'],[]).append((a,b))
    whole={'both_correct':0,'both_wrong':0,'fixed':0,'regressed':0}
    for outcomes in groups.values():
        a,b=all(x for x,y in outcomes),all(y for x,y in outcomes)
        whole['both_correct' if a and b else 'both_wrong' if not a and not b else 'fixed' if b else 'regressed']+=1
    for cell in [*views.values(),whole]:
        n=cell['fixed']+cell['regressed'];k=min(cell['fixed'],cell['regressed'])
        cell['mcnemar_exact_two_sided_p']=min(1.,2*sum(math.comb(n,i) for i in range(k+1))/(2**n)) if n else 1.
    return {'by_view':views,'all_six_by_source_group':whole,
            'scope':'descriptive paired selection diagnostics; no multiplicity correction or held-out test significance claim'}


def rank(row):
    report = row['metrics']
    return (-report['all_six_correct'],
            -sum(p['both_correct'] for p in report['pairs'].values()),
            -sum(v['correct'] for v in report['by_variant'].values()),
            row['id'] != 'parent', row['instruction_characters'], row['id'])


def select(parent_run, candidates, review_path, targets_path, partition_path):
    review = json.loads(review_path.read_text())
    if set(candidates) != set(review['evaluation_eligible']):
        raise ValueError('every reviewed eligible candidate must complete')
    partition = json.loads(partition_path.read_text())
    learning, selection = set(partition['learning']), set(partition['selection'])
    if len(learning) != 64 or len(selection) != 256 or learning & selection:
        raise ValueError('registered disjoint64/256 groups required')
    parent, parent_predictions = load_run(parent_run, 'actual')
    cfg = parent['config']; require_audio_task_route(cfg)
    if review.get('parent_prompt_sha256') != parent['instruction_sha256']:
        raise ValueError('semantic review belongs to another parent prompt')
    if (review.get('selection_or_test_items_read_for_review') is not False
            or review.get('prompt_selected') is not False):
        raise ValueError('review must precede held-out evaluation and selection')
    if (cfg['split'] != 'train' or cfg['parameters_frozen'] is not True
            or set(cfg['source_groups']) != selection):
        raise ValueError('complete frozen selection population required')
    request_path = Path(cfg['requests'])
    if targets_path.resolve() != request_path.with_name('train.targets.jsonl').resolve():
        raise ValueError('paired TRAIN targets required')
    if file_hash(request_path) != parent['requests_sha256']:
        raise ValueError('requests changed')
    requests = [json.loads(line) for line in request_path.open()]
    requests = [r for r in requests if r['source_group'] in selection]
    if len(requests) != 1536: raise ValueError('complete six-view selection required')
    targets = [json.loads(line) for line in targets_path.open()]
    reports = []
    for candidate, root in [('parent', parent_run), *sorted(candidates.items())]:
        identity, indexed = load_run(Path(root), 'actual')
        for key in ('requests_sha256', 'models', 'runner_sha256', 'source_sha256'):
            if identity[key] != parent[key]: raise ValueError('actor identity changed: '+key)
        candidate_config, parent_config = dict(identity['config']), dict(cfg)
        instruction = Path(candidate_config.pop('instruction'))
        parent_config.pop('instruction')
        if candidate_config != parent_config: raise ValueError('only instruction may change')
        if file_hash(instruction) != identity['instruction_sha256']:
            raise ValueError('prompt changed')
        if candidate != 'parent':
            reviewed = next(r for r in review['candidates'] if r['id'] == candidate)
            if (reviewed['decision'] != 'eligible_for_selection_evaluation'
                    or reviewed['instruction_sha256'] != file_hash(instruction)):
                raise ValueError('prompt differs from semantic review')
        scored = score_six_views(requests, targets, list(indexed.values()))
        if len(scored['replicates']) != 1 or scored['source_groups'] != 256:
            raise ValueError('single complete replicate required')
        reports.append({'id': candidate, 'metrics': scored['replicates'][0],
            'instruction_characters': len(instruction.read_text()),
            'instruction': str(instruction), 'instruction_sha256': file_hash(instruction),
            'run': str(root), 'result_sha256': file_hash(Path(root)/'result.json'),
            'paired_to_parent':paired_outcomes(requests,targets,parent_predictions,indexed)})
    winner = min(reports, key=rank)
    return {'schema': 'jev-direct-audio-prompt-selection-v1', 'selected_id': winner['id'],
        'selected_instruction': winner['instruction'],
        'selected_instruction_sha256': winner['instruction_sha256'], 'comparisons': reports,
        'review_sha256': file_hash(review_path), 'partition_sha256': file_hash(partition_path),
        'targets_sha256': file_hash(targets_path), 'test_feedback_used': False,
        'weight_updates': 0, 'split': 'train',
        'rule': 'all-six correct, then sum four paired correct counts, then sum six view correct counts; ties parent, shorter, ID',
        'scope': 'disjoint TRAIN selection; not official test improvement'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('parent', 'review', 'targets', 'partition', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--candidate', action='append', default=[])
    args = p.parse_args()
    candidates = dict(item.split('=', 1) for item in args.candidate)
    if len(candidates) != len(args.candidate): raise ValueError('duplicate candidate')
    report = select(args.parent, candidates, args.review, args.targets, args.partition)
    with args.output.open('xb') as handle: handle.write(canonical(report))
    print(json.dumps(report, indent=2))
