"""Compile shared Stage1 panels and verify full TRAIN exposure across matched arms."""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.stage1_candidate_layout import build_view, epoch_layouts, training_request
from tools.reports.build_jev_boundary_panel import source_rows
from tools.reports.jev_train_confusions import pinned


def run(config, provenance):
    design = json.loads(pinned(Path(config['design']), config['design_sha256']))
    base = json.loads(pinned(Path(config['stage1']), config['stage1_sha256']))['stage1']
    task, rows = source_rows(config)
    ids = [r['id'] for r in rows]
    if base['batch_config']['source_queries'] != len(ids) or base['batch_config']['epochs'] != 2:
        raise ValueError('matched full source Stage1 budget required')
    request = compile_candidates('audio', [dict(id=q['id'], definition=q['description'])
        for q in task['schema']['candidates']], task['thinking']['requirements'], semantics='exclusive_event')
    panels, arms = {}, {}
    if 'training' in design:
        if (design.get('schema')!='jev-next-simple-two-stage-v1' or 'one_factor_arms' in design
                or design['training']['layouts'] != ['native','both']
                or design['stage1']['sha256'] != config['stage1_sha256']):
            raise ValueError('single synchronized training configuration required')
        project=Path(__file__).resolve().parents[2]
        from multimodal_jev.source_snapshot import regular
        for key in ('stage2','dataset'):
            ref=design[key]
            payload=json.loads(pinned(regular(project,ref['path']),ref['sha256']))
            if key=='stage2':
                pinned(regular(project,payload['objective_source']),payload['objective_sha256'])
                if payload['options']['task_rl_weight']!=0. or not all(
                        payload['options'][k]>0 for k in ('ranking_rl_weight','existence_rl_weight')):
                    raise ValueError('two active RL credits required')
        settings={'training':design['training']}
    else:
        settings=dict(control=design['control'], **design['one_factor_arms'])
    for arm, cfg in settings.items():
        assignment = [epoch_layouts(ids, epoch, cfg) for epoch in range(2)]
        if len(cfg['layouts'])==2 and any(assignment[0][uid]==assignment[1][uid] for uid in ids):
            raise ValueError('two different views required for every real recording')
        for layout in cfg['layouts']:
            view = vars(build_view(request, layout=layout, seed=cfg['permutation_seed']))
            if arm=='training':
                training_request(request,layout=layout,seed=cfg['permutation_seed'])
            if layout in panels and panels[layout] != view:
                raise ValueError('one shared compiler output required per layout')
            panels[layout] = view
        arms[arm] = dict(epochs=[dict(counts=dict(Counter(a.values())), assignment_sha256=content_hash(a))
                                 for a in assignment], real_exposures=sum(map(len, assignment)))
    report = dict(arms=arms, panels=panels, stage1_objective=base['instruction_objective'],
        source_identity=config['clean']['sha256'], design_sha256=config['design_sha256'],
        stage1_design_sha256=config['stage1_sha256'], provenance=provenance,
        npu_executed=False, training_dispatch_allowed=False)
    commit_json(config['output'], report)
    return arms


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--local',required=True)
    args=parser.parse_args()
    print(json.dumps(run(*load_config(args.config,args.local)),indent=2))
