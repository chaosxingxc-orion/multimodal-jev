#!/usr/bin/env python3
"""Strict paired scoring after frozen full TEST runs finish; no reselection."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.direct_test_lock import validate_population
from multimodal_jev.six_view_score import score_six_views
from compare_direct_readout_controls import load_run
from select_direct_prompt import paired_outcomes


def score(root):
    lock = json.loads((root/'lock.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    if (file_hash(root/'selection.json') != lock['selection_sha256']
            or selection['selected_instruction_sha256'] != lock['selected_prompt_sha256']
            or selection['test_feedback_used'] is not False):
        raise ValueError('selected prompt changed')
    names = ['parent'] + (['selected'] if selection['selected_id'] != 'parent' else [])
    reports=[];parent_identity=None;parent_predictions=None
    for name in names:
        run=root/name/'run';identity,predictions=load_run(run,'actual');cfg=identity['config']
        launch=json.loads((root/name/'launch.json').read_text())
        terminal=json.loads((root/name/'docker-terminal.json').read_text())
        if (terminal['container_id']!=launch['container_id'] or terminal['state']['Status']!='exited'
                or terminal['state']['ExitCode']!=0):raise ValueError('container must exit successfully')
        if (identity['heldout_lock_sha256']!=file_hash(root/'lock.json')
                or identity['models']!=lock['models']
                or file_hash(cfg['instruction'])!=identity['instruction_sha256']):
            raise ValueError('locked identity changed')
        expected=lock['parent_prompt_sha256'] if name=='parent' else lock['selected_prompt_sha256']
        if identity['instruction_sha256']!=expected:raise ValueError('condition prompt changed')
        rows=[json.loads(line) for line in Path(cfg['requests']).open()]
        validate_population(cfg,rows)
        if set(predictions)!={r['id'] for r in rows}:raise ValueError('incomplete full TEST predictions')
        targets_path=Path(cfg['requests']).with_name('test.targets.jsonl')
        if file_hash(targets_path)!=lock['targets_sha256']:raise ValueError('TEST targets changed')
        targets=[json.loads(line) for line in targets_path.open()]
        if {r['id'] for r in targets}!={r['id'] for r in rows}:raise ValueError('full TEST targets required')
        if parent_identity is None:
            parent_identity,parent_predictions=identity,predictions
        else:
            for key in ('requests_sha256','models','runner_sha256','source_sha256','heldout_lock_sha256'):
                if identity[key]!=parent_identity[key]:raise ValueError('paired actor identity differs: '+key)
            a,b=dict(cfg),dict(parent_identity['config']);a.pop('instruction');b.pop('instruction')
            if a!=b:raise ValueError('only instruction may differ')
        metrics=score_six_views(rows,targets,list(predictions.values()))
        transitions=paired_outcomes(rows,targets,parent_predictions,predictions)
        transitions['scope']='descriptive paired TEST diagnostics; six correlated views; no multiplicity correction'
        reports.append(dict(id=name,run=str(run),metrics=metrics,
                            result_sha256=file_hash(run/'result.json'),targets_sha256=file_hash(targets_path),
                            paired_to_parent=transitions))
    return dict(schema='jev-direct-heldout-result-v1',selected_id=selection['selected_id'],
                selection_sha256=lock['selection_sha256'],lock_sha256=file_hash(root/'lock.json'),
                comparisons=reports,weight_updates=0,test_feedback_used_for_optimization=False,
                scope='final full FSC TEST and six correlated Jev views; format forced by readout; no reselection')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();report=score(a.root)
    with (a.root/'result.json').open('xb') as f:f.write(canonical(report))
    print(json.dumps(report,indent=2))
