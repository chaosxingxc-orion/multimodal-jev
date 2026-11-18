import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from context_selection_curve import summarize


def config():
    return json.loads((Path(__file__).parents[1]/'configs/releases/context-prob-v1/experiment.json').read_text())


def points():
    c=config();result=[]
    for step in c['evaluation']['checkpoints']:
        metrics={k:.5 for k in [c['evaluation']['primary'],*c['evaluation']['secondary']]}
        metrics.update(queries=9226,threshold=.5,candidate_ids=[str(i) for i in range(60)],
            binary_nll_infinite=False,set_precision_defined_queries=9226)
        result.append(dict(step=step,result=dict(policy_id='p'+str(step),role='selection',
            release_sha256=c['release_sha256'],predictions_sha256='synthetic',metrics=metrics)))
    return result


def test_prefix_no_selection_and_full_tie_prefers_early_step():
    p=points();partial=summarize(p[:3],config(),initial_policy_id='p0')
    assert not partial['complete'] and partial['selected_checkpoint'] is None
    assert partial['missing_steps']==config()['evaluation']['checkpoints'][3:]
    full=summarize(p,config(),initial_policy_id='p0')
    assert full['complete'] and full['selected_checkpoint']['step']==0


def test_selection_tiebreak_nll_and_all_regressions_retained():
    p=points();p[1]['result']['metrics']['query_macro_set_f1']=.8
    p[2]['result']['metrics']['query_macro_set_f1']=.8
    p[2]['result']['metrics']['candidate_binary_nll']=.3
    report=summarize(p,config(),initial_policy_id='p0')
    assert report['selected_checkpoint']['step']==128
    assert report['primary_adjacent_decreases'][0]['step']==192
    assert report['last_primary_delta']==0
    assert len(report['curve'])==11


def test_infinite_nll_and_undefined_precision_do_not_disappear():
    p=points();m=p[0]['result']['metrics']
    m.update(candidate_binary_nll=None,binary_nll_infinite=True,set_precision=None,set_precision_defined_queries=0)
    report=summarize(p,config(),initial_policy_id='p0')
    assert report['selected_checkpoint']['step']==64
    assert report['curve'][0]['binary_nll_infinite']
    assert report['curve'][1]['delta_from_step0']['candidate_binary_nll'] is None


@pytest.mark.parametrize('fault',['missing','duplicate','reorder','population','role','threshold','ontology','initial','nan'])
def test_incomparable_curve_rejected(fault):
    p=points();initial='p0'
    if fault=='missing':del p[2]
    elif fault=='duplicate':p[2]=copy.deepcopy(p[1])
    elif fault=='reorder':p[1],p[2]=p[2],p[1]
    elif fault=='population':p[1]['result']['metrics']['queries']=9225
    elif fault=='role':p[1]['result']['role']='learning'
    elif fault=='threshold':p[1]['result']['metrics']['threshold']=.4
    elif fault=='ontology':p[1]['result']['metrics']['candidate_ids'].reverse()
    elif fault=='initial':initial='wrong'
    elif fault=='nan':p[1]['result']['metrics']['candidate_brier']=float('nan')
    with pytest.raises(ValueError):summarize(p,config(),initial_policy_id=initial)
