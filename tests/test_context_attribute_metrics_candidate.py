import copy
import json
from pathlib import Path
import pytest
from tools.reports.context_attribute_metrics_candidate import score_records,select_curve,paired_delta,STEPS


def fixture(arm='q_credit'):
    root=Path('.local/decision-objective-research/slurp-native-attribute-selection-candidate-v1')
    menu=json.loads((root/'catalog.json').read_text())
    def first(name):
        with (root/name).open() as f:return [json.loads(next(f)) for _ in range(4)]
    inputs=first('inputs.jsonl');targets=first('targets.jsonl')
    nodes=[n for n in menu['nodes'] if arm!='direct' or n['family']=='final'];ids=[n['id'] for n in nodes]
    preds=[]
    for inp,t in zip(inputs,targets,strict=True):
        truth=dict(zip(t['candidate_ids'],t['labels'],strict=True))
        preds.append(dict(id=inp['id'],arm=arm,policy_id='p',decision_ids=ids,
                          prob=[.9 if truth[k] else .1 for k in ids]))
    return inputs,targets,preds,menu


def score(data,arm='q_credit'):
    return score_records(*data,arm=arm,policy_id='p',expected_count=4)


def test_attributes_do_not_repair_final_and_direct_equivalence():
    data=fixture();baseline=score(data);assert baseline['metrics']['final']['query_macro_set_f1']==1
    altered=copy.deepcopy(data)
    for p in altered[2]:
        p['prob']=[.1 if not k.startswith('final:') else v for k,v in zip(p['decision_ids'],p['prob'],strict=True)]
    result=score(altered)
    assert result['metrics']['final']==baseline['metrics']['final']
    assert result['metrics']['scenario']['query_macro_set_f1']==0
    direct=score(fixture('direct'),'direct');assert direct['metrics']['final']==baseline['metrics']['final']
    assert paired_delta(result,baseline)['final_set_f1_delta']==0


@pytest.mark.parametrize('fault',['duplicate','partial','policy','order','gold_hash','missing','slice','nan'])
def test_corrupt_population_rejected(fault):
    data=fixture();inputs,targets,preds,menu=data
    if fault=='duplicate':inputs[1]=inputs[0]
    if fault=='partial':preds.pop()
    if fault=='policy':preds[0]['policy_id']='wrong'
    if fault=='order':preds[0]['decision_ids']=list(reversed(preds[0]['decision_ids']))
    if fault=='gold_hash':targets[0]['input_sha256']='wrong'
    if fault=='missing':targets[0]['supervised'][0]=False
    if fault=='slice':targets[0]['structural_slice']='unknown'
    if fault=='nan':preds[0]['prob'][0]=float('nan')
    with pytest.raises(ValueError):score(data)


def test_selection_final_only_ties_and_infinite_nll():
    s=score(fixture());points=[dict(step=k,score=copy.deepcopy(s)) for k in STEPS]
    assert select_curve(points)['step']==0
    points[1]['score']['metrics']['scenario']['query_macro_set_f1']=999
    assert select_curve(points)['step']==0
    points[1]['score']['metrics']['final']['candidate_binary_nll']=.001
    assert select_curve(points)['step']==64
    points[1]['score']['metrics']['final'].update(candidate_binary_nll=None,binary_nll_infinite=True)
    assert select_curve(points)['step']==0
    with pytest.raises(ValueError):select_curve(points[:-1])
    bad=copy.deepcopy(s);bad['records'][0]['source_group']='wrong'
    with pytest.raises(ValueError):paired_delta(s,bad)
