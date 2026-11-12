import math
from copy import deepcopy

import pytest

from multimodal_jev.span_failure_analysis import diagnose


def fixture():
    rows, predictions = [], []
    for polarity, final in [('positive', True), ('negative', False)]:
        rule = dict(op='all', args=[dict(op='fact', key='a'), dict(op='fact', key='b')])
        if not final: rule = dict(op='not', arg=rule)
        row = dict(id=polarity, arm='structured', targets=[True,True,final], supervised=[True]*3,
                   provenance=dict(rule=rule, dependency_keys=['a','b'],dependency_targets=[True,True],
                       recording_id='audio',complement_pair_id='both',rule_polarity=polarity))
        rows.append(row)
        predictions.append(dict(id=polarity,predictions=[True]*3,
            candidate_logprobs=[[math.log(.1),math.log(.9)]]*3))
    return rows,predictions


def test_fact_correctness_does_not_hide_nonresponsive_final():
    rows,preds=fixture();result=diagnose(rows,preds)['arms']['structured']
    assert result['fact_pair_accuracy']==1
    assert result['final_accuracy']==.5
    assert result['final_errors_with_both_facts_correct']==1
    assert result['deterministic_executor_accuracy']==1
    assert result['complement']['prediction_flips']==0
    assert result['complement']['exactly_one_correct']==1
    assert result['complement']['mean_probability_sum_error']==pytest.approx(.8)


def test_reordered_dependencies_are_mapped_before_execution():
    rows,preds=fixture()
    for r,p in zip(rows,preds):
        r['arm']='reordered';r['targets'][:2]=[False,True];r['provenance']['dependency_targets']=[True,False]
        r['targets'][-1]=not r['targets'][-1]
        p['predictions'][:2]=[False,True];p['candidate_logprobs'][0]=[math.log(.9),math.log(.1)]
    assert diagnose(rows,preds)['arms']['reordered']['fact_pair_accuracy']==1


@pytest.mark.parametrize('damage',['duplicate','missing_pair','invalid_probability','wrong_argmax','bad_gold'])
def test_invalid_diagnostics_rejected(damage):
    rows,preds=deepcopy(fixture())
    if damage=='duplicate':preds.append(preds[0])
    elif damage=='missing_pair':rows.pop();preds.pop()
    elif damage=='invalid_probability':preds[0]['candidate_logprobs'][0]=[0.,0.]
    elif damage=='wrong_argmax':preds[0]['predictions'][0]=False
    elif damage=='bad_gold':rows[0]['targets'][-1]=False
    with pytest.raises(ValueError):diagnose(rows,preds)
