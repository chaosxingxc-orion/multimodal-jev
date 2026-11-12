import copy
import json
import math
import pytest
from multimodal_jev.intent_binding_training import expand
from multimodal_jev.span_candidate_intervention import from_input,VARIANTS
from multimodal_jev.intent_binding_metrics import score,compare
from test_intent_policy_metrics import inputs


def fixture():
    item,_=inputs();item['audio_sha256']='audio';rows=[];records=[]
    for uid in ('a','b'):
        source=dict(item,id=uid)
        for row,target in expand(source):
            request=from_input(row['request']);gate=uid+'::Q::'+row['variant']
            rows.append(dict(row,request=request,target=target,gate_id=gate))
            probabilities=[]
            for q,y in zip(request.questions,target['targets'],strict=True):
                p=[.1/(len(q.values)-1)]*len(q.values);p[y]=.9;probabilities.append(list(map(math.log,p)))
            records.append(dict(id=gate,request_identity=dict(request=json.loads(json.dumps(request.model_input()))),candidate_logprobs=probabilities))
    return rows,records


def test_semantic_correctness_survives_rebinding_and_source_denominator():
    rows,records=fixture();result=score(rows,records,'Q')
    assert result['recordings']==2 and result['view_requests']==8
    assert result['all_views_correct']==2 and result['all_views_same_semantic_rate']==1
    assert result['mean_minimum_gold_probability']==pytest.approx(.9)
    assert all(v['metrics']['final_accuracy']==1 for v in result['variants'].values())


def test_fixed_position_is_not_semantic_invariance():
    rows,records=fixture()
    for record in records:record['candidate_logprobs'][-1]=list(map(math.log,[.8,.1,.1]))
    result=score(rows,records,'Q')
    assert result['all_views_correct']==0 and result['all_views_same_semantic_rate']<1


def test_agreement_on_wrong_semantics_is_not_success():
    rows,records=fixture()
    for row,record in zip(rows,records,strict=True):
        values=row['request'].questions[-1].values;wrong=next(v for v in values if v!=row['target']['target_values'][-1])
        # Choose the same wrong semantic label in every view.
        wrong=sorted(set(values)-{row['target']['target_values'][-1]})[0]
        p=[.05]*len(values);p[values.index(wrong)]=.9
        record['candidate_logprobs'][-1]=list(map(math.log,p))
    result=score(rows,records,'Q')
    assert result['all_views_same_semantic_rate']==1 and result['all_views_same_but_wrong']==2
    assert result['all_views_correct']==0


def test_pairing_reports_losses_and_other_sources_as_learning():
    rows,records=fixture();before=score(rows,records,'Q')
    record=records[0];p=record['candidate_logprobs'][-1];p[0],p[1]=p[1],p[0]
    after=score(rows,records,'Q');result=compare(before,after,['a'])
    assert result['groups']['updated']['variants']['native']['lost']==1
    assert result['groups']['other_learning']['recordings']==1
    assert result['groups']['all_learning']['all_views_correct_delta']==-.5


@pytest.mark.parametrize('defect',['missing','duplicate','gold','audio','weight','probability','request'])
def test_rejects_incomplete_or_mismatched_views(defect):
    rows,records=fixture()
    if defect=='missing':rows.pop();records.pop()
    elif defect=='duplicate':records.append(copy.deepcopy(records[0]))
    elif defect=='gold':rows[-1]['target']['target_values'][-1]='wrong'
    elif defect=='audio':rows[-1]['audio_sha256']='other'
    elif defect=='weight':rows[-1]['target']['request_weight']*=2
    elif defect=='probability':records[-1]['candidate_logprobs'][-1][0]=0.
    else:records[-1]['request_identity']['request']['audio']='wrong'
    with pytest.raises(ValueError):score(rows,records,'Q')


def test_comparison_rejects_changed_gold_or_duplicate_update_sources():
    rows,records=fixture();before=score(rows,records,'Q');after=copy.deepcopy(before)
    with pytest.raises(ValueError):compare(before,after,['a','a'])
    after['records'][0]['gold']='other'
    with pytest.raises(ValueError):compare(before,after,['a'])
