import copy
import pytest
from multimodal_jev.intent_matched_effects import compare


def score(correct, adapter, structured=False):
    # Same truth, deliberately different direct versus structured initial accuracy.
    rows=[dict(id='a',gold=['x'],prediction=['x' if correct else 'y'],correct=correct)]
    m=dict(records=rows,final_accuracy=float(correct),final_macro_f1_observed=float(correct),
           final_expected_reward=.8 if correct else .2,final_nll=.2 if correct else 1.6)
    return dict(role='learning',base_id='base',adapter_id=adapter,
        reports={w:dict(admission={'fixed':True},metrics=copy.deepcopy(m)) for w in ('D','Q')})


def fixture():
    arms={a:[dict(update=0,score=score(a!='D','initial')),
             dict(update=1,score=score(a!='Q',a+'-trained'))] for a in ('D','Q','P')}
    return arms,[['a']]


def test_separates_template_start_from_process_credit():
    arms,schedule=fixture();out=compare(arms,schedule);last=out['curve'][-1]
    assert out['complete'] and last['source_uses']==1
    assert last['effects']['process_credit']['final_difference']['final_accuracy']==1
    assert last['effects']['total_structure']['final_difference']['final_accuracy']==0
    assert last['effects']['total_structure']['change_difference']['final_accuracy']==-1
    assert last['effects']['process_credit']['left_only_correct']==1
    assert out['metric_direction']['final_nll']=='lower'


@pytest.mark.parametrize('defect',['partial','initial','qp_initial','label','role','duplicate','correctness','aggregate','schedule'])
def test_rejects_invalid_comparisons(defect):
    arms,schedule=fixture()
    if defect=='partial':arms['P'].pop()
    if defect=='initial':arms['P'][0]['score']['adapter_id']='other'
    if defect=='qp_initial':arms['P'][0]['score']['reports']['Q']['admission']={'different':True}
    if defect=='label':arms['P'][1]['score']['reports']['Q']['metrics']['records'][0]['gold']=['z']
    if defect=='role':arms['D'][1]['score']['role']='test'
    if defect=='duplicate':
        rows=arms['D'][1]['score']['reports']['D']['metrics']['records'];rows.append(copy.deepcopy(rows[0]))
    if defect=='correctness':arms['D'][1]['score']['reports']['D']['metrics']['records'][0]['correct']=False
    if defect=='aggregate':arms['D'][1]['score']['reports']['D']['metrics']['final_accuracy']=.5
    if defect=='schedule':schedule=[['unknown']]
    with pytest.raises(ValueError):compare(arms,schedule)
