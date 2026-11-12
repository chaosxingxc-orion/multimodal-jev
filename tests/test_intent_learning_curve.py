import copy
import pytest
from multimodal_jev.intent_learning_curve import summarize


def fixture():
    rows=[dict(id='a',gold=['x'],correct=False,gold_probability=[.1]),dict(id='b',gold=['y'],correct=True,gold_probability=[.8])]
    score=dict(role='learning',base_id='base',adapter_id='initial',reports={a:dict(admission={'fixed':True},metrics=dict(records=copy.deepcopy(rows),final_macro_f1_observed=.5,final_predicted_classes=1,final_dominant_prediction_fraction=1.)) for a in ('D','Q')})
    points=[dict(update=0,score=score),dict(update=1,score=copy.deepcopy(score)),dict(update=2,score=copy.deepcopy(score))]
    points[1]['score']['adapter_id']='one';points[2]['score']['adapter_id']='two'
    return points,[['a'],['b']]


def test_tracks_cumulative_coverage_not_only_current_batch():
    points,schedule=fixture();result=summarize(points,schedule,2)
    assert result['complete'] and result['curve'][-1]['distinct_updated_sources']==2
    assert result['curve'][0]['arms']['D']['groups']['ever_updated']['final_accuracy'] is None
    assert result['curve'][1]['arms']['D']['groups']['not_yet_updated']['recordings']==1
    assert result['curve'][2]['arms']['D']['groups']['not_yet_updated']['recordings']==0
    assert not summarize(points[:2],schedule,2)['complete']


@pytest.mark.parametrize('bad',['gap','labels','population','schedule','base'])
def test_rejects_curve_splicing(bad):
    p,s=fixture()
    if bad=='gap':p.pop(1)
    if bad=='labels':p[2]['score']['reports']['Q']['metrics']['records'][0]['gold']=['other']
    if bad=='population':p[2]['score']['reports']['D']['metrics']['records'].pop()
    if bad=='schedule':s[1]=['unknown']
    if bad=='base':p[1]['score']['base_id']='different'
    with pytest.raises(ValueError):summarize(p,s,2)
