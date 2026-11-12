import copy
import pytest
from multimodal_jev.intent_learning_trajectory import summarize


def fixture():
    return [dict(update=i, source_uses=x, metrics={'accuracy':a, 'nll':n})
            for i,(x,a,n) in enumerate([(0,.25,2.),(8,.5,1.),(32,.25,3.)])]


def test_exposure_weighted_area_preserves_late_regression():
    result=summarize(fixture(),registered_updates=2,directions={'accuracy':1,'nll':-1})
    assert result['complete'] and result['source_uses']==32
    a,n=result['metrics']['accuracy'],result['metrics']['nll']
    assert a['exposure_area']==12 and a['mean_directional_gain']==.125
    assert a['final_directional_gain']==0 and a['improving_transitions']==a['worsening_transitions']==1
    assert n['mean_over_observed_exposure']==1.875
    assert n['mean_directional_gain']==.125 and n['final_directional_gain']==-1
    assert n['checkpoints_below_initial']==1 and n['worst_directional_gain']==-1


def test_partial_and_initial_only_are_not_complete():
    result=summarize(fixture(),registered_updates=12,directions={'accuracy':1,'nll':-1})
    assert not result['complete'] and result['observed_updates']==2
    result=summarize(fixture()[:1],registered_updates=12,directions={'accuracy':1,'nll':-1})
    assert result['metrics']['accuracy']['mean_directional_gain'] is None
    assert result['metrics']['accuracy']['exposure_area']==0


def test_plateau_is_not_an_improvement():
    rows=fixture()
    for row in rows:row['metrics']={'accuracy':.25,'nll':2.}
    result=summarize(rows,registered_updates=2,directions={'accuracy':1,'nll':-1})
    for m in result['metrics'].values():
        assert m['unchanged_transitions']==2 and m['improving_transitions']==0
        assert m['mean_directional_gain']==0


@pytest.mark.parametrize('defect',['gap','duplicate','exposure','zero_start','nan','bool','missing_metric','budget','direction'])
def test_invalid_or_selected_curve_rejected(defect):
    rows=copy.deepcopy(fixture());budget=2;directions={'accuracy':1,'nll':-1}
    if defect=='gap':rows.pop(1)
    if defect=='duplicate':rows[2]['update']=1
    if defect=='exposure':rows[2]['source_uses']=8
    if defect=='zero_start':rows[0]['source_uses']=1
    if defect=='nan':rows[1]['metrics']['accuracy']=float('nan')
    if defect=='bool':rows[1]['update']=True
    if defect=='missing_metric':del rows[1]['metrics']['nll']
    if defect=='budget':budget=1
    if defect=='direction':directions['nll']=0
    with pytest.raises(ValueError):summarize(rows,registered_updates=budget,directions=directions)
