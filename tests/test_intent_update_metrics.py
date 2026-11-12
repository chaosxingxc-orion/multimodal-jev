import copy
import math
import pytest
from multimodal_jev.intent_update_metrics import margins,summarize,training_metrics


def lp(p):return list(map(math.log,p))


def fixture():
    return [dict(id='a',weight=3.,target=0,old=lp([.1,.8,.1]),new=lp([.7,.2,.1])),
            dict(id='b',weight=1.,target=1,old=lp([.2,.7,.1]),new=lp([.6,.3,.1]))]


def test_margin_measures_gold_not_prediction_confidence():
    m=margins(lp([.1,.8,.1]),0)
    assert m['gold_log_margin']==pytest.approx(math.log(.1/.8))
    assert m['gold_probability_margin']==pytest.approx(-.7)
    assert margins(lp([.5,.5]),1)['gold_log_margin']==0


def test_updated_repair_and_other_regression_are_not_hidden_by_net_accuracy():
    r=summarize(fixture(),['a'])['groups']
    all_=r['all_learning']['metrics']
    assert all_['before']['correct']==all_['after']['correct']==1
    assert all_['wrong_to_right']==all_['right_to_wrong']==1
    assert all_['weighted_drift']['gold_probability_delta']==pytest.approx(.35)
    assert all_['mean_drift']['gold_probability_delta']==pytest.approx(.1)
    assert r['updated_this_step']['metrics']['after']['gold_log_margin']>0
    assert r['other_learning']['metrics']['after']['gold_log_margin']<0
    assert all_['mean_drift']['old_mass_outside_clip']>0


def test_zero_change_and_empty_complement():
    rows=fixture()
    for r in rows:r['new']=r['old'][:]
    result=summarize(rows,['a','b'])
    assert result['groups']['other_learning']==dict(recordings=0,metrics=None)
    assert result['groups']['all_learning']['metrics']['max_kl_old_to_new']==0


@pytest.mark.parametrize('bad',['duplicate','unknown','weight','support','target','normalization'])
def test_rejects_invalid_population(bad):
    rows=fixture();selected=['a']
    if bad=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif bad=='unknown':selected=['c']
    elif bad=='weight':rows[0]['weight']=float('nan')
    elif bad=='support':rows[0]['new']=lp([.5,.5])
    elif bad=='target':rows[0]['target']=True
    else:rows[0]['new']=[0.,0.,0.]
    with pytest.raises(ValueError):summarize(rows,selected)


def ranks():
    return [dict(rank=i,updates=8,adapter_id='p8',metrics={
        'actor/grad_norm':[1.5],'actor/pg_clipfrac':[0.,0.]}) for i in range(4)]


def test_gradient_clipping_is_separate_from_policy_ratio_clipping():
    r=training_metrics(ranks())
    assert r['gradient_clip_triggered'] and r['global_grad_norm_before_clip']==1.5
    assert all(v['preupdate_microbatch_clipfractions']==[0.,0.] for v in r['ranks'])


@pytest.mark.parametrize('bad',['missing','cursor','policy','norm','microbatch','updates'])
def test_rejects_inconsistent_training_traces(bad):
    r=ranks()
    if bad=='missing':r.pop()
    elif bad=='cursor':r[0]['updates']=7
    elif bad=='policy':r[0]['adapter_id']='other'
    elif bad=='norm':r[0]['metrics']['actor/grad_norm']=[.5]
    elif bad=='microbatch':r[0]['metrics']['actor/pg_clipfrac']=[float('nan')]
    else:r[0]['metrics']['actor/grad_norm']=[1.,2.]
    with pytest.raises(ValueError):training_metrics(r)
