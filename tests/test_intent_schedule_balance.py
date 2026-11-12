import pytest
from multimodal_jev.intent_schedule_balance import audit


def sources():
    return [dict(id='a',label='rare',weight=1.5),dict(id='b',label='common',weight=.75),
            dict(id='c',label='common',weight=.75)]


def test_full_population_inverse_weights_balance_classes():
    r=audit(sources(),[['a','b','c']])
    assert r['scheduled_class_coefficient_range']==[.5,.5]
    assert r['total_variation_from_uniform']==0


def test_singleton_self_normalization_cancels_weights():
    r=audit(sources(),[['a'],['b'],['c']])
    assert r['scheduled_class_coefficient_range']==pytest.approx([1/3,2/3])
    assert r['total_variation_from_uniform']==pytest.approx(1/6)
    assert r['pooled_weight_total_variation_from_uniform']==0


def test_repeat_sources_and_missing_classes_remain_visible():
    r=audit(sources(),[['a'],['a']])
    assert r['source_use_histogram']=={0:2,2:1}
    assert r['scheduled_max_to_min_ratio'] is None
    assert r['total_variation_from_uniform']==.5


def test_batch_partition_changes_objective_coefficients_without_changing_exposures():
    a=audit(sources(),[['a','b'],['c']]);b=audit(sources(),[['a','b','c']])
    assert a['source_use_histogram']==b['source_use_histogram']=={1:3}
    assert a['source_uses']==b['source_uses']==3
    assert a['total_variation_from_uniform']==pytest.approx(1/6)
    assert b['total_variation_from_uniform']==0


@pytest.mark.parametrize('fault',['duplicate','unknown','duplicate_batch','weight','inverse','empty'])
def test_rejects_invalid_contract(fault):
    rows=sources();batches=[['a','b','c']]
    if fault=='duplicate':rows.append(rows[0])
    elif fault=='unknown':batches=[['z']]
    elif fault=='duplicate_batch':batches=[['a','a']]
    elif fault=='weight':rows[0]['weight']=float('nan')
    elif fault=='inverse':rows[0]['weight']=2
    else:batches=[]
    with pytest.raises(ValueError):audit(rows,batches)
