from copy import deepcopy
import pytest
from tools.reports.context_paired_decisions import summarize


def rows():
    return [dict(id='a', source_group='same', gold=0, initial=[.2,.8], selected=[.8,.2]),
            dict(id='b', source_group='same', gold=0, initial=[.8,.2], selected=[.2,.8]),
            dict(id='c', source_group='other', gold=0, initial=[.2,.8], selected=[.8,.2])]


def test_group_weighting_and_mixed_recordings_are_not_pooled():
    r = summarize(rows());m = r['metrics']['top1']
    assert r['source_groups'] == 2
    assert m['recording_weighted']['delta'] == pytest.approx(1/3)
    assert m['source_group_macro']['delta'] == .5
    assert m['groups_with_mixed_directions'] == 1
    assert m['transition_counts'] == dict(both_correct=0,both_wrong=0,repaired=2,regressed=1)


def test_set_f1_can_increase_while_exact_set_decreases():
    a = [dict(id='a',source_group='a',gold=0,initial=[.8,.2],selected=[.8,.8]),
         dict(id='b',source_group='b',gold=0,initial=[.2,.8],selected=[.8,.8])]
    r = summarize(a)['metrics']
    assert r['set_f1']['recording_weighted']['delta'] > 0
    assert r['exact_set']['recording_weighted']['delta'] < 0


@pytest.mark.parametrize('kind',['duplicate','missing_group','nan','width','gold','conflict'])
def test_invalid_pairing_rejected(kind):
    a = deepcopy(rows())
    if kind == 'duplicate':a.append(a[0])
    if kind == 'missing_group':a[0]['source_group']=''
    if kind == 'nan':a[0]['selected'][0]=float('nan')
    if kind == 'width':a[0]['selected'].pop()
    if kind == 'gold':a[0]['gold']=True
    if kind == 'conflict':a[1]['gold']=1
    with pytest.raises(ValueError):summarize(a)
