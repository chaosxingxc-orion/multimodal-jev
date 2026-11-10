from copy import deepcopy
import hashlib
import pytest
from test_sfc_jev_preflight import requests
from multimodal_jev.sfc_partition import partition


def rows():
    result=requests()
    # Distinct synthetic recordings; retain shared stage-1 audio within a conversation.
    for row in result:
        row['audio_sha256']=[hashlib.sha256(p.encode()).hexdigest() for p in row['input']['context']['audios']]
    return result


def test_full_train_order_invariant_and_exposed_groups_in_learning():
    data=rows();exposed={data[0]['id']}
    a=partition(data,exposed,selection_fraction=.5)
    b=partition(data[::-1],exposed,selection_fraction=.5)
    assert a==b
    assert set(a['learning']).isdisjoint(a['selection'])
    assert set(a['learning'])|set(a['selection'])=={r['source_group'] for r in data}
    assert data[0]['source_group'] in a['forced_learning_groups']
    assert sum(v['bundles'] for v in a['summary'].values())==len(data)


def test_shared_waveform_components_never_cross_and_exposure_propagates():
    data=rows();data[1]['audio_sha256']=data[0]['audio_sha256'][:]
    report=partition(data,{data[0]['id']},selection_fraction=.5)
    assert data[0]['source_group'] in report['learning'] and data[1]['source_group'] in report['learning']
    assert len(report['multi_group_components'])==1


@pytest.mark.parametrize('change',['missing_stage','duplicate','test','unknown_exposed','prefix'])
def test_invalid_population_or_custody_rejected(change):
    data=rows();exposed=set()
    if change=='missing_stage':data=[r for r in data if r['source_group']!='level_3_1/train_id/id_1' or len(r['audio_sha256'])!=2]
    if change=='duplicate':data.append(deepcopy(data[0]))
    if change=='test':data[0]['split']='test_id'
    if change=='unknown_exposed':exposed={'missing'}
    if change=='prefix':
        next(r for r in data if len(r['audio_sha256'])==2)['audio_sha256'][0]='a'*64
    with pytest.raises(ValueError):partition(data,exposed,selection_fraction=.5)
