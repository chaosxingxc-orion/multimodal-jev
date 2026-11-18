import copy
import pytest
from tools.reports.audit_context_attribute_population import validate_sources,validate_pair


def pair():
    return [dict(arm=arm,tokens=400,decisions=n,positions=list(range(200,200+n)),audio_tokens=10,
                 tensors={'input_features':{'sha256':'a'},'input_features_mask':{'sha256':'b'}})
            for arm,n in [('direct',60),('q_input',124)]]


def test_complete_pair_and_population():
    validate_pair(pair())
    x=[dict(id='a',audio='/a',audio_sha256='s',catalog_sha256='c')]
    assert validate_sources(x,[dict(id='a',audio_sha256='s')],1)['a']['id']=='a'


@pytest.mark.parametrize('fault',['missing_arm','missing_marker','out_of_bounds','feature','audio_count'])
def test_incomplete_or_changed_input_rejected(fault):
    records=pair()
    if fault=='missing_arm':records.pop()
    if fault=='missing_marker':records[1]['positions'].pop()
    if fault=='out_of_bounds':records[1]['tokens']=5000
    if fault=='feature':records[1]['tensors']['input_features']['sha256']='different'
    if fault=='audio_count':records[1]['audio_tokens']=11
    with pytest.raises(ValueError):validate_pair(records)


@pytest.mark.parametrize('fault',['duplicate','labels','audio_identity'])
def test_population_errors_rejected(fault):
    x=[dict(id='a',audio='/a',audio_sha256='s',catalog_sha256='c')];a=[dict(id='a',audio_sha256='s')]
    if fault=='duplicate':x*=2;a*=2
    if fault=='labels':x[0]['label']=1
    if fault=='audio_identity':a[0]['audio_sha256']='different'
    with pytest.raises(ValueError):validate_sources(x,a,len(x))
