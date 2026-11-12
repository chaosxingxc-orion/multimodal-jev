import copy
from dataclasses import replace
import pytest
from multimodal_jev.intent_binding_training import expand
from multimodal_jev.slurp_training import content_hash
from test_span_candidate_intervention import request


def fixture():
    req=request()
    return dict(id='source',request=req,audio_sha256='audio',target=dict(id='source',targets=[3,4,5],
        target_values=['v3','v4','v5'],supervised=[True]*3,span_weights=[.1,.1,1.],request_weight=2.,
        source_group='group',decision_sha256='review',request_sha256='old'))


def test_preserves_semantics_and_total_credit_under_rebinding():
    item=fixture();original=copy.deepcopy(item['target']);views=expand(item)
    assert item['target']==original
    assert len(views)==4 and sum(t['request_weight'] for _,t in views)==2.
    for row,t in views:
        assert t['source_group']=='group' and t['decision_sha256']=='review'
        assert t['span_weights']==[.1,.1,1.] and t['request_sha256']==content_hash(row)
        assert 'target_values' not in row and 'targets' not in row
        for q,index,value in zip(row['request']['questions'],t['targets'],t['target_values'],strict=True):
            assert q['schema']['candidates'][index]['id']==value
    assert views[1][1]['targets']!=original['targets']


def test_gold_changes_neither_transformed_inputs_nor_bindings():
    item=fixture();changed=copy.deepcopy(item);changed['target']['targets']=[1,2,3]
    changed['target']['target_values']=['v1','v2','v3']
    assert [r for r,_ in expand(item)]==[r for r,_ in expand(changed)]


def test_direct_and_structured_final_mapping_matches():
    item=fixture();direct=copy.deepcopy(item);direct['request']=replace(item['request'],questions=(item['request'].questions[-1],))
    for key in ('targets','target_values','supervised','span_weights'):direct['target'][key]=direct['target'][key][-1:]
    for (a,ta),(b,tb) in zip(expand(item),expand(direct),strict=True):
        assert a['binding']['intent']==b['binding']['intent']
        assert ta['targets'][-1]==tb['targets'][-1]


@pytest.mark.parametrize('bad',['index','value','mask','weight','layout','variants'])
def test_rejects_invalid_supervision_or_partial_view_budget(bad):
    item=fixture()
    if bad=='index':item['target']['targets'][0]=True
    if bad=='value':item['target']['target_values'][0]='wrong'
    if bad=='mask':item['target']['supervised'][0]=False
    if bad=='weight':item['target']['request_weight']=float('nan')
    if bad=='layout':item['target']['span_weights'].pop()
    with pytest.raises(ValueError):expand(item,('native',) if bad=='variants' else ('native','shift1','shift30','permutation317'))
