import copy
import pytest
from tools.reports.build_context_attribute_candidate import catalog,request,targets


def fixture():
    return dict(candidate_pairs={'a':['d1','x'],'b':['d1','y'],'c':['d2','x']},
        schema={'candidates':[{'id':k,'description':k+' event'} for k in ('a','b','c')]},
        thinking={'requirements':['Use the audio.']})


def test_same_menu_no_gold_routing_and_matched_final():
    menu=catalog(fixture())
    q=request('audio.wav',menu,arm='q_input')
    assert q.model_input()==request('audio.wav',menu,arm='q_credit').model_input()
    direct=request('audio.wav',menu,arm='direct')
    assert direct.questions==tuple(x for x in q.questions if x.id.startswith('final:'))
    assert direct.requirements==q.requirements
    before=q.model_input()
    for event in ('a','b','c'):targets(menu,event)
    assert request('audio.wav',menu,arm='q_input').model_input()==before


def test_exact_partition_targets_and_missing_not_false():
    menu=catalog(fixture())
    for event in ('a','b','c'):
        result=targets(menu,event)
        for family in ('scenario','action','final'):
            assert sum(y for n,y in zip(menu['nodes'],result['labels']) if n['family']==family)==1
    absent=targets(menu,None)
    assert all(x is None for x in absent['labels']) and not any(absent['supervised'])
    with pytest.raises(ValueError):targets(menu,'unknown')


@pytest.mark.parametrize('fault',['missing','duplicate_pair','duplicate_event'])
def test_invalid_catalog_rejected(fault):
    task=copy.deepcopy(fixture())
    if fault=='missing':task['candidate_pairs'].pop('a')
    if fault=='duplicate_pair':task['candidate_pairs']['b']=task['candidate_pairs']['a']
    if fault=='duplicate_event':task['schema']['candidates'].append(task['schema']['candidates'][0])
    with pytest.raises(ValueError):catalog(task)
