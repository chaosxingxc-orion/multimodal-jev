import copy
import pytest
from multimodal_jev.intent_composition_metrics import compare

PAIRS={'a_x':['a','x'],'a_y':['a','y'],'b_x':['b','x'],'b_y':['b','y'],'c_z':['c','z']}
def row(uid,gold,pred,prob=.1):
    return dict(id=uid,gold=PAIRS[gold]+[gold],prediction=PAIRS[pred]+[pred],gold_probability=[prob]*3,correct=gold==pred)

def test_complete_slices_and_empty_conditional_population():
    before=[row('1','a_x','b_y'),row('2','c_z','a_y')]
    after=[row('1','a_x','a_x'),row('2','c_z','a_y',.2)]
    r=compare(before,after,PAIRS)
    assert r['joint_components']['repaired']==['1']
    assert r['single_component']['probability_improved']==1
    assert r['all']['before']['final_correct_given_process_both_correct'] is None
    assert r['all']['after']['compatible_but_final_wrong']==1
    assert r['all']['after']['final_correct_given_process_both_correct']==1

@pytest.mark.parametrize('bad',['duplicate','missing','gold','correct','probability','projection'])
def test_invalid_pair_is_rejected(bad):
    a=[row('1','a_x','b_y'),row('2','c_z','a_y')];b=copy.deepcopy(a)
    if bad=='duplicate':b.append(b[0])
    elif bad=='missing':b.pop()
    elif bad=='gold':b[0]['gold']=PAIRS['a_y']+['a_y']
    elif bad=='correct':b[0]['correct']=True
    elif bad=='probability':b[0]['gold_probability'][0]=float('nan')
    else:b[0]['gold'][0]='c'
    with pytest.raises(ValueError):compare(a,b,PAIRS)

def test_direct_predictions_and_empty_slice():
    a=[dict(id='1',gold=['c_z'],prediction=['c_z'],gold_probability=[.8],correct=True)]
    r=compare(a,a,PAIRS)
    assert r['joint_components']['after']['final_accuracy'] is None
    assert 'process_both_correct' not in r['all']['after']


def test_mixed_template_population_is_rejected():
    a=[row('1','a_x','b_y'),dict(id='2',gold=['c_z'],prediction=['c_z'],gold_probability=[.8],correct=True)]
    with pytest.raises(ValueError,match='fixed template'):compare(a,a,PAIRS)
