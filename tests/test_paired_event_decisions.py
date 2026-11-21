import math
import pytest
from multimodal_jev.paired_event_decisions import compare

INTERVAL=dict(method='paired_cluster_percentile_bootstrap',confidence=.95,samples=150,seed=317)


def fixture():
    targets=[dict(id='1',speaker='s1',gold='a'),dict(id='2',speaker='s2',gold='b')]
    left={'1':{'a':.8,'b':.2},'2':{'a':.2,'b':.8}}
    return targets,left


def test_semantic_ties_and_bad_probabilities():
    targets,left=fixture()
    right={'1':{'b':.5,'a':.5},'2':{'b':.5,'a':.5}}
    r=compare(['a','b'],targets,left,right,interval=INTERVAL,cluster='speaker')
    assert r['right']['top1_ties']==2 and r['right']['predicted_classes']==1
    assert r['regressed']==1 and r['fixed']==0 and r['right']['top1_macro_f1']==pytest.approx(1/3)
    assert r['right']['brier']==.25 and r['right']['nll']==pytest.approx(math.log(2))
    assert r['left']['brier']==pytest.approx(.04) and r['left']['nll']==pytest.approx(-math.log(.8))
    reverse={k:dict(reversed(list(row.items()))) for k,row in left.items()}
    r=compare(['a','b'],targets,left,reverse,interval=INTERVAL,cluster='speaker')
    assert r['macro_f1_delta_ci95']==[0.,0.] and r['mean_abs_candidate_probability_shift']==0
    wrong={'1':{'a':0.,'b':1.},'2':{'a':1.,'b':0.}}
    r=compare(['a','b'],targets,left,wrong,interval=INTERVAL,cluster='speaker')
    assert r['right']['nll'] is None and r['right']['brier']==1.


def test_collect_transitions_by_semantic_id():
    ids=['a','b']
    targets=[dict(id=str(i),speaker=f's{i}',gold='a') for i in range(4)]
    hit={'a':.8,'b':.2}
    miss={'b':.8,'a':.2}
    left=dict(zip(('0','1','2','3'),(hit,miss,hit,miss)))
    right=dict(zip(('0','1','2','3'),(hit,hit,miss,miss)))
    report=compare(ids,targets,left,right,interval=INTERVAL,cluster='speaker',collect_ids=True)
    assert report['recording_ids']==dict(both_correct=['0'],fixed=['1'],regressed=['2'],both_wrong=['3'])


@pytest.mark.parametrize('fault',['duplicate','miss','id','cluster','nan'])
def test_mismatched_populations_fail(fault):
    targets,left=fixture();right={k:v.copy() for k,v in left.items()}
    if fault=='duplicate':targets.append(targets[0])
    if fault=='miss':del right['1']
    if fault=='id':right['1']={'a':.8,'z':.2}
    if fault=='cluster':targets[0]['speaker']=targets[1]['speaker']
    if fault=='nan':right['1']['a']=float('nan')
    with pytest.raises(ValueError):compare(['a','b'],targets,left,right,interval=INTERVAL,cluster='speaker')
