from copy import deepcopy
import pytest
from multimodal_jev.span_matched_budget import compare_training_exposure


def fixture():
    branches={}
    for method in ('sft','rl','conditional'):
        steps=[]
        for i in range(2):
            steps.append(dict(config=dict(method=method,batch_index=i,request_ids=[f'{i}:a',f'{i}:b'],seed=17,arm='structured',auxiliary_weight=.2,lr=1e-4,samples_per_request=32),
                report=dict(method=method,completed_updates=i+1,optimizer_updates=1,training_requests=2,optimizer_start='fresh' if i==0 else 'resume',input_tokens=100,audio_seconds_with_repetition=2.5,unique_recordings=2)))
        branches[method]=dict(study_sha256='study',initial_policy_id='base:initial',steps=steps,wall_seconds_including_gates=120)
    return branches


def test_matched_exposure_does_not_conflate_compute_cost():
    b=fixture();b['rl']['wall_seconds_including_gates']=200
    r=compare_training_exposure(b)
    assert r['passed'] and r['budgets']['rl']['requests']==4
    assert r['budgets']['rl']['policy_gradient_samples_used']==128
    assert r['budgets']['conditional']['policy_gradient_samples_used']==0
    assert r['budgets']['conditional']['exact_support_integration']


@pytest.mark.parametrize('fault',['initializer','study','order','seed','lr','weight','skip','fresh','audio','nan','method','missing'])
def test_reject_unmatched_trajectories(fault):
    b=fixture();x=b['conditional'];s=x['steps'][1]
    if fault=='initializer':x['initial_policy_id']='other'
    elif fault=='study':x['study_sha256']='other'
    elif fault=='order':s['config']['request_ids'].reverse()
    elif fault=='seed':s['config']['seed']=29
    elif fault=='lr':s['config']['lr']=.01
    elif fault=='weight':s['config']['auxiliary_weight']=0
    elif fault=='skip':s['report']['completed_updates']=3
    elif fault=='fresh':s['report']['optimizer_start']='fresh'
    elif fault=='audio':s['report']['audio_seconds_with_repetition']=3
    elif fault=='nan':s['report']['audio_seconds_with_repetition']=float('nan')
    elif fault=='method':s['config']['method']='sft'
    else:del b['rl']
    with pytest.raises(ValueError):compare_training_exposure(b)
