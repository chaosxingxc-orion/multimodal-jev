import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from prepare_intent_matched_arms import audit_arms
from test_slurp_intent_structure import fixture
from multimodal_jev.slurp_intent_structure import training_targets


def populations():
    task,example=fixture();result={}
    for arm in ('D','Q','P'):
        request,target=training_targets(example,task,arm);target['request_weight']=1.
        result[arm]=[dict(id=str(i),request=request,target=copy.deepcopy(target)) for i in range(9)]
    return result


def test_same_schedule_and_reward_budget():
    result=audit_arms(populations(),317,12)
    assert len(result['schedule'])==12 and result['span_weights']['P']==[.1,.1,1.]


@pytest.mark.parametrize('bad',['labels','weights','credit','population'])
def test_rejects_confounders(bad):
    p=populations()
    if bad=='labels':p['P'][0]['target']['targets'][-1]=0
    if bad=='weights':p['Q'][0]['target']['request_weight']=2.
    if bad=='credit':p['P'][0]['target']['span_weights']=[1.,1.,1.]
    if bad=='population':p['Q'].pop()
    with pytest.raises(ValueError):audit_arms(p,317,12)
