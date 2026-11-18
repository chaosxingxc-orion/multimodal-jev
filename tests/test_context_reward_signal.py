import itertools,math,sys
from collections import defaultdict
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from context_reward_signal import singleton_signal


@pytest.mark.parametrize('probs,gold',[([.2,.7,.4],1),([0.,1.,1.],0),([1.,0.,0.],0),([.6],0)])
def test_exact_distribution_matches_exhaustive_actions(probs,gold):
    distribution=defaultdict(float);empty=exact=multi=0.
    for action in itertools.product((0,1),repeat=len(probs)):
        mass=math.prod(p if a else 1-p for p,a in zip(probs,action))
        reward=2*action[gold]/(sum(action)+1);distribution[reward]+=mass
        empty+=mass*(sum(action)==0);multi+=mass*(sum(action)>1)
        exact+=mass*(action[gold]==1 and sum(action)==1)
    got=singleton_signal(probs,gold,group_size=4)
    mean=sum(r*p for r,p in distribution.items())
    assert got['expected_set_f1']==pytest.approx(mean)
    assert got['reward_variance']==pytest.approx(sum(p*(r-mean)**2 for r,p in distribution.items()))
    assert got['group_all_equal_reward_probability']==pytest.approx(sum(p**4 for p in distribution.values()))
    assert got['empty_set_probability']==pytest.approx(empty)
    assert got['exact_correct_set_probability']==pytest.approx(exact)
    assert got['multiple_selection_probability']==pytest.approx(multi)


def test_confident_empty_and_half_probabilities_have_different_failure_modes():
    empty=singleton_signal([.00001]*60,0)
    diffuse=singleton_signal([.5]*60,0)
    assert empty['group_all_equal_reward_probability']>.999
    assert diffuse['group_all_equal_reward_probability']<.0001
    assert diffuse['expected_selected_count']==30
    assert diffuse['exact_correct_set_probability']<1e-16


@pytest.mark.parametrize('probabilities,gold',[([float('nan')],0),([1.1],0),([.5],True),([.5],1)])
def test_invalid_distribution_rejected(probabilities,gold):
    with pytest.raises(ValueError):singleton_signal(probabilities,gold)
