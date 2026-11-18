import itertools
import math
import pytest
from tools.reports.analyze_context_action_diversity import distribution


@pytest.mark.parametrize('probs',[[.1,.7,.4],[0.,1.],[.5,.5]])
def test_collision_and_hamming_against_action_enumeration(probs):
    atoms=[]
    for action in itertools.product((0,1),repeat=len(probs)):
        mass=math.prod(p if x else 1-p for p,x in zip(probs,action));atoms.append((action,mass))
    out=distribution(probs,3)
    assert out['all_group_actions_identical_probability']==pytest.approx(sum(m**3 for _,m in atoms))
    assert out['expected_pairwise_hamming_distance']==pytest.approx(sum(m*n*sum(x!=y for x,y in zip(a,b)) for a,m in atoms for b,n in atoms))
    assert out['joint_entropy_nats']==pytest.approx(-sum(m*math.log(m) for _,m in atoms if m))


@pytest.mark.parametrize('probs',[[],[float('nan')],[-.1],[1.1]])
def test_invalid_distribution_rejected(probs):
    with pytest.raises(ValueError):distribution(probs)
