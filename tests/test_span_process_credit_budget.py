import math
import pytest
from multimodal_jev.span_process_credit_budget import measure


def logp(p):return list(map(math.log,p))


def test_small_process_reward_budget_can_dominate_low_final_credit():
    r=measure([logp([.5,.5]),logp([.5,.5]),logp([.999,.001])],[0,1,1],[.1,.1,1.])
    assert r['expected_process_reward']==pytest.approx(.1)
    assert r['expected_final_reward']==pytest.approx(.001)
    assert r['process_to_final_norm']>35 and r['process_exceeds_final']


def test_zero_process_credit_is_final_only():
    r=measure([logp([.5,.5]),logp([.2,.8])],[0,1],[0.,1.])
    assert r['process_norm']==0 and r['process_squared_norm_share']==0


def test_additive_policy_derivative_matches_finite_difference():
    logits=[[.2,-.3],[1.,0.,-1.]];targets=[1,2];weights=[.2,1.]
    def probabilities(z):
        total=sum(math.exp(x) for x in z);return [math.exp(x)/total for x in z]
    r=measure([logp(probabilities(z)) for z in logits],targets,weights)
    for j,z in enumerate(logits):
        derivatives=[]
        for i in range(len(z)):
            a=z.copy();b=z.copy();a[i]+=1e-5;b[i]-=1e-5
            derivatives.append(weights[j]*(probabilities(a)[targets[j]]-probabilities(b)[targets[j]])/2e-5)
        assert math.sqrt(sum(d*d for d in derivatives))==pytest.approx(r['span_norms'][j],rel=1e-8)


@pytest.mark.parametrize('weights',[[.1,-1],[.1,0],[float('nan'),1],[True,1],[.1]])
def test_invalid_budget_rejected(weights):
    with pytest.raises(ValueError):measure([logp([.5,.5])]*2,[0,1],weights)
