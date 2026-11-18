import itertools
import math
import pytest
from tools.reports.exact_singleton_f1_gradient import expected_f1_gradient


@pytest.mark.parametrize('p,gold',[([.2,.7,.4],1),([.9,.1,.8,.01],0),([.4],0),([1.,0.,1.],0)])
def test_reverse_recurrence_equals_exhaustive_score_gradient(p,gold):
    value=0.;gradient=[0.]*len(p)
    for action in itertools.product((0,1),repeat=len(p)):
        mass=math.prod(v if a else 1-v for a,v in zip(action,p))
        reward=2*action[gold]/(sum(action)+1)
        value+=mass*reward
        for i in range(len(p)):gradient[i]+=mass*reward*(action[i]-p[i])
    result=expected_f1_gradient(p,gold)
    assert result['expected_f1']==pytest.approx(value,abs=1e-14)
    assert result['logit_gradient']==pytest.approx(gradient,abs=1e-14)


def test_60_candidate_gradients_match_current_logit_finite_differences():
    logits=[(i-30)/7 for i in range(60)];gold=17
    sigmoid=lambda xs:[1/(1+math.exp(-x)) for x in xs]
    result=expected_f1_gradient(sigmoid(logits),gold)
    for i in range(60):
        hi=logits.copy();lo=logits.copy();hi[i]+=1e-4;lo[i]-=1e-4
        numerical=(expected_f1_gradient(sigmoid(hi),gold)['expected_f1']-
                   expected_f1_gradient(sigmoid(lo),gold)['expected_f1'])/2e-4
        assert result['logit_gradient'][i]==pytest.approx(numerical,rel=1e-6,abs=1e-11)
    assert result['logit_gradient'][gold]>0
    assert all(v<0 for i,v in enumerate(result['logit_gradient']) if i!=gold)


def test_false_gold_has_zero_negative_gradient_but_nonzero_probability_derivative():
    r=expected_f1_gradient([0.,.7,.3],0)
    assert r['expected_f1']==0
    assert r['logit_gradient']==[0.,0.,0.]
    assert r['probability_gradient'][0]>0


@pytest.mark.parametrize('p,gold',[([],0),([.3],True),([float('nan')],0),([1.01],0),([.3],1)])
def test_reject_invalid_input(p,gold):
    with pytest.raises(ValueError):expected_f1_gradient(p,gold)
