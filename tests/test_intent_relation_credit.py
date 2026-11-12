import itertools
import math
import pytest
from multimodal_jev.intent_relation_credit import credit


def lp(p):return [math.log(x) for x in p]


def test_sparse_formula_equals_full_joint_enumeration():
    ps=[[.3,.7],[.2,.3,.5],[.1,.2,.3,.4]];pairs=[(0,0),(0,2),(1,1),(1,2)];w=.1
    r=credit([lp(p) for p in ps],pairs,w)
    expectation=0.;conditional=[[0.]*len(p) for p in ps]
    for s,a,f in itertools.product(*[range(len(p)) for p in ps]):
        action=(s,a,f);reward=-w*int((s,a)!=pairs[f]);prob=math.prod(ps[j][v] for j,v in enumerate(action))
        expectation+=prob*reward
        for j,v in enumerate(action):conditional[j][v]+=prob/ps[j][v]*reward
    assert r['expected_penalty']==pytest.approx(expectation)
    for j in range(3):
        assert r['advantages'][j]==pytest.approx([v-expectation for v in conditional[j]])
        assert sum(p*a for p,a in zip(ps[j],r['advantages'][j]))==pytest.approx(0.,abs=1e-15)


def test_logit_gradient_matches_finite_difference():
    logits=[[.1,-.8],[.7,-.5,1.],[.8,-.2,.3,1.]];pairs=[(0,0),(0,2),(1,1),(1,2)]
    def norm(z):
        total=sum(math.exp(x) for x in z);return [x-math.log(total) for x in z]
    def run(z):return credit([norm(x) for x in z],pairs)
    r=run(logits)
    for j,z in enumerate(logits):
        for k in range(len(z)):
            plus=[x[:] for x in logits];minus=[x[:] for x in logits];plus[j][k]+=1e-5;minus[j][k]-=1e-5
            numeric=(run(plus)['expected_penalty']-run(minus)['expected_penalty'])/2e-5
            assert r['logit_gradients'][j][k]==pytest.approx(numeric,abs=1e-11)


def test_consistent_wrong_gets_as_little_penalty_as_consistent_correct():
    # Gold is triple (0,0,0). Relationship reward cannot see that fact.
    good=[[.999,.001]]*3;wrong=[[.001,.999]]*3;pairs=[(0,0),(1,1)]
    a=credit([lp(x) for x in good],pairs);b=credit([lp(x) for x in wrong],pairs)
    assert a['expected_penalty']==pytest.approx(b['expected_penalty'])
    assert b['logit_gradients'][-1][1]>0  # Reinforces the wrong Final.


def test_zero_weight_has_no_credit_and_permutations_preserve_semantics():
    ps=[[.3,.7],[.2,.8],[.4,.6]];pairs=[(0,0),(1,1)]
    assert all(g==0 for gs in credit([lp(x) for x in ps],pairs,0)['logit_gradients'] for g in gs)
    a=credit([lp(x) for x in ps],pairs)
    b=credit([lp(ps[0][::-1]),lp(ps[1]),lp(ps[2][::-1])],[(0,1),(1,0)])
    assert a['expected_penalty']==pytest.approx(b['expected_penalty'])
    assert a['advantages'][0]==pytest.approx(b['advantages'][0][::-1])
    assert a['advantages'][2]==pytest.approx(b['advantages'][2][::-1])


@pytest.mark.parametrize('pairs,weight',[([(0,0),(0,0)],.1),([(0,0),(2,1)],.1),([(0,0)],.1),([(0,0),(1,1)],-1),([(0,0),(1,1)],1),([(0,0),(1,1)],float('nan'))])
def test_invalid_factor_rejected(pairs,weight):
    with pytest.raises(ValueError):credit([lp([.5,.5])]*3,pairs,weight)
