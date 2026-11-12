import math
import pytest
from multimodal_jev.span_credit_scaling import bounded_scales
from multimodal_jev.span_credit_concentration import sensitivity


def test_bounded_normalized_and_common_scaling_not_amplification():
    result=bounded_scales([.000001,.001,.2],[60]*3,[1.,2.,1.])
    assert result['raw_scales']==[60.,60.,5.]
    assert sum(w*c for w,c in zip([1,2,1],result['scales']))==pytest.approx(4.)
    assert bounded_scales([.000001,.001],[60,60],[1,2])['scales']==[1.,1.]


def test_frozen_weight_matches_policy_surrogate_finite_difference():
    logits=[-7.,-1.,2.];y=0
    def probs(z):
        total=sum(math.exp(v) for v in z);return [math.exp(v)/total for v in z]
    old=probs(logits);scale=bounded_scales([old[y],.9],[3,3],[1.,1.])['scales'][0]
    info=sensitivity([math.log(p) for p in old],y,scale)
    advantage=[float(i==y)-old[y] for i in range(3)]
    def surrogate(z):return scale*sum(p*a for p,a in zip(probs(z),advantage))
    for i in range(3):
        a=logits.copy();b=logits.copy();a[i]+=1e-5;b[i]-=1e-5
        assert info['gradient'][i]==pytest.approx((surrogate(a)-surrogate(b))/2e-5,abs=1e-12)


@pytest.mark.parametrize('prob,k,w',[([float('nan')],[2],[1]),([.1],[1],[1]),([.1],[2],[0]),([.1],[2],[-1]),([1.1],[2],[1])])
def test_invalid_inputs_rejected(prob,k,w):
    with pytest.raises(ValueError):bounded_scales(prob,k,w)
