import pytest
import torch

from multimodal_jev.context_behavior_parity import compare,validate_bounds


def logp(z):
    z=torch.tensor(z,dtype=torch.float64)
    return torch.stack((torch.nn.functional.logsigmoid(-z),
                        torch.nn.functional.logsigmoid(z)),-1).flatten(1)


def test_complete_tuple_bound_and_exact_policy():
    a=logp([[1.,-1.,2.]])
    assert compare(a,a,dict(probability_max_abs=0.,binary_logprob_max_abs=0.))['passed']
    b=logp([[1.01,-1.02,2.03]])
    result=compare(a,b,dict(probability_max_abs=.1,binary_logprob_max_abs=.1))
    assert result['passed']
    import itertools
    x=(a-b).reshape(3,2)
    worst=max(abs(sum(float(x[i,bit]) for i,bit in enumerate(bits)))
              for bits in itertools.product((0,1),repeat=3))
    assert worst<=result['joint_logratio_abs_bound']
    assert not compare(a,b,dict(probability_max_abs=0.,binary_logprob_max_abs=0.))['passed']


def test_small_probability_error_cannot_hide_logprob_or_threshold_change():
    a,b=logp([[20.]]),logp([[21.]])
    report=compare(a,b,dict(probability_max_abs=1e-6,binary_logprob_max_abs=.01))
    assert report['probability_max_abs']<1e-6 and not report['passed']
    report=compare(logp([[1e-8]]),logp([[-1e-8]]),
                   dict(probability_max_abs=1.,binary_logprob_max_abs=1.))
    assert report['threshold_flips']==1 and not report['passed']


@pytest.mark.parametrize('bound',[{},dict(probability_max_abs=True,binary_logprob_max_abs=0),
    dict(probability_max_abs=float('nan'),binary_logprob_max_abs=0)])
def test_bounds_explicit(bound):
    with pytest.raises(ValueError):validate_bounds(bound)


def test_unnormalized_transport_rejected():
    with pytest.raises(ValueError):
        compare(torch.zeros(1,2),torch.zeros(1,2),dict(probability_max_abs=0,binary_logprob_max_abs=0))
