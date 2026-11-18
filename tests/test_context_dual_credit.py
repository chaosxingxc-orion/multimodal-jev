import copy
import itertools

import pytest
import torch

from multimodal_jev.context_dual_credit import (
    ALGORITHMS, advantages, components, current_terms, log_probabilities, prepare, verify)


IDENTITY = dict(policy_id='policy-0',input_id='input-0',query_id='query-0',update=0)


def record(z=None, **kw):
    if z is None: z=torch.tensor([.7,-.4,1.2])
    return prepare(z,z.sigmoid(),['c','a','b'],torch.tensor([True,False,False]),**IDENTITY,**kw)


def test_canonical_pairs_and_balanced_weights():
    cs=components(torch.tensor([.7,-.4,1.2]),['c','a','b'],torch.tensor([True,False,False]))
    assert cs['ranking'].keys == (('a','c'),('b','c'))
    assert cs['ranking'].target.tolist() == [False,False]
    assert cs['existence'].keys == (('a',),('b',),('c',))
    assert cs['existence'].weight.tolist() == [.25,.25,.5]


def test_permutation_preserves_probabilities_samples_and_advantages():
    a=record();idx=torch.tensor([2,0,1]);z=torch.tensor([.7,-.4,1.2])[idx]
    b=prepare(z,z.sigmoid(),['b','c','a'],torch.tensor([False,True,False]),**IDENTITY)
    for branch in a['branches']:
        for key in ('probability','actions','old_logprob','advantages','weight'):
            assert torch.equal(a['branches'][branch][key],b['branches'][branch][key])


@pytest.mark.parametrize('algorithm',ALGORITHMS)
def test_advantage_formula_independent(algorithm):
    r=torch.tensor([[0.,1.,1.,0.],[1.,1.,1.,1.]])
    a=advantages(r,algorithm)
    if algorithm==ALGORITHMS[0]:
        expected=torch.tensor([[r[i,j]-sum(r[i,k] for k in range(4) if k!=j)/3
                                for j in range(4)] for i in range(2)])
    else:
        expected=torch.tensor([[-1.,1.,1.,-1.],[0.,0.,0.,0.]])
    torch.testing.assert_close(a,expected)


def test_grpo_floor_limits_small_variance():
    r=torch.zeros(1,64);r[0,0]=1
    torch.testing.assert_close(advantages(r,ALGORITHMS[1]),(r-r.mean())/.25)


def test_rloo_enumeration_matches_expected_binary_reward_gradient():
    # Independent enumeration of all G2 outcomes proves the on-policy score
    # gradient; no native verl, model or NPU is simulated by this test.
    z=torch.tensor(.4,requires_grad=True);p=z.sigmoid()
    expected_loss=z*0
    for actions in itertools.product((0,1),repeat=2):
        a=torch.tensor([actions],dtype=torch.long)
        r=a.float();adv=advantages(r,ALGORITHMS[0])
        logs=log_probabilities(p.reshape(1)).gather(1,a)
        mass=torch.prod(torch.where(a.bool(),p.detach(),1-p.detach()))
        expected_loss=expected_loss-mass*(logs*adv).mean()
    grad,=torch.autograd.grad(expected_loss,z)
    torch.testing.assert_close(grad,-p.detach()*(1-p.detach()))


def test_shift_affects_existence_not_rank_and_gradient_connects():
    z=torch.tensor([.7,-.4,1.2],requires_grad=True);y=torch.tensor([True,False,False])
    a=components(z,['c','a','b'],y);b=components(z+1,['c','a','b'],y)
    torch.testing.assert_close(a['ranking'].probability,b['ranking'].probability)
    assert not torch.equal(a['existence'].probability,b['existence'].probability)
    g,=torch.autograd.grad(a['ranking'].probability.sum(),z)
    assert abs(float(g.sum()))<1e-7 and bool((g!=0).all())


@pytest.mark.parametrize('field',['actions','rewards','advantages','old_logprob','weight','probability'])
def test_modified_receipt_rejected(field):
    a=record();v=a['branches']['existence'][field]
    v.reshape(-1)[0]+=1
    with pytest.raises(ValueError):verify(a)


def test_stale_identity_and_numerical_mismatch_rejected():
    a=record();z=torch.tensor([.7,-.4,1.2],requires_grad=True)
    with pytest.raises(ValueError):current_terms(z,a,**dict(IDENTITY,update=1))
    with pytest.raises(ValueError):current_terms(z+.1,a,**IDENTITY)
    terms=current_terms(z,a,**IDENTITY)
    assert terms['ranking']['new_logprob'].requires_grad
    assert torch.equal(terms['existence']['new_logprob'].detach(),terms['existence']['old_logprob'])


@pytest.mark.parametrize('truth',[[True,True],[False,False]])
def test_empty_rank_full_weight_existing_side(truth):
    z=torch.tensor([.1,-.1]);a=prepare(z,z.sigmoid(),['a','b'],torch.tensor(truth),**IDENTITY)
    verify(a)
    assert a['branches']['ranking']['actions'].shape==(0,16)
    assert a['branches']['existence']['weight'].tolist()==[.5,.5]


def test_fp32_saturated_pair_has_finite_fp64_logs_and_connected_gradient():
    z=torch.tensor([10.,-10.],requires_grad=True)
    assert (z[0]-z[1]).sigmoid().item()==1.
    a=prepare(z.detach(),z.detach().sigmoid(),['a','b'],torch.tensor([True,False]),**IDENTITY)
    verify(a);terms=current_terms(z,a,**IDENTITY)
    q=a['branches']['ranking']['probability']
    assert q.dtype==torch.float64 and 0<q.item()<1
    assert torch.isfinite(terms['ranking']['new_logprob']).all()
    gradient,=torch.autograd.grad(terms['ranking']['new_logprob'].sum(),z)
    assert torch.isfinite(gradient).all() and (gradient!=0).all()



def test_detached_and_complete_truth_required():
    with pytest.raises(ValueError):record(torch.tensor([.7,-.4,1.2],requires_grad=True))
    with pytest.raises(ValueError):components(torch.zeros(2),['a','a'],torch.tensor([True,False]))


def test_zero_signal_is_retained_not_filtered():
    a=record(group_size=2)
    assert a['branches']['existence']['actions'].shape==(3,2)
    rewards=torch.ones((3,2))
    assert not bool(advantages(rewards,ALGORITHMS[0]).any())
