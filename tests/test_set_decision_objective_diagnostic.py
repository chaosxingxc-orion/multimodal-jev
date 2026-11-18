import importlib.util
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')

spec=importlib.util.spec_from_file_location('set_diagnostic',Path(__file__).parents[1]/'tools/jev/analyze_set_decision_objectives.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_equal_valid_mass_has_different_ranking_and_uniform_target_objectives():
    a=m.evaluate([.45,.45,.1],[True,True,False]);b=m.evaluate([.89,.01,.1],[True,True,False])
    for name in ['set_mass_nll','negative_expected_membership_reward']:
        assert a['losses'][name]==pytest.approx(b['losses'][name])
    for name in ['uniform_valid_ce','pairwise_valid_invalid']:
        assert b['losses'][name]>a['losses'][name]
    # Set mass can accept suppression of one valid candidate; do not use it to
    # assert the model has recognized every applicable route.
    assert b['logit_gradients']['set_mass_nll'][1]<0
    assert abs(b['logit_gradients']['set_mass_nll'][1])<abs(b['logit_gradients']['pairwise_valid_invalid'][1])


def test_exact_reward_gradient_is_valid_mass_times_set_nll_gradient():
    for p,v in [([.89,.01,.1],[True,True,False]),([.05,.05,.9],[True,True,False])]:
        r=m.evaluate(p,v)
        torch.testing.assert_close(torch.tensor(r['logit_gradients']['negative_expected_membership_reward']),
            torch.tensor(r['logit_gradients']['set_mass_nll'])*r['acceptable_mass'])


def test_candidate_permutation_changes_neither_loss_nor_semantic_gradient():
    x=torch.tensor([1.2,-.3,.7,2.],dtype=torch.float64,requires_grad=True)
    valid=torch.tensor([True,False,True,False]);order=torch.tensor([3,0,2,1])
    a=m.objectives(x,valid);y=x.detach()[order].requires_grad_();b=m.objectives(y,valid[order])
    for k in a:
        torch.testing.assert_close(a[k],b[k])
        ga=torch.autograd.grad(a[k],x,retain_graph=True)[0]
        gb=torch.autograd.grad(b[k],y,retain_graph=True)[0]
        torch.testing.assert_close(ga[order],gb)


def test_single_valid_set_nll_reduces_to_candidate_ce_and_binary_pair_matches():
    x=torch.tensor([.3,.9],dtype=torch.float64,requires_grad=True)
    r=m.objectives(x,torch.tensor([False,True]))
    torch.testing.assert_close(r['set_mass_nll'],torch.nn.functional.cross_entropy(x[None,:],torch.tensor([1])))
    torch.testing.assert_close(r['pairwise_valid_invalid'],r['set_mass_nll'])


def test_gradients_match_central_finite_differences():
    x=torch.tensor([.2,-1.,.7],dtype=torch.float64,requires_grad=True);valid=torch.tensor([True,True,False])
    for name,value in m.objectives(x,valid).items():
        grad=torch.autograd.grad(value,x,retain_graph=True)[0]
        for i in range(3):
            delta=torch.zeros(3,dtype=torch.float64);delta[i]=1e-5
            fd=(m.objectives(x.detach()+delta,valid)[name]-m.objectives(x.detach()-delta,valid)[name])/(2e-5)
            assert float(grad[i])==pytest.approx(float(fd),abs=1e-9)


@pytest.mark.parametrize('mask',[[False,False,False],[True,True,True]])
def test_uninformative_support_requires_explicit_handling(mask):
    with pytest.raises(ValueError):m.objectives(torch.zeros(3),torch.tensor(mask))
