import pytest
import torch

from tools.reports.analyze_context_dual_signal import branch_summary,aggregate


def branch():
    target=torch.tensor([True,False,False])
    actions=torch.tensor([[1]*16,[1]*16,[0]*8+[1]*8])
    rewards=(actions==target[:,None]).float();centered=rewards-rewards.mean(1,keepdim=True)
    return dict(target=target,actions=actions,rewards=rewards,
        advantages=centered/centered.square().mean(1,keepdim=True).sqrt().clamp_min(.25),
        probability=torch.tensor([.99,.99,.5]),weight=torch.tensor([.5,.25,.25]))


def test_correct_wrong_mixed_and_class_balanced_weight_are_distinct():
    r=branch_summary(branch())
    assert r['all']['counts']==dict(all_correct=1,all_wrong=1,mixed=1)
    assert r['all']['weight']==dict(all_correct=.5,all_wrong=.25,mixed=.25)
    assert r['positive']['counts']['all_correct']==1
    assert r['negative']['counts']['all_wrong']==1
    x=aggregate([dict(branches={'ranking':r,'existence':r})])
    assert x['queries']['has_all_wrong_component']==1
    assert x['queries']['has_mixed_component']==1
    assert x['queries']['has_wrong_but_no_mixed']==0


def test_zero_rl_signal_does_not_mean_correct():
    b=branch();b['actions'][:]=1-b['target'][:,None].long();b['rewards'].zero_();b['advantages'].zero_()
    r=branch_summary(b);x=aggregate([dict(branches={'ranking':r,'existence':r})])
    assert x['queries']['has_wrong_but_no_mixed']==1
    assert x['queries']['all_components_sampled_correct']==0
    assert x['branches']['existence']['all']['fractions']['all_wrong']==1


@pytest.mark.parametrize('fault',['advantage','rewards','probability','weights','group'])
def test_reject_inconsistent_recorded_signal(fault):
    b=branch()
    if fault=='advantage':b['advantages'].fill_(1)
    if fault=='rewards':b['rewards'].fill_(.5)
    if fault=='probability':b['probability'][0]=float('nan')
    if fault=='weights':b['weight'].zero_()
    if fault=='group':b['rewards']=b['rewards'][:,:8]
    with pytest.raises(ValueError):branch_summary(b)
