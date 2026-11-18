import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.span_group_advantage import group_advantages


@pytest.mark.parametrize('algorithm', ['grpo_v1','rloo_v1'])
def test_native_verl_matches_group_algebra(algorithm):
    pytest.importorskip('verl')
    rewards=torch.tensor([[0.,0.,0.,0.],[1.,1.,1.,1.],[0.,1.,0.,0.],[1.,0.,1.,0.]])
    original=rewards.clone()
    actual=group_advantages(rewards,['zero','one','mixed','half'],algorithm=algorithm)
    centered=rewards-rewards.mean(-1,keepdim=True)
    expected=(centered/(rewards.std(-1,keepdim=True)+1e-6) if algorithm=='grpo_v1'
              else rewards-(rewards.sum(-1,keepdim=True)-rewards)/3)
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
    assert torch.equal(rewards,original)
    assert torch.equal(actual[:2],torch.zeros(2,4))
    assert not actual.requires_grad


@pytest.mark.parametrize('fault',['singleton','duplicate','nonfinite','attached','unknown'])
def test_invalid_groups_rejected_before_import(fault):
    rewards=torch.zeros(2,4);ids=['a','b'];algorithm='grpo_v1'
    if fault=='singleton':rewards=torch.zeros(2,1)
    if fault=='duplicate':ids=['a','a']
    if fault=='nonfinite':rewards[0,0]=float('nan')
    if fault=='attached':rewards.requires_grad_()
    if fault=='unknown':algorithm='ppo_critic_v1'
    with pytest.raises(ValueError):group_advantages(rewards,ids,algorithm=algorithm)
