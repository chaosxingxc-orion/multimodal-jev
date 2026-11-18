import copy
import itertools
import json
from pathlib import Path

import pytest
import torch

from multimodal_jev.context_task_reward import score, advantages, validate


def config():
    return json.loads((Path(__file__).parents[1]/'configs/experiments/context-stage2-task-reward-design-v1.json').read_text())['task_reward']


def test_exhaustive_sets_match_manual_reward_and_strict_exact_gap():
    bits=list(itertools.product([0,1], repeat=4))
    actions=torch.tensor([bits]*4)
    labels=torch.eye(4,dtype=torch.bool)
    result=score(actions,labels,torch.ones_like(labels),config())
    for i in range(4):
        for j,values in enumerate(bits):
            exact=values[i]==1 and sum(values)==1
            expected=.5*exact+.5*(2*values[i]/(sum(values)+1))
            assert result['reward'][i,j].item()==pytest.approx(expected)
            if not exact:assert result['reward'][i,j]<.5
    assert not result['reward'].requires_grad


def test_sixty_candidates_partial_all_empty_wrong_and_permutation():
    labels=torch.zeros(1,60,dtype=torch.bool);labels[0,17]=True
    exact=labels[0].long();partial=exact.clone();partial[18]=1
    wrong=torch.zeros(60,dtype=torch.long);wrong[18]=1
    actions=torch.stack([exact,partial,torch.ones_like(exact),torch.zeros_like(exact),wrong])[None]
    result=score(actions,labels,torch.ones_like(labels),config())
    torch.testing.assert_close(result['reward'],torch.tensor([[1.,1/3,1/61,0.,0.]]))
    permutation=torch.arange(59,-1,-1)
    changed=score(actions[:,:,permutation],labels[:,permutation],torch.ones_like(labels),config())
    for key in result:torch.testing.assert_close(result[key],changed[key])


def test_pure_exact_configuration_and_missing_truth_rejection():
    c=config();c.update(exact_weight=1.,overlap_weight=0.)
    actions=torch.tensor([[[1,0],[1,1]]]);labels=torch.tensor([[True,False]])
    assert score(actions,labels,torch.ones_like(labels),c)['reward'].tolist()==[[1.,0.]]
    with pytest.raises(ValueError,match='complete set'):score(actions,labels,torch.zeros_like(labels),c)
    with pytest.raises(ValueError,match='exactly one'):score(actions,torch.ones_like(labels),torch.ones_like(labels),c)


@pytest.mark.parametrize('key,value',[('exact_weight',0.),('beta',float('nan')),('beta',True),('overlap_weight',.9)])
def test_invalid_configuration_rejected(key,value):
    c=config();c[key]=value
    with pytest.raises(ValueError):validate(c)


@pytest.mark.parametrize('algorithm',['task_grpo_floor_v1','task_rloo_v1'])
def test_continuous_group_credit_preserves_order_and_zero_variance(algorithm):
    rewards=torch.tensor([[0.,1/3,1.],[1.,1.,1.]])
    a=advantages(rewards,algorithm=algorithm,std_floor=.25)
    assert a[0,0]<a[0,1]<a[0,2]
    torch.testing.assert_close(a.mean(-1),torch.zeros(2),atol=1e-6,rtol=0)
    assert not a[1].any() and not a.requires_grad
    if algorithm=='task_rloo_v1':
        torch.testing.assert_close(a[0],torch.tensor([-2/3,-1/6,5/6]))


def test_old_binary_algorithm_not_silently_relaxed():
    from multimodal_jev.context_dual_credit import advantages as binary, ALGORITHMS
    rewards=torch.tensor([[0.,.5,1.]])
    with pytest.raises(ValueError,match='binary'):binary(rewards,ALGORITHMS[-1])
    with pytest.raises(ValueError):advantages(rewards.requires_grad_(),algorithm='task_grpo_floor_v1',std_floor=.25)
