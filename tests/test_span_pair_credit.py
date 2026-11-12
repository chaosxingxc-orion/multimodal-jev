import itertools
import pytest
import torch

from multimodal_jev.span_pair_credit import exact_pair_credit,expected_pair_reward,pair_surrogate,learning_pairs


@pytest.mark.parametrize('weight',[0.,.5,1.])
def test_conditional_credit_matches_enumeration_and_expected_reward_gradient(weight):
    logits=torch.tensor([[.3,-.5],[.7,.1],[-.1,.6],[.8,-.4]],dtype=torch.float64,requires_grad=True)
    actor=logits.log_softmax(-1);old=actor.detach()
    target=torch.tensor([1,0,0,1]);pairs=torch.tensor([[0,2],[3,1]])
    adv=exact_pair_credit(old,target,pairs,weight)
    actions=torch.tensor(list(itertools.product((0,1),repeat=2)))
    for a,b in pairs.tolist():
        probability=old[a].exp()[actions[:,0]]*old[b].exp()[actions[:,1]]
        correct=actions==target[torch.tensor([a,b])]
        reward=(1-weight)*correct.double().mean(-1)+weight*correct.all(-1).double()
        expected=(probability*reward).sum()
        for position,row in enumerate([a,b]):
            for value in (0,1):
                mask=actions[:,position]==value
                conditional=(probability[mask]*reward[mask]).sum()/probability[mask].sum()
                torch.testing.assert_close(adv[row,value],conditional-expected)
    g=torch.autograd.grad(pair_surrogate(old,actor,target,pairs,weight),logits,retain_graph=True)[0]
    oracle=torch.autograd.grad(-expected_pair_reward(actor,target,pairs,weight).mean(),logits)[0]
    torch.testing.assert_close(g,oracle)


def test_wrong_but_complementary_answers_receive_no_pair_correctness_bonus():
    target=torch.tensor([1,0]);pairs=torch.tensor([[0,1]])
    wrong=torch.tensor([[.99,.01],[.01,.99]],dtype=torch.float64).log()
    right=wrong.flip(-1);constant_false=torch.tensor([[.99,.01],[.99,.01]],dtype=torch.float64).log()
    assert expected_pair_reward(wrong,target,pairs,1.).item()==pytest.approx(.0001)
    assert expected_pair_reward(right,target,pairs,1.).item()==pytest.approx(.9801)
    assert expected_pair_reward(constant_false,target,pairs,1.).item()==pytest.approx(.0099)


def test_tied_model_can_still_have_stationary_uninformative_solution():
    # Same logits for both polarities: pair reward alone cannot establish rule use.
    logit=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    logits=torch.stack([logit*0,logit]).expand(2,-1)
    reward=expected_pair_reward(logits.log_softmax(-1),torch.tensor([1,0]),torch.tensor([[0,1]]),1.).mean()
    assert reward.item()==.25
    assert torch.autograd.grad(reward,logit)[0].item()==0.


@pytest.mark.parametrize('fault',['duplicate','same_gold','missing','unnormalized','weight','gradient_old'])
def test_invalid_pair_contract_rejected(fault):
    old=torch.full((4,2),.5).log();gold=torch.tensor([0,1,0,1]);pairs=torch.tensor([[0,1],[2,3]]);weight=.5
    if fault=='duplicate':pairs=torch.tensor([[0,1],[0,3]])
    elif fault=='same_gold':gold[1]=0
    elif fault=='missing':pairs=pairs[:1]
    elif fault=='unnormalized':old=old+1
    elif fault=='weight':weight=float('nan')
    elif fault=='gradient_old':old.requires_grad_()
    with pytest.raises(ValueError):exact_pair_credit(old,gold,pairs,weight)


@pytest.mark.parametrize('fault',[None,'selection','ineligible','missing','wrong_rule','wrong_audio','gold'])
def test_pairs_require_source_identity_and_reviewed_learning_semantics(fault):
    from copy import deepcopy
    rule={'op':'fact','key':'a'}
    a=dict(id='a',split='learning',optimization_allowed=True,arm='final_only',targets=[True],supervised=[True],
           provenance=dict(source_group='source',recording_id='audio',audio_sha256='digest',
                           complement_pair_id='rule',rule_polarity='positive',rule=rule,
                           dependency_keys=['a'],dependency_targets=[True]))
    b=deepcopy(a);b.update(id='b',targets=[False]);b['provenance'].update(rule_polarity='negative',rule={'op':'not','arg':rule})
    if fault=='selection':b['split']='selection'
    elif fault=='ineligible':b['optimization_allowed']=False
    elif fault=='missing':b['supervised']=[False]
    elif fault=='wrong_rule':b['provenance']['rule']=rule
    elif fault=='wrong_audio':b['provenance']['audio_sha256']='other'
    elif fault=='gold':b['targets']=[True]
    if fault:
        with pytest.raises(ValueError):learning_pairs([b,a],{'a','b'})
    else:
        assert learning_pairs([b,a],{'a','b'}).tolist()==[[1,0]]
        with pytest.raises(ValueError):learning_pairs([b,a],{'a'})
