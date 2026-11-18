import itertools
import pytest
import torch
from multimodal_jev.context_set_reward import set_f1


def test_all_four_bit_truth_and_prediction_sets_against_python_sets():
    bits=list(itertools.product([0,1],repeat=4))
    actions=torch.tensor([bits]*len(bits))
    labels=torch.tensor(bits,dtype=torch.bool)
    actual=set_f1(actions,labels,torch.ones_like(labels))
    for i,truth in enumerate(bits):
        positive={k for k,v in enumerate(truth) if v}
        for j,pred in enumerate(bits):
            selected={k for k,v in enumerate(pred) if v}
            expected=2*len(positive & selected)/(len(positive)+len(selected)) if positive or selected else 1
            assert float(actual[i,j])==pytest.approx(expected)
    assert not actual.requires_grad


def test_sixty_candidate_controls_and_permutation():
    truth=torch.zeros(1,60,dtype=torch.bool);truth[0,17]=True
    actions=torch.stack([torch.zeros(60,dtype=torch.long),torch.ones(60,dtype=torch.long),truth[0].long()])[None]
    mask=torch.ones_like(truth)
    reward=set_f1(actions,truth,mask)
    torch.testing.assert_close(reward,torch.tensor([[0.,2/61,1.]]))
    order=torch.randperm(60)
    assert torch.equal(reward,set_f1(actions[:,:,order],truth[:,order],mask[:,order]))
    truth[0,18]=True
    assert float(set_f1(actions,truth,mask)[0,2])==pytest.approx(2/3)


def test_missing_truth_and_nonbinary_actions_fail():
    actions=torch.zeros(1,4,3,dtype=torch.long)
    labels=torch.zeros(1,3,dtype=torch.bool);mask=torch.ones_like(labels)
    mask[0,2]=False
    with pytest.raises(ValueError,match='complete set'):set_f1(actions,labels,mask)
    mask[:]=True;actions[0,0,0]=2
    with pytest.raises(ValueError,match='binary'):set_f1(actions,labels,mask)
