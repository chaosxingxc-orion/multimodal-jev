import pytest
import torch
from multimodal_jev.span_pair_verl import reference_pair_loss
from multimodal_jev.span_pair_credit import expected_pair_reward


@pytest.mark.parametrize('beta',[0.,.5,1.])
def test_pair_mean_and_duplicate_batch_gradient_normalization(beta):
    logits=torch.tensor([[.3,-.2],[.5,.1],[-.4,.8],[.8,.1]],dtype=torch.float64,requires_grad=True)
    gold=torch.tensor([1,0,0,1]);pairs=torch.tensor([[0,2],[3,1]])
    actor=logits.log_softmax(-1);old=actor.detach()
    loss=reference_pair_loss(old,actor,gold,pairs,beta)
    grad=torch.autograd.grad(loss,logits,retain_graph=True)[0]
    oracle=torch.autograd.grad(-expected_pair_reward(actor,gold,pairs,beta).mean(),logits,retain_graph=True)[0]
    torch.testing.assert_close(grad,oracle)
    repeated=reference_pair_loss(old.repeat(2,1),actor.repeat(2,1),gold.repeat(2),torch.cat([pairs,pairs+4]),beta)
    torch.testing.assert_close(torch.autograd.grad(repeated,logits)[0],grad)


@pytest.mark.parametrize('clip',[0.,1.,float('nan'),True])
def test_invalid_clipping_rejected(clip):
    old=torch.full((2,2),.5).log()
    with pytest.raises(ValueError):reference_pair_loss(old,old,torch.tensor([1,0]),torch.tensor([[0,1]]),.5,clip)
