#!/usr/bin/env python3
"""Small exact comparison of whole-tuple verl loss and manual PPO reference."""
import json
import torch
from multimodal_jev.span_policy import verl_joint_loss

torch.manual_seed(17)
old=torch.tensor([[-1.,-2.,-3.],[-1.5,-.5,-2.5]])
new=(old+torch.tensor([[.05,-.07,.03],[-.04,.02,.08]])).requires_grad_()
rewards=torch.tensor([[0.,1.,.5],[1.,0.,0.]])
loss,metrics=verl_joint_loss(old,new,rewards,['request-a','request-b'])
# Pinned verl GRPO uses sample std and epsilon 1e-6.
adv=(rewards-rewards.mean(-1,keepdim=True))/(rewards.std(-1,keepdim=True)+1e-6)
ratio=(new-old).exp()
expected=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2)).mean()
torch.testing.assert_close(loss,expected,atol=1e-6,rtol=0)
grad=torch.autograd.grad(loss,new,retain_graph=True)[0]
expected_grad=torch.autograd.grad(expected,new)[0]
torch.testing.assert_close(grad,expected_grad,atol=1e-6,rtol=0)
assert grad.abs().sum()>0
zero,_=verl_joint_loss(old,new,torch.ones_like(rewards),['request-a','request-b'])
assert float(zero.detach())==0
print(json.dumps({'status':'passed','loss':float(loss.detach()),'gradient_max_error':float((grad-expected_grad).abs().max()),'constant_reward_loss':float(zero.detach()),'scope':'CPU mathematical comparison using installed verl; no actor update or NPU/RL performance claim.'}))
