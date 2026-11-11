#!/usr/bin/env python3
"""Pinned-image CPU comparison of exact-support loss against actual verl."""
import argparse
import json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json
from multimodal_jev.span_conditional_credit import reference_conditional_loss,verl_conditional_loss


def check():
    cases=[]
    target=torch.tensor([[1,-100,0],[0,1,1]]);mask=target!=-100
    for shift in (0.,.1,1.,8.,30.):
        for weight in (0.,.2):
            old_logits=torch.tensor([[[.3,-.2],[.5,.1],[-.4,.8]],[[.8,.1],[-.3,.6],[.2,-.1]]])
            logits=(old_logits+torch.tensor([shift,-shift])).requires_grad_()
            old=tuple(old_logits[:,j].log_softmax(-1) for j in range(3))
            actor=tuple(logits[:,j].log_softmax(-1) for j in range(3))
            actual,metrics=verl_conditional_loss(old,actor,target,mask,weight)
            expected=reference_conditional_loss(old,actor,target,mask,weight)
            a=torch.autograd.grad(actual,logits,retain_graph=True)[0]
            b=torch.autograd.grad(expected,logits)[0]
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=0)
            cases.append(dict(logit_shift=shift,auxiliary_weight=weight,loss=float(actual.detach()),
                gradient_max_error=float((a-b).abs().max()),metrics=metrics))
    return dict(passed=True,cases=cases,optimizer_updates=0,
        scope='CPU actual verl marginal clipped-loss/gradient audit, including off-policy clipping and log-ratio clamp; not NPU update, joint-PPO equivalence or task gain')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    r=check();atomic_json(a.output,r);print(json.dumps(r,indent=2))
