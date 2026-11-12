#!/usr/bin/env python3
"""Compare real pinned verl losses/gradients with pair-credit CPU oracles."""
import argparse
import inspect
import json
from pathlib import Path
import torch
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_pair_credit import expected_pair_reward
from multimodal_jev.span_pair_verl import reference_pair_loss,verl_pair_loss


def check():
    cases=[]
    gold=torch.tensor([1,0,0,1]);pairs=torch.tensor([[0,2],[3,1]])
    old_logits=torch.tensor([[.3,-.2],[.5,.1],[-.4,.8],[.8,.1]])
    for shift in (0.,.1,1.,8.,30.):
        for beta in (0.,.5,1.):
            logits=(old_logits+torch.tensor([shift,-shift])).requires_grad_()
            old=old_logits.log_softmax(-1);actor=logits.log_softmax(-1)
            actual,metrics=verl_pair_loss(old,actor,gold,pairs,beta)
            expected=reference_pair_loss(old,actor,gold,pairs,beta)
            a=torch.autograd.grad(actual,logits,retain_graph=True)[0]
            b=torch.autograd.grad(expected,logits,retain_graph=True)[0]
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=0)
            oracle_error=None
            if shift==0:
                oracle=torch.autograd.grad(-expected_pair_reward(actor,gold,pairs,beta).mean(),logits)[0]
                torch.testing.assert_close(a,oracle,atol=1e-6,rtol=0)
                oracle_error=float((a-oracle).abs().max())
            cases.append(dict(logit_shift=shift,pair_weight=beta,loss=float(actual.detach()),
                gradient_max_error=float((a-b).abs().max()),onpolicy_expected_gradient_error=oracle_error,
                metrics={k:float(v) for k,v in metrics.items()}))
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    import verl
    path=Path(inspect.getsourcefile(compute_policy_loss_vanilla))
    return dict(passed=True,cases=cases,torch_version=torch.__version__,verl_version=verl.__version__,
        verl_loss_source=str(path),verl_loss_source_sha256=file_hash(path),
        optimizer_updates=0,training_admitted=False,
        scope='CPU real verl marginal pair loss and gradient; off-policy clamp/clipping cases. '
              'Not joint-pair PPO equivalence, NPU update, probability parity or task gain.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=check();atomic_json(a.output,report);print(json.dumps(report,indent=2))
