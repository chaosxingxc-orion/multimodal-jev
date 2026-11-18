#!/usr/bin/env python3
"""Pinned native verl surrogate plus candidate ranking on controlled logits.

Rewards below are arithmetic fixtures, not an adopted Jev reward contract.
No model or dataset training, synthetic inputs do not prove task improvement.
"""
import argparse
import json
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,output):
    import torch
    from multimodal_jev.context_binary_policy import binary_logprobs
    from multimodal_jev.context_candidate_objective import loss as auxiliary
    from multimodal_jev.span_policy import sample_spans,joint_logprobs
    from multimodal_jev.span_group_advantage import group_advantages
    from multimodal_jev.span_sampled_hybrid import loss as native_loss
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('dependency changed '+p)
    torch.set_num_threads(2);torch.manual_seed(317)
    old_z=torch.tensor([[.3,-.5,.8],[-.7,.6,.2],[.4,-.9,.1],[.2,.4,-.6]])
    labels=torch.tensor([[True,False,False],[True,True,False],[False,False,False],[True,False,False]])
    observed=torch.tensor([[True,True,True],[True,True,False],[False,False,False],[True,True,True]])
    weights=torch.tensor([1.,.7,1.3,0.]);real=torch.tensor([True,True,True,False])
    request_ids=['q0','q1','q2','padding0']
    policy=binary_logprobs(old_z)
    actions,old_joint=sample_spans(policy,16,generator=torch.Generator().manual_seed(317))
    # Reward solely for testing estimator/surrogate composition. All-zero group
    # retained; production masks and set rewards must be specified separately.
    rewards=((actions.bool()==labels[:,None,:]) & observed[:,None,:]).sum(-1).float()/3
    results=[]
    for algorithm in ('grpo_v1','rloo_v1'):
        advantages=group_advantages(rewards,request_ids,algorithm=algorithm)
        centered=rewards-rewards.mean(1,keepdim=True)
        reference_adv=(centered/(rewards.std(1,keepdim=True)+1e-6) if algorithm=='grpo_v1'
                       else centered*16/15)
        if not torch.allclose(advantages,reference_adv,atol=2e-6,rtol=0):raise ValueError('advantage differs')
        # Enough displacement to exercise clipping in both directions.
        z=(old_z+torch.tensor([[1.2,-1.,.7]])).requires_grad_()
        current=binary_logprobs(z)
        native,_=native_loss(current,actions,old_joint,advantages,labels.long(),observed,
            torch.ones_like(z),weights,real,global_real_requests=3,coefficient=0.)
        aux,diag=auxiliary(z,labels,observed,weights,real,global_real_requests=3,
            ranking_coefficient=.4,probability_coefficient=.2)
        total=native+aux
        from multimodal_jev.context_hybrid_objective import loss as composed_loss
        composed,composed_metrics=composed_loss(z,labels,observed,weights,real,stage='task',
            global_real_requests=3,ranking_coefficient=.4,probability_coefficient=.2,
            actions=actions,old_joint=old_joint,advantages=advantages)
        gradient=torch.autograd.grad(total,z,retain_graph=True)[0]
        composed_gradient=torch.autograd.grad(composed,z,retain_graph=True)[0]
        atomic_json(output/(algorithm+'-composition.json'),dict(
            loss_max_error=float((composed-total).detach().abs()),
            gradient_max_error=float((composed_gradient-gradient).abs().max())))
        # Separate equivalent autograd graphs need not accumulate FP32 terms
        # bit-for-bit. Use the existing independent-formula diagnostic bound.
        if (not torch.allclose(composed,total,atol=2e-6,rtol=0) or
                not torch.allclose(composed_gradient,gradient,atol=2e-6,rtol=0)):
            raise ValueError('composed context objective differs from separate native/auxiliary')
        ratio=(joint_logprobs(current,actions)-old_joint).exp()
        reference_rl=(torch.maximum(-advantages*ratio,-advantages*ratio.clamp(.8,1.2)).mean(1)*weights).sum()/3
        reference=reference_rl+aux
        ref_grad=torch.autograd.grad(reference,z)[0]
        if not torch.allclose(total,reference,atol=2e-6,rtol=0) or not torch.allclose(gradient,ref_grad,atol=2e-6,rtol=0):
            raise ValueError('native clipped hybrid differs from independent equation')
        split=z.detach().clone().requires_grad_();accum=split.sum()*0
        for i in range(4):
            part,_=native_loss(binary_logprobs(split[i:i+1]),actions[i:i+1],old_joint[i:i+1],advantages[i:i+1],
                labels[i:i+1].long(),observed[i:i+1],torch.ones_like(split[i:i+1]),weights[i:i+1],real[i:i+1],
                global_real_requests=3,world_size=2,coefficient=0.)
            extra,_=auxiliary(split[i:i+1],labels[i:i+1],observed[i:i+1],weights[i:i+1],real[i:i+1],
                global_real_requests=3,world_size=2,ranking_coefficient=.4,probability_coefficient=.2)
            accum=accum+(part+extra)/2
        split_grad=torch.autograd.grad(accum,split)[0]
        if not torch.allclose(split_grad,gradient,atol=2e-6,rtol=0):raise ValueError('rank/micro reduction differs')
        if gradient[2:].abs().max()!=0:raise ValueError('zero-signal or padding query got gradient')
        results.append(dict(algorithm=algorithm,loss=float(total.detach()),
            advantage_max_error=float((advantages-reference_adv).abs().max()),
            reference_gradient_max_error=float((gradient-ref_grad).abs().max()),
            composed_gradient_max_error=float((composed_gradient-gradient).abs().max()),
            split_gradient_max_error=float((gradient-split_grad).abs().max()),
            ratio_outside_clip=int(((ratio<.8)|(ratio>1.2)).sum()),auxiliary=diag))
    return dict(passed=True,records=results,training_dispatch_allowed=False,
        scope='Pinned native verl GRPO/RLOO clipped plus pairwise/probability auxiliary on synthetic logits. '
              'Analytic world-size mean/micro accumulation, not a launched distributed Qwen actor or reward selection.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'result.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
