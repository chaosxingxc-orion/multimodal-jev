"""Fixed-image tests; real pinned verl is required, never skipped."""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from test_context_loss_metrics_candidate import ACTOR, function
from tools.reports.context_loss_metrics_candidate import generate, validate_total, PREFIX
from multimodal_jev.context_hybrid_objective import loss
from multimodal_jev.context_binary_policy import binary_logprobs
from multimodal_jev.context_set_reward import set_f1
from multimodal_jev.span_policy import joint_logprobs
from multimodal_jev.span_group_advantage import group_advantages, CORE_SHA256


def payloads(algorithm, stage):
    old = torch.linspace(-2., 2., 8*60).reshape(8,60)
    generator = torch.Generator().manual_seed(317)
    actions = (torch.rand(8,16,60,generator=generator) < old.sigmoid()[:,None]).long()
    labels = torch.zeros(8,60,dtype=torch.bool);labels[torch.arange(8),torch.arange(8)] = True
    actions[:,0] = labels.long();actions[:,1] = 0
    actions[2] = 0  # Real zero-signal query remains in denominator.
    mask = torch.ones_like(labels)
    adv = group_advantages(set_f1(actions,labels,mask),[str(i) for i in range(8)],algorithm=algorithm)
    result=[]
    for i in range(8):
        behavior = tuple(p.detach() for p in binary_logprobs(old[i:i+1]))
        reference = tuple(p.detach() for p in binary_logprobs(old[i:i+1]*.5))
        task=stage=='task'
        result.append(dict(labels=labels[i:i+1],observed=mask[i:i+1],
            request_weight=0. if i==7 else 1.,real_request=i!=7,global_real_requests=7,
            stage=stage,ranking_coefficient=.3,probability_coefficient=.2,
            kl_coefficient=.01 if task else 0.,actions=actions[i:i+1] if task else None,
            old_joint=joint_logprobs(behavior,actions[i:i+1]) if task else None,
            advantages=adv[i:i+1] if task else None,reference_policy=reference if task else None))
    return old+.13,result


def invoke(ns,z,payload):
    state=dict(consumed=False,log_probs=z,logits=z,payload=payload)
    token=ns['_active'].set(state)
    try:
        cfg=NS(policy_loss=NS(loss_mode='test'),clip_ratio=.2,
               ppo_mini_batch_size=2,ppo_micro_batch_size_per_gpu=1)
        return ns['registered_loss'](None,z,None,None,config=cfg)
    finally:ns['_active'].reset(token)


@pytest.mark.parametrize('algorithm',['grpo_v1','rloo_v1'])
@pytest.mark.parametrize('stage',['instruction','task'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_actual_verl_loss_gradient_adam_and_reduction(algorithm,stage,dtype):
    from verl.trainer.ppo import core_algos
    assert hashlib.sha256(Path(inspect.getfile(core_algos)).read_bytes()).hexdigest()==CORE_SHA256
    source,_=generate(ACTOR)
    original=function(ACTOR.read_text(),loss);candidate=function(source,loss)
    initial,payload=payloads(algorithm,stage)
    results=[]
    for ns in (original,candidate):
        p=torch.nn.Parameter(initial.clone());opt=torch.optim.AdamW([p],lr=1e-4,foreach=False,fused=False)
        global_loss=p.sum()*0;logged=0.
        for rank in range(4):
            for micro in range(2):
                i=rank*2+micro
                value,metrics=invoke(ns,p[i:i+1].to(dtype),payload[i:i+1])
                global_loss=global_loss+value/2/4
                if ns is candidate:
                    validate_total(metrics)
                    logged+=metrics[PREFIX+'reconstructed_total']/4
        global_loss.backward();gradient=p.grad.clone();opt.step()
        results.append((global_loss.detach(),gradient,p.detach().clone(),opt.state_dict(),logged))
    for i in range(3):assert torch.equal(results[0][i],results[1][i])
    for key,value in results[0][3]['state'][0].items():
        assert torch.equal(value,results[1][3]['state'][0][key])
    assert results[1][4]==pytest.approx(float(results[1][0]),rel=2e-6,abs=2e-7)
    assert torch.equal(results[1][1][7],torch.zeros(60))
    assert results[1][1][2].abs().sum()>0  # Aux/KL survive constant reward.
