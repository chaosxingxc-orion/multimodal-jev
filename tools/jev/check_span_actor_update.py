#!/usr/bin/env python3
"""Installed-verl CPU check of the BF16 update adapter, using synthetic policy."""
import json
import torch
from multimodal_jev.bf16_lora import BF16MasterAdamW
from multimodal_jev.span_actor_update import record_rollout,update_span_actor

torch.set_num_threads(2)
parameter=torch.nn.Parameter(torch.tensor([[.1,-.1]],dtype=torch.bfloat16))
forward=lambda:(parameter.float().log_softmax(-1),)
rollout=record_rollout('synthetic-v1',('synthetic-input',),forward(),64,
                       torch.Generator().manual_seed(17))
rewards=(rollout.actions[...,0]==1).float()
assert rewards.min()==0 and rewards.max()==1
before=parameter.detach().clone()
old=rollout.old_joint_logprobs.clone()
optimizer=BF16MasterAdamW([parameter],lr=.01)
receipt=update_span_actor(rollout,forward,optimizer,[parameter],rewards,
                          rollout.policy_id,rollout.request_ids)
assert receipt['gradient_norm']>0 and not torch.equal(parameter,before)
assert torch.equal(old,rollout.old_joint_logprobs)
assert forward()[0][0,1]>before.float().log_softmax(-1)[0,1]
print(json.dumps({'status':'passed','update':receipt,'precision':optimizer.receipt(),
                  'scope':'Synthetic CPU BF16 optimizer bridge using installed verl. No vLLM rollout or speech RL claim.'}))
