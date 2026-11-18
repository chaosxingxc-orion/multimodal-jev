#!/usr/bin/env python3
"""Actual four-rank native verl context update on synthetic audio-like inputs."""
import argparse
import copy
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from multimodal_jev.baseline import atomic_json
from multimodal_jev.context_decision_model import ContextDecisionModel
from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.context_audio_collate import forward_context_audio_batch
from multimodal_jev.context_native_actor import MODE,build_actor,make_batch,admit
from multimodal_jev.context_binary_policy import binary_logprobs
from multimodal_jev.context_set_reward import set_f1
from multimodal_jev.span_policy import sample_spans,joint_logprobs
from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
from probe_span_native_actor_dp_cpu import config,snapshot

COLLATION=dict(decision_token_id=99,pad_token_id=0,audio_token_id=98,audio_chunk_size=100,audio_infer_window=200)

class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(100,8);self.audio=torch.nn.Linear(4,8,bias=False)
    def forward(self,input_ids,attention_mask,input_features,input_features_mask,position_ids,use_cache,return_dict):
        feature=(input_features*input_features_mask[:,None]).sum(-1)/input_features_mask.sum(-1)[:,None]
        hidden=(self.embedding(input_ids)+self.audio(feature)[:,None])*attention_mask[:,:,None]
        return SimpleNamespace(last_hidden_state=hidden.cumsum(1))

def model():return ContextDecisionModel(Backbone(),ContextDecisionHead(8,2,16),99)

def optimizer(module):
    # Match the context-event training baseline, not the older linear-head
    # diagnostic's lr=.01. Adam can amplify near-zero attention-key bias
    # roundoff; keep the independent comparison tolerance unchanged.
    return torch.optim.AdamW(module.parameters(),lr=1e-4,betas=(.9,.999),
        eps=1e-8,weight_decay=0.,foreach=False,fused=False)

def items_for(ref,step,stage,fixed):
    generator=torch.Generator().manual_seed(500+step);items=[]
    for i in range(8):
        frames=100 if i%2==0 else 190;count=13 if frames==100 else 25;width=100 if frames==100 else 200
        ids=torch.tensor([[1]+[98]*count+[2,99,3,99]])
        inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),input_features=torch.randn(1,4,width,generator=generator),
                    input_features_mask=(torch.arange(width)<frames).long()[None])
        item=dict(request_id=f'{step}/{i}',policy_id=f'policy{step}',inputs=inputs,positions=torch.tensor([[count+2,count+4]]),
            labels=torch.tensor([[i%3==0,i%3==1]]),observed=torch.ones(1,2,dtype=torch.bool),request_weight=0. if i==7 else 1.,real_request=i!=7)
        if stage=='task':
            transport={k:item[k] for k in ('request_id','policy_id','inputs','positions')}
            with torch.no_grad():
                logits=forward_context_audio_batch(ref,[transport],device='cpu',dtype=torch.float32,policy_id=item['policy_id'],**COLLATION)
                reference=forward_context_audio_batch(fixed,[transport],device='cpu',dtype=torch.float32,policy_id=item['policy_id'],**COLLATION)
            item['behavior']=tuple(binary_logprobs(logits));item['actions']=sample_spans(item['behavior'],8,generator)[0]
            item['reference_policy']=tuple(binary_logprobs(reference));item['reference_policy_id']='fixed'
        items.append(item)
    return items

def independent_loss(ref,items,stage,algorithm):
    inputs=[{k:p[k] for k in ('request_id','policy_id','inputs','positions')} for p in items]
    z=forward_context_audio_batch(ref,inputs,device='cpu',dtype=torch.float32,policy_id=items[0]['policy_id'],**COLLATION)
    total=z.sum()*0
    for i,p in enumerate(items):
        y=p['labels'][0];v=z[i];positive=v[y];negative=v[~y]
        ranking=torch.nn.functional.softplus(negative[None]-positive[:,None]).mean() if positive.numel() and negative.numel() else v.sum()*0
        probability=torch.nn.functional.binary_cross_entropy_with_logits(v,y.float())
        value=.4*ranking+.2*probability
        if stage=='task':
            rewards=set_f1(p['actions'],p['labels'],p['observed'])
            centered=rewards-rewards.mean(-1,keepdim=True)
            adv=centered/(rewards.std(-1,keepdim=True)+1e-6) if algorithm=='grpo_v1' else centered*8/7
            policy=binary_logprobs(z[i:i+1]);old=joint_logprobs(p['behavior'],p['actions']);new=joint_logprobs(policy,p['actions'])
            ratio=(new-old).clamp(-20,20).exp();clipped=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2))
            value=value+torch.where(adv<0,torch.minimum(-adv*3,clipped),clipped).mean()
            value=value+.01*sum((a.exp()*(a-b)).sum() for a,b in zip(policy,p['reference_policy'],strict=True))
        total=total+p['request_weight']*value/7
    return total

def worker(rank,root):
    root=Path(root);torch.set_num_threads(1);torch.manual_seed(317)
    dist.init_process_group('gloo',init_method=(root/'gloo-init').resolve().as_uri(),rank=rank,world_size=4,timeout=timedelta(seconds=240))
    try:
        initial=copy.deepcopy(model().state_dict());cases=[]
        for algorithm in ('grpo_v1','rloo_v1'):
            for micro in (1,2):
                bare=model();bare.load_state_dict(initial);wrapped=DDP(bare);opt=optimizer(wrapped)
                ref=model();ref.load_state_dict(initial);refopt=optimizer(ref)
                fixed=model();fixed.load_state_dict(initial)
                for step in range(3):
                    stage='instruction' if step==0 else 'task';items=items_for(ref,step,stage,fixed)
                    options=dict(stage=stage,ranking_coefficient=.4,probability_coefficient=.2,algorithm=algorithm,
                                 kl_coefficient=0. if stage=='instruction' else .01,reference_policy_id=None if stage=='instruction' else 'fixed')
                    cfg=config(micro,f'policy{step}');cfg.ppo_mini_batch_size=2;cfg.policy_loss.loss_mode=MODE;cfg.jev_context_objective=options
                    # Synthetic CPU arithmetic only, not the production BF16
                    # runtime's acceptance bounds.
                    cfg.jev_context_parity=dict(probability_max_abs=2e-6,binary_logprob_max_abs=2e-6)
                    chunks=list(make_batch(items,**options).split(2)) if rank==0 else None
                    incoming=[None];dist.scatter_object_list(incoming,chunks,src=0);batch=incoming[0]
                    refopt.zero_grad();independent_loss(ref,items,stage,algorithm).backward()
                    grads=[p.grad.clone() for p in ref.parameters()];torch.nn.utils.clip_grad_norm_(ref.parameters(),1.);refopt.step()
                    actor=build_actor(cfg,wrapped,opt,collation=COLLATION)
                    before_update_model=copy.deepcopy(bare.state_dict())
                    captured={};original=actor._optimizer_step
                    def capture():
                        captured['grads']=[p.grad.clone() for p in wrapped.parameters()]
                        return original()
                    actor._optimizer_step=capture
                    with native_actor_cpu_device(actor):actor.update_policy(batch)
                    expected=snapshot(ref,refopt,grads);actual=snapshot(wrapped,opt,captured['grads']);errors={}
                    for name,values in expected.items():
                        errors[name]=max(float((a-b).abs().max()) for a,b in zip(values,actual[name],strict=True))
                    atomic_json(root/f'comparison-rank-{rank}.json',dict(algorithm=algorithm,
                        micro=micro,step=step,errors=errors))
                    for name,values in expected.items():
                        for a,b in zip(values,actual[name],strict=True):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)
                    assert all(opt.state[p]['step'].item()==step+1 for p in wrapped.parameters())
                    for fault in ('denominator','mask','padding','duplicate'):
                        bad=copy.deepcopy(batch)
                        if fault=='duplicate':
                            shared=[bad if rank==0 else None];dist.broadcast_object_list(shared,src=0);bad=shared[0]
                        elif rank==3:
                            if fault=='denominator':bad.non_tensor_batch['multi_modal_inputs'][0]['global_real_requests']+=1
                            if fault=='mask':bad.batch['response_mask'][0,0]=0
                            if fault=='padding':bad.non_tensor_batch['multi_modal_inputs'][-1]['request_weight']=1.
                        try:admit(bad,cfg)
                        except ValueError:pass
                        else:raise AssertionError('corrupt rank batch accepted '+fault)
                    if stage=='task' and micro==2:
                        corrupted=copy.deepcopy(items)
                        corrupted[6]['behavior']=tuple(binary_logprobs(torch.tensor([[2.,-2.]])))
                        parts=list(make_batch(corrupted,**options).split(2)) if rank==0 else None
                        invalid=[None];dist.scatter_object_list(invalid,parts,src=0)
                        # Rollout metadata is internally consistent and current
                        # ID is unchanged; model agreement must still reject it.
                        saved_model=copy.deepcopy(bare.state_dict())
                        saved_opt=copy.deepcopy(opt.state_dict())
                        bare.load_state_dict(before_update_model)
                        try:
                            with native_actor_cpu_device(actor):actor.update_policy(invalid[0])
                        except ValueError as exc:
                            assert 'context behavior parity failed' in str(exc)
                        else:raise AssertionError('different behavior accepted')
                        assert all(torch.equal(v,bare.state_dict()[k]) for k,v in before_update_model.items())
                        for key,fields in saved_opt['state'].items():
                            for name,value in fields.items():
                                assert torch.equal(value,opt.state_dict()['state'][key][name])
                        bare.load_state_dict(saved_model)
                    cases.append(dict(algorithm=algorithm,micro=micro,step=step,errors=errors,admission=actor.last_admission,
                        behavior_parity=None if stage=='instruction' else actor.last_behavior_parity,
                        mismatched_behavior_rejected=stage=='task' and micro==2))
                    if step==1:
                        path=root/f'restore-{algorithm}-{micro}-{rank}.pt'
                        torch.save(dict(model=bare.state_dict(),optimizer=opt.state_dict()),path)
                        with torch.no_grad():
                            for p in bare.parameters():p.add_(1.)
                        saved=torch.load(path,weights_only=True);bare.load_state_dict(saved['model']);opt=optimizer(wrapped);opt.load_state_dict(saved['optimizer'])
        atomic_json(root/f'rank-{rank}.json',dict(passed=True,cases=cases))
    finally:dist.destroy_process_group()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only diagnostic')
    args.output.mkdir(exist_ok=False)
    mp.spawn(worker,args=(str(args.output),),nprocs=4,join=True)
    atomic_json(args.output/'result.json',dict(passed=True,ranks=4,cases_per_rank=12,
        optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),
        scope='Actual native verl actor/DDP optimizer and restore with synthetic backbone and actual shared Transformer head; not Qwen/NPU or efficacy.'))
