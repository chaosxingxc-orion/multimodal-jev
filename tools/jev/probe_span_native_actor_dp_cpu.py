#!/usr/bin/env python3
"""Four real Gloo/DDP ranks exercise native typed verl actor updates.

Synthetic float32 features only. No audio/NPU/FSDP/Ray lifecycle claim.
"""
import argparse
import copy
import json
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from multimodal_jev.baseline import atomic_json


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head=torch.nn.Linear(3,60,bias=False)

    def forward(self,inputs,positions,candidates):
        z=self.head(inputs['input_features'])
        return z,tuple(torch.log_softmax(z[:,j,ids],-1) for j,ids in enumerate(candidates))


def config(micro, policy):
    from omegaconf import OmegaConf
    return OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,
        ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,
        entropy_from_logits_with_chunking=False,use_torch_compile=False,
        fsdp_config={'dtype':'float32'},calculate_sum_pi_squared=False,entropy_checkpointing=False,
        ppo_mini_batch_size=16,ppo_micro_batch_size_per_gpu=micro,ppo_epochs=1,use_kl_loss=False,
        calculate_entropy=False,entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,
        policy_loss={'loss_mode':'vanilla'},clip_ratio=.2,clip_ratio_low=None,clip_ratio_high=None,
        clip_ratio_c=3.,global_batch_info={},grad_clip=1.,jev_group_size=8,jev_policy_id=policy))


def make_items(model,step):
    from multimodal_jev.span_actor_update import record_rollout
    gen=torch.Generator().manual_seed(918+step)
    tasks=([list(range(60))],[[0,1]]*6,[[0,1,2],[3,4]])
    items=[]
    for index in range(8):
        candidates=tasks[index%3];n=len(candidates)
        inputs=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),
                    input_features=torch.randn(1,n,3,generator=gen),
                    input_features_mask=torch.ones(1,n,dtype=torch.long))
        positions=torch.arange(n)[None]
        with torch.no_grad():probabilities=model(inputs,positions,candidates)[1]
        rollout=record_rollout(f'policy{step}',[f'step{step}/request{index}'],probabilities,8,gen)
        # Detached synthetic reward varies across requests and sampled actions.
        rewards=((rollout.actions.float().sum(-1)+torch.arange(8)[None]+index)%5)/4
        items.append(dict(inputs=inputs,positions=positions,candidates=candidates,
                          rollout=rollout,rewards=rewards))
    return items


def optimizer(model):
    return torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0.,foreach=False,fused=False)


def snapshot(model,opt,gradients):
    return dict(gradients=gradients,parameters=[p.detach().clone() for p in model.parameters()],
                exp_avg=[opt.state[p]['exp_avg'].clone() for p in model.parameters()],
                exp_avg_sq=[opt.state[p]['exp_avg_sq'].clone() for p in model.parameters()])


def worker(rank,output,algorithm='grpo_v1'):
    from multimodal_jev.span_native_actor import make_batch,build_actor,ACTOR_SHA256
    from multimodal_jev.span_native_dp import admit_rank_batch
    from multimodal_jev.span_policy import joint_logprobs,verl_joint_loss
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
    output=Path(output);torch.set_num_threads(1);torch.manual_seed(317)
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    dist.init_process_group('gloo',init_method=(output/'gloo-init').resolve().as_uri(),
                            rank=rank,world_size=4,timeout=timedelta(seconds=90))
    try:
        initial=copy.deepcopy(Model().state_dict());cases=[]
        for micro in (8,16):
            module=Model();module.load_state_dict(initial);model=DDP(module);opt=optimizer(model)
            reference=Model();reference.load_state_dict(initial);refopt=optimizer(reference)
            for step in range(2):
                # Rank zero owns rollout creation; scatter serialized complete
                # DataProto chunks, preserving distinct request ownership.
                items=make_items(reference,step)
                chunks=list(make_batch(items,8,algorithm=algorithm).split(16)) if rank==0 else None
                incoming=[None];dist.scatter_object_list(incoming,chunks,src=0);batch=incoming[0]
                cfg=config(micro,f'policy{step}');batch.meta_info['micro_batch_size']=micro
                cfg.jev_advantage_algorithm=algorithm
                admission=admit_rank_batch(batch,cfg)
                refopt.zero_grad()
                for item in items:
                    probabilities=reference(item['inputs'],item['positions'],item['candidates'])[1]
                    logp=joint_logprobs(probabilities,item['rollout'].actions)
                    loss,_=verl_joint_loss(item['rollout'].old_joint_logprobs,logp,item['rewards'],item['rollout'].request_ids,algorithm=algorithm)
                    (loss/len(items)).backward()
                gradients=[p.grad.clone() for p in reference.parameters()]
                torch.nn.utils.clip_grad_norm_(reference.parameters(),1.);refopt.step()
                expected=snapshot(reference,refopt,gradients)
                actor=build_actor(cfg,model,opt);captured={};original=actor._optimizer_step
                def capture():
                    captured['gradients']=[p.grad.detach().clone() for p in model.parameters()]
                    return original()
                actor._optimizer_step=capture
                old=batch.batch['old_log_probs'].clone()
                with native_actor_cpu_device(actor):
                    actual=actor.compute_log_prob(batch)['log_probs']
                    torch.testing.assert_close(actual,old,atol=2e-7,rtol=2e-5)
                    metrics=actor.update_policy(batch)
                actual=snapshot(model,opt,captured['gradients']);errors={}
                for key,values in expected.items():
                    errors[key]=max(float((a-b).abs().max()) for a,b in zip(values,actual[key],strict=True))
                    for a,b in zip(values,actual[key],strict=True):
                        torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
                assert torch.equal(old,batch.batch['old_log_probs'])
                assert all(opt.state[p]['step'].item()==step+1 for p in model.parameters())
                cases.append(dict(micro=micro,step=step+1,admission=admission,errors=errors,metrics=metrics))
                # A local defect must fail all ranks before entering model
                # collectives; otherwise one worker can hang its peers.
                bad=copy.deepcopy(batch)
                if rank==3:bad.batch['response_mask'][0,0]=0
                try:admit_rank_batch(bad,cfg)
                except ValueError as exc:
                    if 'whole action' not in str(exc):raise
                else:raise AssertionError('bad rank mask accepted')
                # All ranks accidentally receiving the same complete requests.
                same=[batch if rank==0 else None];dist.broadcast_object_list(same,src=0)
                try:admit_rank_batch(same[0],cfg)
                except ValueError as exc:
                    if 'duplicated' not in str(exc):raise
                else:raise AssertionError('duplicate rank inputs accepted')
                unequal=copy.deepcopy(cfg)
                small=batch[:8] if rank==3 else batch
                if rank==3:unequal.ppo_mini_batch_size=8;unequal.ppo_micro_batch_size_per_gpu=8
                try:admit_rank_batch(small,unequal)
                except ValueError as exc:
                    if 'equal complete groups' not in str(exc):raise
                else:raise AssertionError('unequal rank weights accepted')
                changed=copy.deepcopy(batch)
                if rank==3:changed.batch['old_log_probs'][0,0]+=.1
                try:admit_rank_batch(changed,cfg)
                except ValueError as exc:
                    if 'old action logprob' not in str(exc):raise
                else:raise AssertionError('changed old probabilities accepted')
            del model
        atomic_json(output/f'rank-{rank}.json',dict(passed=True,rank=rank,cases=cases,
            actor_sha256=ACTOR_SHA256,collective_bad_mask_rejected=True,duplicate_rank_requests_rejected=True,
            unequal_rank_weights_rejected=True,collective_changed_old_logprobs_rejected=True))
    except BaseException as exc:
        atomic_json(output/f'rank-{rank}-failure.json',dict(error=repr(exc)));raise
    finally:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--algorithm',choices=('grpo_v1','rloo_v1'),default='grpo_v1')
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        mp.spawn(worker,args=(str(a.output),a.algorithm),nprocs=4,join=True)
        ranks=[json.loads((a.output/f'rank-{i}.json').read_text()) for i in range(4)]
        assert all(r['passed'] for r in ranks)
        atomic_json(a.output/'result.json',dict(passed=True,ranks=ranks,world_size=4,algorithm=a.algorithm,
            requests_per_step=8,actions_per_step=64,steps_per_case=2,
            scope='Actual four-rank Gloo/DDP native typed verl actor, serialized DataProto scatter, synthetic float32 features. No audio/BF16/NPU/FSDP/Ray/vLLM lifecycle validation.'))
        atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise


if __name__=='__main__':main()
