#!/usr/bin/env python3
"""Two real Gloo ranks check pinned verl loss scaling on saved audio rollouts.

The trainable object is a small log-prob perturbation fixture, NOT an ASR model.
No model update, new rollout or distributed audio worker claim follows.
"""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json,file_hash


def worker(rank, output, rollout, core_sha256):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage,compute_policy_loss_vanilla
    from multimodal_jev.rl_run import load_rollout_run
    from multimodal_jev.paired_rollout import compose_rewards
    output=Path(output);torch.set_num_threads(1)
    try:
        if file_hash(inspect.getfile(compute_policy_loss_vanilla))!=core_sha256:
            raise ValueError('pinned verl core source mismatch')
        identity,examples,records=load_rollout_run(Path(rollout))
        if len(records)<4:raise ValueError('at least four real responses required')
        lookup={ex.example_id:ex for ex in examples}
        rewards,_=compose_rewards([lookup[r.example_id] for r in records],records,'pair-grpo',.2)
        width=max(len(r.token_ids) for r in records);n=len(records)
        mask=torch.zeros(n,width);old=torch.zeros_like(mask);token_rewards=torch.zeros_like(mask)
        for i,r in enumerate(records):
            mask[i,:len(r.token_ids)]=1;old[i,:len(r.token_ids)]=torch.tensor(r.log_probs)
            token_rewards[i,len(r.token_ids)-1]=rewards[i]
        advantages,_=compute_grpo_outcome_advantage(token_rewards,mask,
            np.array([r.example_id for r in records],dtype=object),norm_adv_by_std_in_grpo=True)
        # Deterministic features make each log-prob depend on shared parameters.
        index=torch.arange(n*width,dtype=torch.float32).reshape(n,width)
        features=torch.stack([torch.sin(index*.17),torch.cos(index*.31),
                              (index.remainder(7)-3)/7],dim=-1)
        class Perturbation(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.012,-.008,.004]))
            def forward(self,x):return (x*self.weight).sum(-1)
        def cfg(global_info):
            return OmegaConf.create({'clip_ratio':.2,'clip_ratio_low':None,'clip_ratio_high':None,
                'clip_ratio_c':3.,'global_batch_info':global_info})
        def loss(model,ids,global_info):
            return compute_policy_loss_vanilla(old[ids],old[ids]+model(features[ids]),
                advantages[ids],mask[ids],config=cfg(global_info))[0]
        reference=Perturbation();full=loss(reference,list(range(n)),{});full.backward()
        expected=reference.weight.grad.detach().clone()
        dist.init_process_group('gloo',init_method=(output/'gloo-init').resolve().as_uri(),
                                rank=rank,world_size=2)
        try:
            # Uneven row/token counts stress global normalization explicitly.
            cut=n//3;ids=list(range(0,cut)) if rank==0 else list(range(cut,n))
            local_tokens=int(mask[ids].sum());total=int(mask.sum())
            model=DistributedDataParallel(Perturbation())
            correct=loss(model,ids,{'dp_size':2,'batch_num_tokens':total})
            correct.backward();actual=model.module.weight.grad.detach().clone()
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
            # Accumulated microbatches: global normalization already accounts for size.
            model.zero_grad(set_to_none=True)
            chunks=[ids[i:i+3] for i in range(0,len(ids),3)]
            with model.no_sync():
                for part in chunks:
                    loss(model,part,{'dp_size':2,'batch_num_tokens':total}).backward()
            # Manual reduction after no_sync is explicit, since ranks have unequal chunk counts.
            accumulated=model.module.weight.grad.detach().clone()
            dist.all_reduce(accumulated);accumulated/=2
            torch.testing.assert_close(accumulated,expected,atol=2e-6,rtol=2e-5)
            # Negative control: mean each rank locally then average ranks.
            local=Perturbation();loss(local,ids,{}).backward()
            wrong=local.weight.grad.detach().clone();dist.all_reduce(wrong);wrong/=2
            negative_error=float((wrong-expected).abs().max())
            if negative_error<=1e-5:
                raise ValueError('negative local-mean control did not distinguish weighting')
            atomic_json(output/f'rank-{rank}.json',{'rank':rank,'world_size':2,
                'responses':len(ids),'effective_tokens':local_tokens,'global_tokens':total,
                'reference_gradient':expected.tolist(),'ddp_gradient':actual.tolist(),
                'accumulated_gradient':accumulated.tolist(),
                'ddp_max_error':float((actual-expected).abs().max()),
                'microbatch_max_error':float((accumulated-expected).abs().max()),
                'wrong_local_mean_max_error':negative_error,'microbatches':len(chunks),
                'rollout_sha256':file_hash(Path(rollout)/'rollouts.jsonl'),
                'policy_id':identity['policy_id']})
        finally:
            dist.destroy_process_group()
    except BaseException as exc:
        atomic_json(output/f'rank-{rank}-failure.json',{'error':repr(exc),'time':time.time()})
        raise


def main():
    import torch.multiprocessing as mp
    p=argparse.ArgumentParser();p.add_argument('--rollout',required=True)
    p.add_argument('--core-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:
        mp.spawn(worker,args=(str(a.output),a.rollout,a.core_sha256),nprocs=2,join=True)
        ranks=[json.loads((a.output/f'rank-{i}.json').read_text()) for i in range(2)]
        assert sum(r['effective_tokens'] for r in ranks)==ranks[0]['global_tokens']
        atomic_json(a.output/'result.json',{'status':'passed','ranks':ranks,
            'core_sha256':a.core_sha256,'atol':2e-6,'rtol':2e-5,
            'scope':'actual two-rank Gloo/DDP loss-gradient scaling using real saved rollout '
                    'rewards/logprobs and controlled perturbation parameters; no ASR forward, '
                    'ASR optimization, Ray/FSDP worker or efficacy validation'})
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',{'exit_code':1,'error':repr(exc),'finished_unix':time.time()})
        raise
    atomic_json(a.output/'terminal.json',{'exit_code':0,'finished_unix':time.time()})


if __name__=='__main__':main()
