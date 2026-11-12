#!/usr/bin/env python3
"""Actual pinned verl typed actor compute/update vs explicit complete loss.

Synthetic feature-conditioned model, CPU only. Checks multiple request groups,
ragged candidate sets, native gradient accumulation, Adam and stale rejection.
"""
import argparse
import copy
import inspect
import json
import pickle
from pathlib import Path
import torch
import torch.distributed as dist
from multimodal_jev.baseline import atomic_json,file_hash


def run(output):
    from omegaconf import OmegaConf
    from multimodal_jev.span_native_actor import make_batch,build_actor,ACTOR_SHA256
    from multimodal_jev.span_actor_update import record_rollout
    from multimodal_jev.span_policy import joint_logprobs,verl_joint_loss
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    torch.set_num_threads(2);torch.manual_seed(317)
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.head=torch.nn.Linear(3,60,bias=False)
        def forward(self,inputs,positions,candidates):
            # Distinct per-request evidence; neither actions nor gold reaches it.
            z=self.head(inputs['input_features'])
            return z,tuple(torch.log_softmax(z[:,j,ids],-1) for j,ids in enumerate(candidates))
    model=Model();initial=copy.deepcopy(model.state_dict());group=8;items=[]
    for index,candidates in enumerate(([list(range(60))],[[0,1]]*6,[[0,1,2],[3,4]])):
        n=len(candidates);inputs=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones((1,3),dtype=torch.long),
            input_features=torch.randn(1,n,3),input_features_mask=torch.ones((1,n),dtype=torch.long))
        positions=torch.arange(n)[None]
        with torch.no_grad():probs=model(inputs,positions,candidates)[1]
        rollout=record_rollout('policy0',[f'request{index}'],probs,group,torch.Generator().manual_seed(8+index))
        # Nonconstant detached engineering rewards; no efficacy assertion.
        rewards=torch.arange(group,dtype=torch.float32)[None]/(group-1)
        items.append(dict(inputs=inputs,positions=positions,candidates=candidates,rollout=rollout,rewards=rewards))
    results=[]
    dist.init_process_group('gloo',init_method=(output/'gloo-init').resolve().as_uri(),rank=0,world_size=1)
    try:
        for micro in (group,group*3):
            model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0.,foreach=False,fused=False)
            opt.zero_grad();losses=[]
            for item in items:
                new=joint_logprobs(model(item['inputs'],item['positions'],item['candidates'])[1],item['rollout'].actions)
                loss,_=verl_joint_loss(item['rollout'].old_joint_logprobs,new,item['rewards'],item['rollout'].request_ids)
                (loss/len(items)).backward();losses.append(float(loss.detach()))
            ref_grad=[p.grad.clone() for p in model.parameters()]
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            ref=dict(parameters=[p.detach().clone() for p in model.parameters()],
                exp_avg=[opt.state[p]['exp_avg'].clone() for p in model.parameters()],exp_avg_sq=[opt.state[p]['exp_avg_sq'].clone() for p in model.parameters()])
            model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0.,foreach=False,fused=False)
            config=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,ulysses_sequence_parallel_size=1,
                use_dynamic_bsz=False,use_prefix_grouper=False,entropy_from_logits_with_chunking=False,use_torch_compile=False,
                fsdp_config={'dtype':'float32'},calculate_sum_pi_squared=False,entropy_checkpointing=False,
                ppo_mini_batch_size=group*3,ppo_micro_batch_size_per_gpu=micro,ppo_epochs=1,use_kl_loss=False,
                calculate_entropy=False,entropy_coeff=0.,loss_agg_mode='token-mean',use_rollout_log_probs=True,
                policy_loss={'loss_mode':'vanilla'},clip_ratio=.2,clip_ratio_low=None,clip_ratio_high=None,clip_ratio_c=3.,
                global_batch_info={},grad_clip=1.,jev_group_size=group,jev_policy_id='policy0'))
            actor=build_actor(config,model,opt)
            # Exercise object aliasing across DataProto serialization, then
            # native select/split. This does not establish Ray transport parity.
            batch=pickle.loads(pickle.dumps(make_batch(items,group)))
            old=batch.batch['old_log_probs'].clone()
            batch.meta_info['micro_batch_size']=micro
            captured={}
            # Native optimizer zeroes gradients after step; capture summed
            # pre-clipping gradients by reading at optimizer hook boundary.
            original=actor._optimizer_step
            def capture():
                captured['gradients']=[p.grad.detach().clone() for p in model.parameters()]
                return original()
            actor._optimizer_step=capture
            with native_actor_cpu_device(actor):
                actual=actor.compute_log_prob(batch)['log_probs']
                torch.testing.assert_close(actual,old,atol=0,rtol=0)
                metrics=actor.update_policy(batch)
            errors={}
            for key,values in dict(gradients=ref_grad,**ref).items():
                observed=captured['gradients'] if key=='gradients' else [p.detach() for p in model.parameters()] if key=='parameters' else [opt.state[p][key] for p in model.parameters()]
                errors[key]=max(float((a-b).abs().max()) for a,b in zip(values,observed,strict=True))
                for a,b in zip(values,observed,strict=True):torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
            assert torch.equal(batch.batch['old_log_probs'],old)
            assert all(opt.state[p]['step'].item()==1 for p in model.parameters())
            results.append(dict(micro_batch_size=micro,passed=True,errors=errors,metrics=metrics,reference_mean_loss=sum(losses)/3))
        # Fresh model for stale-policy and partial-group refusal.
        model.load_state_dict(initial);config.jev_policy_id='wrong';actor=build_actor(config,model,opt)
        with native_actor_cpu_device(actor):
            try:actor.compute_log_prob(batch)
            except ValueError as exc:
                if 'stale' not in str(exc):raise
            else:raise AssertionError('stale policy accepted')
        config.jev_policy_id='policy0';actor=build_actor(config,model,opt)
        corrupted=pickle.loads(pickle.dumps(batch))
        corrupted.batch['old_log_probs'][0,0]+=.1
        with native_actor_cpu_device(actor):
            try:actor.update_policy(corrupted)
            except ValueError as exc:
                if 'old action logprob transport' not in str(exc):raise
            else:raise AssertionError('changed old probabilities accepted')
        reordered=batch[torch.tensor([1,0]+list(range(2,len(batch))))]
        with native_actor_cpu_device(actor):
            try:actor.compute_log_prob(reordered)
            except ValueError as exc:
                if 'ordered sample group' not in str(exc):raise
            else:raise AssertionError('reordered sample group accepted')
        config.ppo_micro_batch_size_per_gpu=4
        try:build_actor(config,model,opt)
        except ValueError:pass
        else:raise AssertionError('split request group accepted')
        return dict(passed=True,actual_native_compute_and_update=True,actor_sha256=file_hash(inspect.getfile(DataParallelPPOActor)),
            cases=results,old_probabilities_unchanged=True,stale_policy_rejected=True,partial_group_rejected=True,
            pickle_roundtrip=True,changed_old_probabilities_rejected=True,reordered_group_rejected=True,
            scope='Actual verl DataProto and native actor loop with explicit typed forward and scoped CPU placement; synthetic float32 model, not BF16 audio/NPU/FSDP/Ray lifecycle.')
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:atomic_json(a.output/'result.json',run(a.output));atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
