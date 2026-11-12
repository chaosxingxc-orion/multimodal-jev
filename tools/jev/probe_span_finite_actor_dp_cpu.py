#!/usr/bin/env python3
"""Native verl finite-support updates on four Gloo ranks; synthetic CPU only."""
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
from probe_span_native_actor_dp_cpu import Model, config, optimizer, snapshot


def make_items(model,step):
    gen=torch.Generator().manual_seed(918+step)
    tasks=([list(range(60))],[[0,1]]*6,[list(range(60)),[0,1],[0,1,2,3,4]])
    items=[]
    for index in range(8):
        candidates=tasks[index%3];n=len(candidates)
        inputs=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),
            input_features=torch.randn(1,n,3,generator=gen),input_features_mask=torch.ones(1,n,dtype=torch.long))
        positions=torch.arange(n)[None]
        with torch.no_grad():old=tuple(p.detach().clone() for p in model(inputs,positions,candidates)[1])
        mask=torch.ones(1,n,dtype=torch.bool)
        if index in (1,4,5):mask[:]=False
        if index==2:mask[0,1]=False
        y=torch.tensor([[(index+step+j)%len(c) for j,c in enumerate(candidates)]])
        y[~mask]=-100
        weights=torch.full((1,n),.2/max(n-1,1));weights[0,-1]=1.
        items.append(dict(request_id=f'step{step}/request{index}',policy_id=f'policy{step}',
            inputs=inputs,positions=positions,candidates=candidates,behavior=old,targets=y,
            supervised=mask,span_weights=weights,request_weight=float(index+1)))
    return items


def effective(item):
    return item['request_weight'] if (item['span_weights']*item['supervised']).sum()>0 else 0.


def backward_reference(model,items,local_normalization=False):
    from multimodal_jev.span_finite_credit import reference_loss
    total=sum(effective(x) for x in items)
    for i,item in enumerate(items):
        weight=effective(item)
        if not weight:continue
        denominator=sum(effective(x) for x in items[(i//2)*2:(i//2)*2+2])*4 if local_normalization else total
        new=model(item['inputs'],item['positions'],item['candidates'])[1]
        loss=reference_loss(item['behavior'],new,item['targets'],item['supervised'],item['span_weights'])
        (loss*weight/denominator).backward()


def worker(rank,output):
    from multimodal_jev.span_finite_actor import make_global_batch,build_actor,admit_batch,_transport
    from multimodal_jev.span_native_actor import ACTOR_SHA256
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
    output=Path(output);torch.set_num_threads(1);torch.manual_seed(317)
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    dist.init_process_group('gloo',init_method=(output/'gloo-init').resolve().as_uri(),
        rank=rank,world_size=4,timeout=timedelta(seconds=90))
    try:
        initial=copy.deepcopy(Model().state_dict());cases=[]
        for micro in (1,2):
            module=Model();module.load_state_dict(initial);model=DDP(module);opt=optimizer(model)
            reference=Model();reference.load_state_dict(initial);refopt=optimizer(reference)
            for step in range(2):
                items=make_items(reference,step)
                chunks=list(make_global_batch(items).split(2)) if rank==0 else None
                incoming=[None];dist.scatter_object_list(incoming,chunks,src=0);batch=incoming[0]
                cfg=config(micro,f'policy{step}');cfg.ppo_mini_batch_size=2
                batch.meta_info['micro_batch_size']=micro
                refopt.zero_grad();backward_reference(reference,items)
                gradients=[p.grad.clone() for p in reference.parameters()]
                wrong=Model();wrong.load_state_dict(reference.state_dict());backward_reference(wrong,items,True)
                wrong_error=max(float((p.grad-g).abs().max()) for p,g in zip(wrong.parameters(),gradients,strict=True))
                if wrong_error<1e-4:raise AssertionError('negative control is not discriminating')
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
                rejections=[]
                for defect in ('mask','old_probability','quadrature','denominator','duplicate'):
                    bad=copy.deepcopy(batch)
                    if defect=='duplicate':
                        same=[bad if rank==0 else None];dist.broadcast_object_list(same,src=0);bad=same[0]
                    elif defect=='denominator':
                        # Internally consistent transported weights but wrong global population.
                        for i,payload in enumerate(bad.non_tensor_batch['multi_modal_inputs']):
                            payload['global_weight']+=1.
                            for key,value in _transport(payload).items():bad.batch[key][i]=value
                    elif rank==3:
                        if defect=='mask':bad.batch['response_mask'][0,0]=0
                        elif defect=='old_probability':bad.batch['old_log_probs'][0,0]+=.1
                        else:bad.batch['rollout_is_weights'][0,0]+=.1
                    try:admit_batch(bad,cfg)
                    except ValueError as exc:rejections.append(dict(defect=defect,error=str(exc)))
                    else:raise AssertionError(f'{defect} accepted')
                cases.append(dict(micro=micro,step=step+1,admission=actor.last_admission,
                    errors=errors,wrong_local_normalization_gradient_error=wrong_error,metrics=metrics,rejections=rejections))
            del model
        atomic_json(output/f'rank-{rank}.json',dict(passed=True,rank=rank,cases=cases,actor_sha256=ACTOR_SHA256))
    except BaseException as exc:
        atomic_json(output/f'rank-{rank}-failure.json',dict(error=repr(exc)));raise
    finally:dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    try:
        mp.spawn(worker,args=(str(args.output),),nprocs=4,join=True)
        ranks=[json.loads((args.output/f'rank-{i}.json').read_text()) for i in range(4)]
        assert all(r['passed'] for r in ranks)
        atomic_json(args.output/'result.json',dict(passed=True,ranks=ranks,world_size=4,
            requests_per_step=8,steps_per_case=2,
            scope='Actual native verl finite-support compute/update on four Gloo/DDP CPU ranks; unequal effective rank weights including a fully masked rank; synthetic FP32 only. No NPU/audio/BF16/Ray/vLLM validation.'))
        atomic_json(args.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise


if __name__=='__main__':main()
