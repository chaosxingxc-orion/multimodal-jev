#!/usr/bin/env python3
"""Candidate utility through real native verl DP4; synthetic features only."""
import argparse
import copy
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from multimodal_jev.baseline import atomic_json
from multimodal_jev.span_smoothed_utility import transform
from probe_span_native_actor_dp_cpu import Model,config,optimizer,snapshot


def items(model,step):
    gen=torch.Generator().manual_seed(918+step);rows=[]
    for i in range(8):
        k=(2,31,60)[i%3];inputs=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),
            input_features=torch.randn(1,1,3,generator=gen),input_features_mask=torch.ones(1,1,dtype=torch.long))
        positions=torch.tensor([[0]]);candidates=[list(range(k))]
        with torch.no_grad():old=model(inputs,positions,candidates)[1]
        rows.append(dict(request_id=f'{step}:{i}',policy_id=f'policy{step}',inputs=inputs,positions=positions,candidates=candidates,
            behavior=tuple(p.detach() for p in old),targets=torch.tensor([[(i+step)%k if i not in (4,5) else -100]]),
            supervised=torch.tensor([[i not in (4,5)]]),span_weights=torch.ones(1,1),request_weight=float(i+1)))
    return rows


def utility_loss(model,rows):
    total=sum(x['request_weight'] for x in rows if x['supervised'].item())
    for item in rows:
        if not item['supervised'].item():continue
        p=model(item['inputs'],item['positions'],item['candidates'])[1][0][0,item['targets'].item()].exp()
        tau=1/len(item['candidates'][0])**2
        loss=-torch.log((p+tau)/(1+tau))/torch.log(torch.tensor((1+tau)/tau))
        (loss*item['request_weight']/total).backward()


def worker(rank,output):
    from multimodal_jev.span_finite_actor import make_global_batch,build_actor
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
    output=Path(output);torch.set_num_threads(1);torch.manual_seed(317)
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only')
    dist.init_process_group('gloo',init_method=(output/'gloo').resolve().as_uri(),rank=rank,world_size=4,timeout=timedelta(seconds=90))
    try:
        initial=copy.deepcopy(Model().state_dict());cases=[]
        for micro in (1,2):
            ref=Model();ref.load_state_dict(initial);ropt=optimizer(ref)
            module=Model();module.load_state_dict(initial);model=DDP(module);opt=optimizer(model)
            for step in range(2):
                raw=items(ref,step);scaled,receipt=transform(raw)
                for a,b in zip(raw,scaled,strict=True):
                    assert a['inputs'] is b['inputs'] and a['behavior'] is b['behavior']
                    assert a['request_weight']==b['request_weight'] and torch.equal(a['span_weights'],torch.ones(1,1))
                # Independent nonlinear utility derivative, no PPO/scaling helper.
                ropt.zero_grad();utility_loss(ref,raw);gradients=[p.grad.clone() for p in ref.parameters()]
                torch.nn.utils.clip_grad_norm_(ref.parameters(),1.);ropt.step();expected=snapshot(ref,ropt,gradients)
                chunks=list(make_global_batch(scaled).split(2)) if rank==0 else None
                incoming=[None];dist.scatter_object_list(incoming,chunks,src=0);batch=incoming[0]
                cfg=config(micro,f'policy{step}');cfg.ppo_mini_batch_size=2
                actor=build_actor(cfg,model,opt);captured={};original=actor._optimizer_step
                def capture():
                    captured['gradient']=[p.grad.detach().clone() for p in model.parameters()];return original()
                actor._optimizer_step=capture
                with native_actor_cpu_device(actor):metrics=actor.update_policy(batch)
                actual=snapshot(model,opt,captured['gradient']);errors={}
                for key,values in expected.items():
                    errors[key]=max((a-b).abs().max().item() for a,b in zip(values,actual[key],strict=True))
                    for a,b in zip(values,actual[key],strict=True):torch.testing.assert_close(a,b,atol=3e-7,rtol=3e-5)
                cases.append(dict(update=step+1,micro=micro,errors=errors,admission=actor.last_admission,metrics=metrics,credit=receipt))
        atomic_json(output/f'rank-{rank}.json',dict(passed=True,rank=rank,cases=cases))
    except BaseException as exc:
        atomic_json(output/f'rank-{rank}-failure.json',dict(error=repr(exc)));raise
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        mp.spawn(worker,args=(str(a.output),),nprocs=4,join=True)
        import json
        ranks=[json.loads((a.output/f'rank-{i}.json').read_text()) for i in range(4)]
        atomic_json(a.output/'result.json',dict(passed=True,ranks=ranks,scope='Synthetic CPU FP32 DP4 actual verl gradient/master/Adam versus independent nonlinear utility; 2updates, micro1/2, mixed2/31/60support and fully masked rank. No NPU/audio/BF16 utility training or task gain.'))
        atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
