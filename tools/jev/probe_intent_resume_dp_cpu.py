#!/usr/bin/env python3
"""Actual native finite actor: uninterrupted vs serialized resume on CPU DP4.

Synthetic FP32/Gloo test; does not establish audio/NPU/BF16 execution parity.
"""
import argparse
import copy
from datetime import timedelta
import json
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from multimodal_jev.baseline import atomic_json
from probe_span_native_actor_dp_cpu import Model,config,optimizer
from probe_span_finite_actor_dp_cpu import make_items


def worker(rank,root):
    from multimodal_jev.span_finite_actor import make_global_batch,build_actor
    from multimodal_jev.span_fsdp_resume import restore_local_shard
    from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device
    root=Path(root);torch.set_num_threads(1);torch.manual_seed(317)
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    dist.init_process_group('gloo',init_method=(root/'gloo-init').resolve().as_uri(),
        rank=rank,world_size=4,timeout=timedelta(seconds=90))
    try:
        model=DDP(Model());opt=optimizer(model)
        def update(module,optim,step,items):
            chunks=list(make_global_batch(items).split(2)) if rank==0 else None
            incoming=[None];dist.scatter_object_list(incoming,chunks,src=0);batch=incoming[0]
            cfg=config(1,f'policy{step}');cfg.ppo_mini_batch_size=2
            actor=build_actor(cfg,module,optim)
            with native_actor_cpu_device(actor):
                computed=actor.compute_log_prob(batch)['log_probs']
                torch.testing.assert_close(computed,batch.batch['old_log_probs'],atol=2e-7,rtol=2e-5)
                return actor.update_policy(batch)
        update(model,opt,0,make_items(model,0))
        path=root/f'rank-{rank}-state.pt'
        torch.save(dict(rank=rank,world_size=4,updates=1,names=[n for n,p in model.named_parameters()],
            parameters={n:p.detach().clone() for n,p in model.named_parameters()},optimizer=opt.state_dict()),path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        loaded=DDP(Model())
        with torch.no_grad():
            for n,p in loaded.named_parameters():p.copy_(state['parameters'][n].bfloat16().float())
        fresh=optimizer(loaded)
        lost_precision=any(not torch.equal(p,state['parameters'][n]) for n,p in loaded.named_parameters())
        if not lost_precision:raise ValueError('restore fixture lacks sub-BF16 precision')
        receipt=restore_local_shard(loaded,fresh,state,rank=rank,world_size=4,updates=1)
        # Control has exact master weights but deliberately starts Adam fresh.
        reset=DDP(copy.deepcopy(model.module));reset_opt=optimizer(reset)
        items=make_items(model,1)
        update(model,opt,1,items);update(loaded,fresh,1,items);update(reset,reset_opt,1,items)
        if any(not torch.equal(a,b) for a,b in zip(model.parameters(),loaded.parameters(),strict=True)):
            raise ValueError('resumed second native update differs')
        a,b=opt.state_dict(),fresh.state_dict()
        if a['param_groups']!=b['param_groups']:raise ValueError('Adam groups differ')
        for key,fields in a['state'].items():
            for field,value in fields.items():
                if not torch.equal(value,b['state'][key][field]):raise ValueError('resumed second Adam state differs')
        reset_error=max(float((a-b).abs().max()) for a,b in zip(model.parameters(),reset.parameters(),strict=True))
        if reset_error<1e-6:raise ValueError('optimizer reset negative control not discriminating')
        atomic_json(root/f'rank-{rank}.json',dict(passed=True,rank=rank,updates=2,restore=receipt,
            exact_next_parameters=True,exact_next_adam=True,sub_bf16_precision_restored=True,
            reset_optimizer_parameter_error=reset_error))
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    try:
        mp.spawn(worker,args=(str(a.output),),nprocs=4,join=True)
        ranks=[json.loads((a.output/f'rank-{r}.json').read_text()) for r in range(4)]
        atomic_json(a.output/'result.json',dict(passed=True,ranks=ranks,
            scope='Synthetic CPU Gloo DP4 native finite actor serialized master/Adam recovery; no NPU/audio/held-out improvement claim.'))
        atomic_json(a.output/'terminal.json',{'exit_code':0})
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
