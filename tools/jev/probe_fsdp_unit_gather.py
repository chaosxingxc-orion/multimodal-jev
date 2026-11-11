#!/usr/bin/env python3
"""Two-rank CPU/Gloo test of the exact FSDP unit integrity traversal.

Small deterministic parameters on both ranks, no pretrained model or forward.
sync_module_states=False is required on CPU: not an HCCL/meta-sync proof.
"""
import argparse
from datetime import timedelta
import inspect
import os
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from probe_omni_fsdp_initialization import tensor_identity, owned_parameters, logical_path


def run(output, rank):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    torch.set_num_threads(2)
    begin=time.monotonic()

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Odd element counts ensure that sharding introduces padding.
            self.first=torch.nn.Linear(5,3,bias=True,dtype=torch.bfloat16)
            self.second=torch.nn.Linear(3,7,bias=False,dtype=torch.bfloat16)
            self.register_buffer('position',torch.tensor([.123456789,.33333333]),persistent=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=torch.nn.ModuleList([Block(),Block()])
            self.head=torch.nn.Linear(7,3,bias=False,dtype=torch.bfloat16)
            self.register_buffer('frequency',torch.tensor([.22222222,.77777777]),persistent=False)

    model=Model().eval().requires_grad_(False)
    with torch.no_grad():
        for i,p in enumerate(model.parameters()):
            p.copy_(torch.arange(p.numel()).reshape(p.shape)/8+i)
    expected={n:tensor_identity(p) for n,p in model.named_parameters()}
    expected_buffers={n:tensor_identity(b) for n,b in model.named_buffers()}
    expected_counts={'<root>':21,'blocks.0':39,'blocks.1':39}
    dist.init_process_group('gloo',timeout=timedelta(seconds=60))
    try:
        wrapped=FSDP(model,device_id=torch.device('cpu'),
            auto_wrap_policy=ModuleWrapPolicy({Block}),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32),
            use_orig_params=False,sync_module_states=False,forward_prefetch=False,limit_all_gathers=True)
        units=[(logical_path(n),m) for n,m in wrapped.named_modules() if isinstance(m,FSDP)]
        if [n or '<root>' for n,m in units]!=list(expected_counts):
            raise ValueError('actual nested unit order differs')
        shards_before={n or '<root>':tensor_identity(m._flat_param) for n,m in units}
        observed={};counts={};mismatch_detected=False
        for name,unit in units:
            with FSDP.summon_full_params(unit,recurse=False,writeback=False,rank0_only=False,offload_to_cpu=False):
                count=0
                for relative,param in owned_parameters(unit.module,FSDP):
                    full=name+'.'+relative if name else relative
                    if full in observed or tensor_identity(param)!=expected[full]:
                        raise ValueError('unit traversal did not reproduce original parameter')
                    observed[full]=tensor_identity(param);count+=param.numel()
                    # Distinguish exact equality from mere coverage/shape.
                    changed=param.detach().clone();changed.flatten()[0]+=1
                    mismatch_detected |= tensor_identity(changed)!=expected[full]
                counts[name or '<root>']=count
        if counts!=expected_counts or observed!=expected or not mismatch_detected:
            raise ValueError('complete original unit ownership and value identity required')
        buffers={logical_path(n):tensor_identity(b) for n,b in wrapped.named_buffers()}
        if buffers!=expected_buffers:
            raise ValueError('nonpersistent FP32 buffers changed')
        shards_after={n or '<root>':tensor_identity(m._flat_param) for n,m in units}
        if shards_after!=shards_before:
            raise ValueError('unit gathering changed local shards')
        # Each odd-sized unit has ceil(N/2) elements including one pad on rank1.
        if any(v['shape']!=[(expected_counts[n]+1)//2] for n,v in shards_after.items()):
            raise ValueError('actual two-rank local shards not observed')
        peer=[None,None]
        dist.all_gather_object(peer,dict(parameters=observed,buffers=buffers))
        if peer[0]!=peer[1]:raise ValueError('rank identities differ after gathering')
        atomic_json(output/'result.json',dict(status='passed',rank=rank,world_size=dist.get_world_size(),
            backend=dist.get_backend(),unit_counts=counts,parameters=observed,buffers=buffers,
            local_shards=shards_after,shards_unchanged=True,cross_rank_exact=True,
            mutation_detected=mismatch_detected,forward_calls=0,weight_updates=0,
            seconds=time.monotonic()-begin,torch_version=torch.__version__,
            source_sha256={'runner':file_hash(__file__),
                           'shared_traversal':file_hash(inspect.getfile(owned_parameters)),
                           'fsdp':file_hash(inspect.getfile(FSDP))},
            scope='Actual two-rank CPU/Gloo FSDP FULL_SHARD, synthetic BF16 parameters, nested unit gathering and FP32 buffer preservation. Both ranks start materialized; sync_module_states=False. No Omni weights, meta-rank broadcast, NPU/HCCL, forward or training.'))
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=2 or rank!=int(os.environ['LOCAL_RANK']):
        raise ValueError('exact single-node two-rank topology required')
    output=args.output/f'rank-{rank}';output.mkdir(parents=True,exist_ok=False)
    try:run(output,rank)
    except BaseException as exc:
        atomic_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(output/'terminal.json',dict(exit_code=0))
