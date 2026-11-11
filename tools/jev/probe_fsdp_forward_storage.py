#!/usr/bin/env python3
"""Measure FSDP1 root storage around frozen forwards and explicit reshard.

Two CPU/Gloo ranks, synthetic deterministic BF16 weights, no optimizer.
This isolates storage lifecycle; it does not claim Omni/verl/NPU parity.
"""
import argparse
from datetime import timedelta
import os
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash


def run(output, rank):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    torch.set_num_threads(2)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear=torch.nn.Linear(5,5,dtype=torch.bfloat16)
        def forward(self,x):
            return self.linear(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block=Block()
            self.head=torch.nn.Linear(5,3,dtype=torch.bfloat16)
        def forward(self,x):
            return self.head(self.block(x))

    model=Model().eval().requires_grad_(False)
    with torch.no_grad():
        for i,p in enumerate(model.parameters()):
            p.copy_((torch.arange(p.numel()).reshape(p.shape)%7-3)/16+i/32)
    inputs=torch.arange(10,dtype=torch.bfloat16).reshape(2,5)/8+rank/16
    with torch.no_grad():reference=model(inputs)
    dist.init_process_group('gloo',timeout=timedelta(seconds=60))
    try:
        wrapped=FSDP(model,device_id=torch.device('cpu'),auto_wrap_policy=ModuleWrapPolicy({Block}),
            sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=False,
            sync_module_states=False,forward_prefetch=False,limit_all_gathers=True)

        def storage():
            rows={}
            for name,module in wrapped.named_modules():
                if not isinstance(module,FSDP) or module._handle is None:continue
                param=module._handle.flat_param
                rows[name or '<root>']=dict(local_shape=list(param.shape),
                    active_storage_bytes=param.untyped_storage().nbytes(),
                    full_storage_bytes=param._full_param_padded.untyped_storage().nbytes()
                        if hasattr(param,'_full_param_padded') else None,
                    local_shard_bytes=param._local_shard.untyped_storage().nbytes()
                        if hasattr(param,'_local_shard') else None)
            return rows

        before=storage();observations=[]
        for index in range(2):
            with torch.no_grad():actual=wrapped(inputs)
            if not torch.equal(actual,reference):raise ValueError('frozen FSDP forward changed output')
            after_forward=storage()
            # This is the exact root cleanup used by the pinned verl FSDP worker.
            wrapped._handle.reshard(True)
            after_reshard=storage()
            if after_reshard['<root>']['full_storage_bytes']!=0:
                raise ValueError('explicit root reshard did not release full storage')
            observations.append(dict(iteration=index,output=actual.float().tolist(),
                exact_reference=True,after_forward=after_forward,after_reshard=after_reshard))
        if any(p.requires_grad or p.grad is not None for p in wrapped.parameters()):
            raise ValueError('unexpected trainable parameter or gradient')
        atomic_json(output/'result.json',dict(status='passed',rank=rank,world_size=2,
            torch_version=torch.__version__,before=before,observations=observations,
            reference=reference.float().tolist(),weight_updates=0,forward_calls=2,
            runner_sha256=file_hash(__file__),scope='Actual two-rank CPU/Gloo FSDP1 frozen forward and root storage lifecycle. Synthetic model, no Omni, actual verl actor, NPU/HCCL, probabilities or training.'))
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=2 or rank!=int(os.environ['LOCAL_RANK']):
        raise ValueError('single-node two-rank topology required')
    output=args.output/f'rank-{rank}';output.mkdir(parents=True,exist_ok=False)
    try:run(output,rank)
    except BaseException as exc:
        atomic_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(output/'terminal.json',dict(exit_code=0))
