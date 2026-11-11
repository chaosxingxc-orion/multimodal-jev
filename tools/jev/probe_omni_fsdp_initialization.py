#!/usr/bin/env python3
"""Two-rank, frozen full Thinker FSDP1 initialization and state integrity probe.

Run via torchrun only after separately verified NPU/device/resource readiness.
The standalone CLI has no forward, optimizer, rollout, export or retries.
An in-process continuation may execute a separately scoped frozen diagnostic
after full initialization verification and before communication teardown.
"""
import argparse
from datetime import timedelta
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_fsdp_loading import (
    WRAP_CLASSES, MetaParameterMaterializer, build_meta_thinker, load_cpu_thinker,
    require_buffers,
)

IMAGE_HF_SOURCE = 'a55c7951f8ad316ab3abeadb09c51bc577ae3ffb673b40ebb5f17abac1348340'


def tensor_identity(value):
    import torch
    if value.is_meta or not torch.isfinite(value).all():
        raise ValueError('materialized finite state required')
    # numpy() marks its backing storage non-resizable. On CPU, .cpu() and
    # .contiguous() may alias FSDP's original flat storage; always copy first.
    cpu = value.detach().to(device='cpu', copy=True).contiguous()
    return dict(shape=list(cpu.shape), dtype=str(cpu.dtype),
                sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())


def owned_parameters(module, fsdp_type, prefix=''):
    """Visit one FSDP unit, excluding already nested FSDP children."""
    for name, parameter in module.named_parameters(recurse=False):
        yield prefix+name, parameter
    for name, child in module.named_children():
        if not isinstance(child, fsdp_type):
            yield from owned_parameters(child, fsdp_type, prefix+name+'.')


def logical_path(path):
    return '.'.join(p for p in path.split('.') if p != '_fsdp_wrapped_module')


def run(args, rank, local_rank, out, *, after_verification=None):
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    begin = time.monotonic()
    torch.set_num_threads(8)
    progress = lambda phase, **kw: atomic_json(out/'progress.json',dict(
        phase=phase, heartbeat_unix=time.time(), **kw))
    if not torch.npu.is_available() or torch.npu.device_count() != 2:
        raise ValueError('exactly two externally authorized NPU devices required')
    if file_hash(inspect.getfile(hf)) != IMAGE_HF_SOURCE:
        raise ValueError('original fixed HF implementation required')
    torch.npu.set_device(local_rank)
    device = torch.device('npu', local_rank)
    torch.npu.reset_peak_memory_stats()
    read = lambda p: json.loads(p.read_text())
    preflight = read(args.preflight/'acceptance.json')
    if (file_hash(args.preflight/'acceptance.json') != args.preflight_hash
            or file_hash(args.preflight/'run/result.json') != preflight['result_sha256']):
        raise ValueError('registered independently accepted CPU preflight required')
    result = read(args.preflight/'run/result.json')
    if (not result['all_checkpoint_values_exact'] or not result['all_buffers_exact']
            or result['parameters'] != 31719205488):
        raise ValueError('complete original CPU checkpoint preflight required')
    for name, pin in result['artifacts'].items():
        if file_hash(args.preflight/'run'/name) != pin:
            raise ValueError('preflight artifact changed')
    expected_buffers = read(args.preflight/'run/buffers.json')
    progress('verify_original_model')
    # Rank0 performs payload hashing; both ranks authenticate config locally.
    if file_hash(args.model/'config.json') != result['model_files']['config.json']:
        raise ValueError('model config changed')
    if rank == 0:
        for name,pin in result['model_files'].items():
            if Path(name).name != name or file_hash(args.model/name) != pin:
                raise ValueError('original model asset changed')
    dist.init_process_group('hccl', timeout=timedelta(seconds=1800))
    try:
        progress('load_cpu' if rank == 0 else 'build_meta')
        if rank == 0:
            model, _ = load_cpu_thinker(args.model)
            expected = {n:tensor_identity(p) for n,p in model.named_parameters()}
            atomic_json(args.output/'original-parameters.json',expected)
        else:
            model = build_meta_thinker(read(args.model/'config.json'))
        require_buffers(model,expected_buffers)
        dist.barrier()
        expected = read(args.output/'original-parameters.json')
        # Release CPU hash copies already freed by tensor_identity. Rank1 owns
        # only meta parameters until the recursive wrapper materializes units.
        callback = MetaParameterMaterializer(device)
        progress('fsdp_initialization')
        wrapped = FSDP(model, auto_wrap_policy=get_fsdp_wrap_policy(model,config={
            'transformer_layer_cls_to_wrap':WRAP_CLASSES}), param_init_fn=callback,
            device_id=device, sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,
                                           buffer_dtype=torch.float32),
            sync_module_states=True, use_orig_params=False, forward_prefetch=False,
            limit_all_gathers=True)
        wrapped.eval()
        torch.npu.synchronize()
        initial_memory = dict(allocated=torch.npu.memory_allocated(),reserved=torch.npu.memory_reserved())
        units = [(logical_path(n),m) for n,m in wrapped.named_modules() if isinstance(m,FSDP)]
        if len(units) != 108 or len({n for n,m in units}) != 108:
            raise ValueError('exact original 108 disjoint FSDP units required')
        observed = {}; buffers = {}; unit_rows = []
        for name, unit in units:
            progress('verify_sharded_unit',unit=name or '<root>',completed=len(unit_rows))
            # Only one unit is gathered; never materialize the full model on
            # one NPU. All ranks execute identical unit order and collectives.
            with FSDP.summon_full_params(unit,recurse=False,writeback=False,rank0_only=False,offload_to_cpu=False):
                count = 0
                for relative,param in owned_parameters(unit.module,FSDP):
                    full = name+'.'+relative if name else relative
                    if full in observed or full not in expected:
                        raise ValueError('duplicate or unknown sharded parameter')
                    if param.requires_grad or param.grad is not None or param.dtype != torch.bfloat16:
                        raise ValueError('unexpected trainable or changed-dtype parameter')
                    actual = tensor_identity(param)
                    if actual != expected[full]:
                        raise ValueError('FSDP parameter differs from original CPU weight: '+full)
                    observed[full] = actual; count += param.numel()
                unit_rows.append(dict(name=name or '<root>',parameters=count))
        if set(observed) != set(expected) or sum(u['parameters'] for u in unit_rows) != 31719205488:
            raise ValueError('incomplete full-model FSDP parameter coverage')
        for name,buf in wrapped.named_buffers():
            full = logical_path(name)
            if full in buffers or full not in expected_buffers:
                raise ValueError('unknown or duplicate wrapped buffer')
            actual = tensor_identity(buf)
            target = {k:v for k,v in expected_buffers[full].items() if k != 'persistent'}
            if actual != target or buf.device != device:
                raise ValueError('buffer changed during initialization or gathering')
            buffers[full] = actual
        if set(buffers) != set(expected_buffers):
            raise ValueError('incomplete wrapped buffer coverage')
        torch.npu.synchronize()
        atomic_json(out/'parameters.json',observed)
        atomic_json(out/'result.json',dict(status='initialization_verified',rank=rank,local_rank=local_rank,
            device=str(device),visible_devices=torch.npu.device_count(),parameters=31719205488,
            units=unit_rows,buffers=buffers,materializer_events=callback.events,
            initial_memory=initial_memory,peak_allocated_bytes=torch.npu.max_memory_allocated(),
            peak_reserved_bytes=torch.npu.max_memory_reserved(),elapsed_seconds=time.monotonic()-begin,
            parameters_sha256=file_hash(out/'parameters.json'),
            original_parameters_sha256=file_hash(args.output/'original-parameters.json'),
            preflight_sha256=args.preflight_hash,weight_updates=0,forward_calls=0,
            scope='Two-rank frozen FSDP1 initialization and every unit weight/buffer integrity only; no forward, sampled probabilities, optimizer or task-quality acceptance.'))
        dist.barrier()
        # The standalone initialization CLI never supplies this continuation.
        # A separately authenticated diagnostic runner may consume this exact
        # fully verified model before the process group is destroyed.
        if after_verification is not None:
            after_verification(wrapped)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ('model','preflight','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight-hash',required=True)
    args=parser.parse_args()
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2 or rank not in (0,1) or rank!=local_rank:
        raise ValueError('single-node two-rank torchrun topology required')
    out=args.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    try:run(args,rank,local_rank,out)
    except BaseException as exc:
        atomic_json(out/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
    atomic_json(out/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
