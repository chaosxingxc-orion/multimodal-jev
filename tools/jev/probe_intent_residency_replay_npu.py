#!/usr/bin/env python3
"""Replay accepted D update2 with real-model offload/reload before its update.

Not a new learning arm. Exact post-state comparison and backend gate required
separately. No permission for concurrent vLLM execution is implied.
"""
import argparse
import inspect
import os
from pathlib import Path
import time
from multimodal_jev.baseline import atomic_json,file_hash
from train_intent_finite_npu import run
from queue_span_native_actor_npu import read


def validate(config):
    if (config['arm']!='D' or config['step']!=1 or config['global_requests']!=8
            or config['seed']!=317 or config['lr']!=1e-4
            or config['lora_execution_dtype']!='bfloat16'):
        raise ValueError('fixed accepted D second update replay required')
    reference=Path(config['replay_reference']);expected=read(reference/'config.json')
    for p in (reference/'config.json',reference/'launch.json'):
        if config['pins'].get(str(p))!=file_hash(p):raise ValueError('pinned reference required')
    ignored={'pins','replay_reference','replay_reference_suite'}
    if {k:v for k,v in config.items() if k not in ignored}!={k:v for k,v in expected.items() if k not in ignored}:
        raise ValueError('replay changes non-residency training factors')
    for rank in range(4):
        p=reference/'run'/f'rank-{rank}'/'optimizer-local.pt'
        if config['pins'].get(str(p))!=file_hash(p):raise ValueError('exact post-update reference shards required')


def roundtrip(model,optimizer,out,config,rank):
    import torch
    from verl.utils import fsdp_utils
    from multimodal_jev.span_residency import SpanResidency
    if file_hash(inspect.getfile(fsdp_utils))!='5dff5af2125de68d3cb15deb519f30a47e603a36e8f75d6a250a12c9e9941840':
        raise ValueError('FSDP movement implementation differs')
    if not optimizer.state:raise ValueError('real restored Adam moments required')
    optimizer.zero_grad(set_to_none=True)
    residency=SpanResidency(model,optimizer,
        offload_model=lambda m:fsdp_utils.offload_fsdp_model_to_cpu(m,empty_cache=False),
        load_model=fsdp_utils.load_fsdp_model_to_gpu,synchronize=torch.npu.synchronize)
    before=torch.npu.memory_allocated();residency.offload();torch.npu.empty_cache()
    offloaded=torch.npu.memory_allocated();residency.reload()
    atomic_json(out/'residency.json',dict(passed=True,rank=rank,
        exact_parameters_and_adam=True,original_tensor_devices_preserved=True,
        parameter_optimizer_references_preserved=True,allocated_before=before,
        allocated_offloaded=offloaded,allocated_restored=torch.npu.memory_allocated(),
        scope='Real audio model resident state boundary only; post-update/reference and independent backend verification still required.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=4 or int(os.environ['LOCAL_RANK'])!=rank:raise ValueError('single-node DP4 required')
    out=args.output/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
    try:
        cfg=read(args.config);validate(cfg);run(cfg,args.output,out,rank,before_update=roundtrip)
        atomic_json(out/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(out/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
