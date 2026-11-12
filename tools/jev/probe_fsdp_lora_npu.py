#!/usr/bin/env python3
"""Four-rank HCCL/FSDP BF16 PEFT LoRA/verl update and fixed-topology resume.

Synthetic tiny model only. No pretrained speech model, vLLM or TP parity claim.
FSDP owns FP32 LoRA masters and casts compute to BF16; no local master bridge.
"""
import argparse
import copy
from datetime import timedelta
import hashlib
import os
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def digest(t):
    import torch
    c=t.detach().to(device='cpu',copy=True).contiguous()
    return hashlib.sha256(c.view(torch.uint8).numpy().tobytes()).hexdigest()


def run(output, rank, residency_roundtrip=False):
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from peft import LoraConfig, get_peft_model
    from multimodal_jev.span_policy import sample_spans, joint_logprobs, verl_joint_loss
    from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3
    torch.set_num_threads(2)
    initialize_verl_preserving_qwen3()
    if torch.npu.device_count()!=4:
        raise ValueError('four isolated authorized NPUs required')
    torch.npu.set_device(rank)
    torch.npu.config.allow_internal_format=False
    device=torch.device('npu',rank)
    torch.npu.reset_peak_memory_stats()
    dist.init_process_group('hccl',timeout=timedelta(seconds=180))
    start=time.monotonic()
    def progress(stage):atomic_json(output/'progress.json',dict(stage=stage,updated_unix=time.time()))
    try:
        # Actual collective with rank-specific values, not merely group creation.
        probe=torch.tensor([rank+1.],device=device)
        dist.all_reduce(probe)
        if probe.item()!=10.:raise ValueError('HCCL reduction failed')
        progress('hccl_passed')
        execution=[]
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection=torch.nn.Linear(16,6,bias=False,dtype=torch.bfloat16)
            def forward(self,x):return self.projection(x)
        def build():
            torch.manual_seed(17)
            model=get_peft_model(Tiny(),LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,
                target_modules=['projection']),autocast_adapter_dtype=False).eval()
            # Persistent masters FP32; mixed precision must cast actual A/B to BF16.
            for name,p in model.named_parameters():
                if p.requires_grad:p.data=p.data.float()
            def inspect_compute(module,args):
                execution.append(dict(weight=str(module.weight.dtype),input=str(args[0].dtype)))
                if module.weight.dtype!=torch.bfloat16 or args[0].dtype!=torch.bfloat16:
                    raise ValueError('LoRA execution not BF16: '+str(execution[-1]))
            for name,m in model.named_modules():
                if isinstance(m,torch.nn.Linear) and ('lora_A.' in name or 'lora_B.' in name):
                    m.register_forward_pre_hook(inspect_compute)
            # Separate frozen BF16 base and FP32 trainable masters into distinct units.
            wrapped=FSDP(model,auto_wrap_policy=ModuleWrapPolicy({torch.nn.Linear}),
                device_id=device,sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,keep_low_precision_grads=False,cast_forward_inputs=True),
                use_orig_params=True,sync_module_states=True,limit_all_gathers=True)
            parameters=[p for p in wrapped.parameters() if p.requires_grad]
            if any(p.dtype!=torch.float32 for p in parameters):raise ValueError('FP32 masters required')
            optimizer=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=0.,foreach=False,fused=False)
            return wrapped,optimizer
        model,optimizer=build()
        names=[n for n,p in model.named_parameters()]
        frozen={n:digest(p) for n,p in model.named_parameters() if not p.requires_grad}
        initial={n:digest(p) for n,p in model.named_parameters() if p.requires_grad}
        local_counts={n:p.numel() for n,p in model.named_parameters()}
        x=(torch.arange(32,dtype=torch.float32).reshape(2,16)/32+rank/8).to(device,dtype=torch.bfloat16)
        losses=[]
        def step(m,opt,index):
            opt.zero_grad(set_to_none=True)
            logits=m(x).float().reshape(2,3,2)
            probabilities=tuple(logits[:,j].log_softmax(-1) for j in range(3))
            saved=tuple(p.detach().cpu() for p in probabilities)
            actions,old=sample_spans(saved,32,torch.Generator().manual_seed(1700+rank*10+index))
            targets=torch.tensor([[0,1,0],[1,0,1]])
            rewards=(actions==targets[:,None,:]).float().mean(-1)
            joint=joint_logprobs(probabilities,actions.to(device))
            loss,_=verl_joint_loss(old.to(device),joint,rewards.to(device),[f'{rank}-a',f'{rank}-b'])
            loss.backward()
            grads=[p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
            if not grads or any(g.dtype!=torch.float32 or not torch.isfinite(g).all() for g in grads):
                raise ValueError('finite FP32 reduced gradients required')
            opt.step()
            return float(loss.detach())
        for index in range(2):losses.append(step(model,optimizer,index))
        progress('two_updates_complete')
        saved=dict(schema='synthetic-fsdp-local-v1',world_size=4,rank=rank,names=names,
            parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters()},
            optimizer=copy.deepcopy(optimizer.state_dict()),updates=2)
        torch.save(saved,output/'local-checkpoint.pt')
        with torch.no_grad():expected_before=model(x).detach().clone()
        losses.append(step(model,optimizer,2))
        with torch.no_grad():expected_after=model(x).detach().clone()
        uninterrupted={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        optimizer_expected=copy.deepcopy(optimizer.state_dict())
        restored,restored_optimizer=build()
        state=torch.load(output/'local-checkpoint.pt',map_location='cpu',weights_only=True)
        if state['world_size']!=dist.get_world_size() or state['rank']!=rank or state['names']!=[n for n,p in restored.named_parameters()]:
            raise ValueError('fixed-topology checkpoint mismatch')
        with torch.no_grad():
            for n,p in restored.named_parameters():p.copy_(state['parameters'][n].to(device))
        restored_optimizer.load_state_dict(state['optimizer'])
        residency_report=None
        if residency_roundtrip:
            import inspect
            from verl.utils import fsdp_utils
            from multimodal_jev.span_residency import SpanResidency
            if file_hash(inspect.getfile(fsdp_utils))!='5dff5af2125de68d3cb15deb519f30a47e603a36e8f75d6a250a12c9e9941840':
                raise ValueError('audited FSDP movement implementation changed')
            # Materialize FSDP lazy state before moving its local shards.
            with torch.no_grad():restored(x)
            restored_optimizer.zero_grad(set_to_none=True)
            residency=SpanResidency(restored,restored_optimizer,
                offload_model=lambda m:fsdp_utils.offload_fsdp_model_to_cpu(m,empty_cache=False),
                load_model=fsdp_utils.load_fsdp_model_to_gpu,synchronize=torch.npu.synchronize)
            allocated_before=torch.npu.memory_allocated()
            residency.offload();torch.npu.empty_cache()
            allocated_offloaded=torch.npu.memory_allocated()
            residency.reload()
            residency_report=dict(exact_parameters_and_adam=True,original_tensor_devices_preserved=True,
                parameter_references_preserved=True,allocated_before=allocated_before,
                allocated_offloaded=allocated_offloaded,allocated_reloaded=torch.npu.memory_allocated(),
                scope='One tiny restored branch only; uninterrupted branch and HCCL remain allocated. Not vLLM capacity evidence.')
        with torch.no_grad():
            if not torch.equal(restored(x),expected_before):raise ValueError('restored outputs differ')
        step(restored,restored_optimizer,2)
        with torch.no_grad():
            if not torch.equal(restored(x),expected_after):raise ValueError('resumed next-step output differs')
        if any(not torch.equal(p.detach().cpu(),uninterrupted[n]) for n,p in restored.named_parameters()):
            raise ValueError('resumed masters differ')
        actual=restored_optimizer.state_dict()
        if actual['param_groups']!=optimizer_expected['param_groups']:raise ValueError('optimizer groups differ')
        for key,expected in optimizer_expected['state'].items():
            if any(not torch.equal(v.cpu(),actual['state'][key][k].cpu()) for k,v in expected.items()):
                raise ValueError('resumed Adam state differs')
        if frozen!={n:digest(p) for n,p in restored.named_parameters() if not p.requires_grad}:
            raise ValueError('frozen base changed')
        changed=sum(initial[n]!=digest(p) for n,p in restored.named_parameters() if p.requires_grad)
        total_changed=torch.tensor([changed],device=device);dist.all_reduce(total_changed)
        if not total_changed.item():raise ValueError('no LoRA parameters changed')
        with FSDP.summon_full_params(restored,writeback=False,rank0_only=False):
            exported={n:p.detach().to(device='cpu',dtype=torch.bfloat16).clone()
                for n,p in restored.named_parameters() if p.requires_grad}
        export_hashes={n:digest(p) for n,p in exported.items()}
        peers=[None]*4;dist.all_gather_object(peers,export_hashes)
        if any(p!=export_hashes for p in peers):raise ValueError('gathered adapters differ across ranks')
        if rank==0:torch.save(exported,output/'bf16-adapter.pt')
        result=dict(status='passed',rank=rank,world_size=4,backend=dist.get_backend(),
            updates=3,restored_at_update=2,exact_resume_output=True,exact_resume_masters=True,
            exact_resume_adam=True,frozen_base_unchanged=True,global_changed_shards=int(total_changed.item()),
            execution_dtypes=sorted({(x['weight'],x['input']) for x in execution}),
            local_parameter_counts=local_counts,adapter_export_hashes=export_hashes,cross_rank_export_exact=True,
            peak_allocated_bytes=torch.npu.max_memory_allocated(),losses=losses,seconds=time.monotonic()-start,
            checkpoint_sha256=file_hash(output/'local-checkpoint.pt'),
            scope='Synthetic four-rank NPU HCCL/FSDP PEFT BF16 compute, native FP32 masters/Adam and real verl loss. '
            'Fixed-topology local-shard resume only; no speech, Omni weights, vLLM TP4 or task-gain acceptance.')
        if residency_report is not None:result['residency_roundtrip']=residency_report
        atomic_json(output/'result.json',result)
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--residency-roundtrip',action='store_true')
    args=parser.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=4 or rank!=int(os.environ['LOCAL_RANK']):
        raise ValueError('single-node four-rank topology required')
    output=args.output/f'rank-{rank}';output.mkdir(parents=True,exist_ok=False)
    try:run(output,rank,args.residency_roundtrip)
    except BaseException as exc:
        atomic_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(output/'terminal.json',dict(exit_code=0))
