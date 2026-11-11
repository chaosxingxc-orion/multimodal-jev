#!/usr/bin/env python3
"""Actual verl actor + two-rank CPU FSDP using a synthetic feature-conditioned LM.

Explicit existing CPU placement/logprob adapters; no Omni weights or NPU claim.
"""
import argparse
from datetime import timedelta
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.omni_actor_forward import compute_log_probs
from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_device, native_actor_cpu_logprobs


def run(output,rank,probability_shards=False):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    torch.set_num_threads(2)
    if file_hash(inspect.getfile(DataParallelPPOActor))!='207ba1d50a0344181779256b9bb32c733032c136e405d1846a57f9c78cda4cce':
        raise ValueError('fixed native actor implementation required')

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(7,7,dtype=torch.bfloat16)
        def forward(self,x):return self.linear(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.block=Block();self.head=torch.nn.Linear(7,7,dtype=torch.bfloat16)
            self.calls=[]
        def forward(self,input_ids,attention_mask,position_ids,input_features,use_cache=False):
            if use_cache:raise ValueError('unexpected generation cache')
            self.calls.append(dict(batch=len(input_ids),features=input_features.float().tolist()))
            values=torch.nn.functional.one_hot(input_ids,num_classes=7).to(torch.bfloat16)
            values=values+input_features[:,None,:].to(torch.bfloat16)
            return SimpleNamespace(logits=self.head(self.block(values)))

    model=Model().eval().requires_grad_(False)
    with torch.no_grad():
        for i,p in enumerate(model.parameters()):p.copy_((torch.arange(p.numel()).reshape(p.shape)%5-2)/16+i/32)
    count=8 if probability_shards else 4
    ids=(torch.arange(count*6).reshape(count,6)+(0 if probability_shards else rank))%7
    response=ids[:,-3:].clone();mask=torch.tensor([[1,1,1],[1,1,0],[1,0,0],[1,1,1]]).repeat(count//4,1)
    response[mask==0]=0;ids[:,-3:]=response
    attention=torch.cat([torch.ones(count,3,dtype=torch.long),mask],dim=1)
    positions=torch.arange(6).expand(count,-1)
    features=torch.arange(count*7,dtype=torch.float32).reshape(count,7)/32+(0 if probability_shards else rank/16)
    multimodal=np.empty(count,dtype=object)
    for i in range(count):multimodal[i]={'input_features':features[i:i+1]}
    data=DataProto.from_dict(tensors=dict(input_ids=ids,attention_mask=attention,position_ids=positions,
        responses=response,response_mask=mask,old_log_probs=torch.full((count,3),-2.)),
        non_tensors={'multi_modal_inputs':multimodal},meta_info=dict(
            audio_contract='sfc-jev-qwen3-omni-padded-v1',requires_remove_padding=False,requires_sequence_parallel_size=1))
    with torch.no_grad():
        logits=model(ids,attention,positions,features).logits[:, -4:-1].float()
        expected=logits.log_softmax(-1).gather(-1,response.unsqueeze(-1)).squeeze(-1).masked_fill(~mask.bool(),0)
        changed=model(ids,attention,positions,torch.zeros_like(features)).logits[:, -4:-1].float()
        changed=changed.log_softmax(-1).gather(-1,response.unsqueeze(-1)).squeeze(-1)
    if torch.equal(expected[mask.bool()],changed[mask.bool()]):raise ValueError('features must affect reference probabilities')
    if probability_shards:
        from multimodal_jev.omni_probability_shards import shard_plan, recompute_shard, merge_shards
        from multimodal_jev.omni_probability_audit import compare
        from multimodal_jev.sfc_reward_batch import digest
        records=[];identities=np.empty(count,dtype=object)
        for i in range(count):
            length=int(mask[i].sum())
            records.append(dict(example_id=f'synthetic-row-{i}',decision_id='calls',replicate=0,
                token_ids=response[i,:length].tolist(),log_probs=expected[i,:length].tolist(),
                policy_sha256='a'*64,instruction_sha256='b'*64,finish_reason='length'))
            identities[i]=(f'synthetic-row-{i}','calls',0)
        data.non_tensor_batch['record_identity']=identities
        data.batch['old_log_probs']=expected.clone()
        data.meta_info.update(diagnostic_only=True,targets_read=False,synthetic_old_log_probs=False,
            waveform_boundary='vllm_hop',records_sha256=digest(records),policy_sha256='a'*64,
            instruction_sha256='b'*64)
        # Synthetic records exercise production contracts only. This flag is
        # not a claim of real rollout provenance; result scope remains explicit.
        plans={size:shard_plan(records,world_size=2,micro_batch_size=size) for size in (1,2)}
        protocol=dict(schema='omni-probability-comparison-v1',records_sha256=digest(records),
            policy_sha256='a'*64,instruction_sha256='b'*64,run_names=['micro1','micro2'],
            max_abs_logprob_error=0,token_ratio_lower=1,token_ratio_upper=1,
            max_between_run_logprob_error=0)
        atomic_json(output/'registered-plan.json',dict(plans=plans,protocol=protocol,
                    scope='Synthetic CPU integration fixture, not an NPU acceptance threshold.'))
    model.calls.clear()
    config=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,
        ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,
        entropy_from_logits_with_chunking=False,use_torch_compile=False,
        fsdp_config={'dtype':'bfloat16'},calculate_sum_pi_squared=False,entropy_checkpointing=False))
    dist.init_process_group('gloo',timeout=timedelta(seconds=60))
    try:
        wrapped=FSDP(model,device_id=torch.device('cpu'),auto_wrap_policy=ModuleWrapPolicy({Block}),
            sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=False,sync_module_states=False)
        actor=DataParallelPPOActor(config,wrapped)
        old=data.batch['old_log_probs'].clone();meta=dict(data.meta_info);observations=[];runs=[]
        with native_actor_cpu_device(actor),native_actor_cpu_logprobs(),torch.no_grad():
            for size in (1,2):
                begin=len(model.calls)
                if probability_shards:
                    plan=plans[size]
                    shard=recompute_shard(actor,data[[6,0,5,2,7,1,4,3]],records,
                        name=f'micro{size}',rank=rank,plan=plan,plan_sha256=digest(plan),pad_token_id=0)
                    shards=[None,None];dist.all_gather_object(shards,shard)
                    merged=merge_shards(records,list(reversed(shards)),name=f'micro{size}',
                        plan=plan,plan_sha256=digest(plan));runs.append(merged)
                    actual=torch.zeros(4,3)
                    for i,r in enumerate(shard['run']['responses']):
                        actual[i,:len(r['log_probs'])]=torch.tensor(r['log_probs'])
                    expected_local=expected[rank*4:(rank+1)*4]
                    features_local=features[rank*4:(rank+1)*4]
                else:
                    actual=compute_log_probs(actor,data,micro_batch_size=size,pad_token_id=0)
                    expected_local=expected;features_local=features
                calls=model.calls[begin:]
                if len(calls)!=4//size or any(c['batch']!=size for c in calls):raise ValueError('native actor microbatch schedule differs')
                if [f for call in calls for f in call['features']]!=features_local.tolist():raise ValueError('multimodal feature order changed')
                if not torch.equal(actual,expected_local):raise ValueError('actual actor probabilities differ from independent reference')
                if wrapped._handle.flat_param._full_param_padded.untyped_storage().nbytes()!=0:
                    raise ValueError('project adapter failed to release root full storage')
                observations.append(dict(micro_batch_size=size,calls=calls,log_probs=actual.tolist(),
                    reference_exact=True,root_full_storage_bytes=0))
        if not torch.equal(data.batch['old_log_probs'],old) or data.meta_info!=meta:
            raise ValueError('original probabilities or metadata mutated')
        if probability_shards:
            audit=compare(records,runs,protocol=protocol,protocol_sha256=digest(protocol))
            if not audit['numerical_gate_passed']:raise ValueError('complete merged probability audit failed')
            atomic_json(output/'probability-audit.json',audit)
            atomic_json(output/'merged-runs.json',runs)
        atomic_json(output/'result.json',dict(status='passed',rank=rank,world_size=2,observations=observations,
            feature_perturbation_detected=True,old_probabilities_unchanged=True,weight_updates=0,
            actual_actor=True,torch_version=torch.__version__,reference=expected.tolist(),
            probability_shards=probability_shards,synthetic_records=True,
            sources={'runner':file_hash(__file__),'actor':file_hash(inspect.getfile(DataParallelPPOActor)),
                'adapter':file_hash(inspect.getfile(compute_log_probs))},
            scope='Actual verl DataParallelPPOActor/DataProto and CPU/Gloo FSDP1; synthetic feature-conditioned model. Explicit CPU device/logprob adapters, not unmodified accelerator dispatch. No audio processor, Omni weights, NPU/HCCL or training.'))
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probability-shards',action='store_true')
    args=parser.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=2 or rank!=int(os.environ['LOCAL_RANK']):raise ValueError('two local ranks required')
    output=args.output/f'rank-{rank}';output.mkdir(parents=True,exist_ok=False)
    try:run(output,rank,args.probability_shards)
    except BaseException as exc:
        atomic_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(output/'terminal.json',dict(exit_code=0))
