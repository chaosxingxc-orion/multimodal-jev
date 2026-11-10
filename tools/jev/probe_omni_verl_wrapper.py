#!/usr/bin/env python3
"""Real verl actor wrapper and DataProto; tiny text model and synthetic audio fields."""
import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.omni_actor_forward import compute_log_probs
from multimodal_jev.verl_cpu_logprobs import native_actor_cpu_logprobs,native_actor_cpu_device


def run(out):
    import numpy as np
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.utils.device import get_device_name
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextModel
    torch.set_num_threads(2);torch.manual_seed(17)
    if get_device_name()!='cpu':raise ValueError('explicit CPU device required')
    cfg=Qwen3OmniMoeTextConfig(vocab_size=64,hidden_size=24,intermediate_size=48,
        num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=1,
        moe_intermediate_size=16,num_experts=4,num_experts_per_tok=2,
        max_position_embeddings=128,pad_token_id=0,
        rope_parameters={'rope_type':'default','rope_theta':1000000.,'mrope_section':[2,1,1]})
    cfg._attn_implementation='eager'
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__();self.text=Qwen3OmniMoeThinkerTextModel(cfg)
            self.head=torch.nn.Linear(24,64,bias=False);self.audio_seen=[]
        def forward(self,input_ids,attention_mask,position_ids,input_features,feature_attention_mask,use_cache):
            assert not use_cache
            self.audio_seen.append(dict(values=input_features[:,0,0].tolist(),
                                        lengths=feature_attention_mask.sum(-1).tolist()))
            return SimpleNamespace(logits=self.head(self.text(input_ids=input_ids,
                attention_mask=attention_mask,position_ids=position_ids,use_cache=False).last_hidden_state))
    model=Wrapper().eval().requires_grad_(False)
    ids=torch.tensor([[0,0,7,8,9,10,11,12],[0,3,4,5,6,13,14,0]])
    mask=ids.ne(0).long();pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,1)
    audio=np.empty(2,dtype=object)
    audio[0]=dict(input_features=torch.stack((torch.ones(4,7),torch.full((4,7),2.))),
                  feature_attention_mask=torch.tensor([[1,1,1,1,1,0,0],[1,1,1,1,1,1,0]]))
    audio[1]=dict(input_features=torch.full((1,4,7),3.),feature_attention_mask=torch.ones(1,7,dtype=torch.long))
    data=DataProto.from_dict(tensors=dict(input_ids=ids,attention_mask=mask,position_ids=pos,
        responses=ids[:,-3:].clone(),response_mask=mask[:,-3:].clone(),old_log_probs=torch.full((2,3),-2.)),
        non_tensors={'multi_modal_inputs':audio},meta_info=dict(audio_contract='sfc-jev-qwen3-omni-padded-v1',
            requires_remove_padding=False,requires_sequence_parallel_size=1))
    native=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,
        ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,
        entropy_from_logits_with_chunking=False,use_torch_compile=False,
        fsdp_config={'dtype':'bfloat16'},calculate_sum_pi_squared=False,entropy_checkpointing=False))
    dist.init_process_group('gloo',init_method=(out/'gloo-init').resolve().as_uri(),rank=0,world_size=1)
    try:
        actor=DataParallelPPOActor(native,model)
        old=data.batch['old_log_probs'].clone();meta=dict(data.meta_info);results=[];audio_calls=[]
        with native_actor_cpu_logprobs(),native_actor_cpu_device(actor):
            for size in (1,2):
                model.audio_seen=[]
                result=compute_log_probs(actor,data,micro_batch_size=size,pad_token_id=0)
                results.append(result);audio_calls.append(model.audio_seen)
        assert audio_calls==[[dict(values=[1.,2.],lengths=[5,6]),dict(values=[3.],lengths=[7])],
                             [dict(values=[1.,2.,3.],lengths=[5,6,7])]]
        torch.testing.assert_close(results[0],results[1],rtol=0,atol=.04)
        assert torch.equal(old,data.batch['old_log_probs']) and data.meta_info==meta
        assert all(p.grad is None for p in model.parameters())
        return dict(scope='Actual verl compute_log_prob and DataProto, random tiny Omni text model; '
            'synthetic audio tensors captured but not consumed by audio encoder. Explicit CPU logprob/device '
            'adapters; no pretrained weights, vLLM, NPU, FSDP or training.',
            actor_source_sha256=file_hash(inspect.getfile(DataParallelPPOActor)),
            helper_sha256=file_hash(Path(__file__).parents[2]/'src/multimodal_jev/omni_actor_forward.py'),
            micro_batch_sizes=[1,2],audio_calls=audio_calls,
            max_logprob_difference=float((results[0]-results[1]).abs().max()),atol=.04,
            old_probabilities_unchanged=True,metadata_unchanged=True,weight_updates=0,
            process_group=dict(backend='gloo',world_size=1))
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    try:result=run(a.output);atomic_json(a.output/'result.json',result);print(json.dumps(result,indent=2))
    except BaseException as exc:atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(a.output/'terminal.json',dict(exit_code=0))
