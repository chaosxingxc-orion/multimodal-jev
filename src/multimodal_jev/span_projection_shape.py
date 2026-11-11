"""Diagnostic full-width BF16 GEMMs with zero-padded output shards.

Frozen local weights stay sharded. Padding is transient and adds computation;
this is not a production-efficient tensor parallel implementation.
"""
import torch


def full_width_local_linear(x, weight, rank, size=4):
    if (type(rank) is not int or type(size) is not int or size!=4 or not 0<=rank<size
            or weight.ndim!=2 or x.shape[-1]!=weight.shape[1]
            or x.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16):
        raise ValueError('BF16 aligned input and four output shards required')
    width=weight.shape[0]; start=rank*width
    padded=weight.new_zeros((width*size,weight.shape[1]))
    padded[start:start+width].copy_(weight)
    output=torch.nn.functional.linear(x,padded)
    return output[...,start:start+width].contiguous()


def full_width_qkv_apply(module,x,bias=None):
    if bias is not None:
        raise ValueError('unbiased QKV shape diagnostic required')
    weights=module.base_layer.weight.split(tuple(module.output_slices),0)
    output=torch.cat([full_width_local_linear(x,w,rank) for w,rank in
                      zip(weights,module.output_ids,strict=True)],-1)
    module._jev_full_width_calls+=1
    return module._apply_lora_to_output(x,output)


def full_width_mlp_forward(module,x):
    gate_weight,up_weight=module.gate_up_proj.weight.chunk(2,0)
    rank=module.gate_up_proj.tp_rank
    gate=full_width_local_linear(x,gate_weight,rank)
    up=full_width_local_linear(x,up_weight,rank)
    module._jev_full_width_calls+=1
    return module.down_proj(torch.nn.functional.silu(gate)*up)[0]


def install_full_width_projections(model):
    from types import MethodType
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    if get_tensor_model_parallel_world_size()!=4 or type(model).__name__!='Qwen3ASRForSpanDecisionsTP4Diagnostic':
        raise ValueError('isolated TP4 diagnostic required')
    rank=get_tensor_model_parallel_rank(); layers=model.language_model.model.layers
    for layer in layers:
        qkv=layer.self_attn.qkv_proj
        if (qkv.tp_size!=4 or tuple(qkv.output_ids)!=(rank,rank,rank)
                or qkv.base_layer.bias is not None or layer.mlp.gate_up_proj.bias is not None
                or any(not hasattr(m,'_jev_complete_k_calls') for m in
                       (layer.self_attn.o_proj,layer.mlp.down_proj))):
            raise ValueError('unique QKV shards and installed complete-K projections required')
    maximum=0
    for layer in layers:
        qkv=layer.self_attn.qkv_proj
        qkv.apply=MethodType(full_width_qkv_apply,qkv); qkv._jev_full_width_calls=0
        layer.mlp.forward=MethodType(full_width_mlp_forward,layer.mlp); layer.mlp._jev_full_width_calls=0
        for module in (layer.self_attn.o_proj,layer.mlp.down_proj):
            module._jev_full_width_rank=rank
            maximum=max(maximum,module.weight.numel()*4*module.weight.element_size())
        maximum=max(maximum,max(qkv.output_slices)*4*qkv.base_layer.weight.shape[1]*2,
                    layer.mlp.gate_up_proj.weight.numel()*4)
    return dict(rank=rank,tensor_parallel_size=4,layers=len(layers),diagnostic_only=True,
                maximum_single_padded_weight_bytes=maximum,
                scope='Zero-padded full output width QKV/gate/up/o/down BF16 GEMMs; native LoRA untouched; LM head unchanged')


def inspect_full_width_projections(model):
    return {name:module._jev_full_width_calls for name,module in model.named_modules()
            if hasattr(module,'_jev_full_width_calls')}
