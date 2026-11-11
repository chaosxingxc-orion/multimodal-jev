"""Experimental BF16 complete-K output-sharded projections.

Redistribute each frozen row-parallel weight once, retaining one quarter of
its elements per rank. Gather actual inputs and outputs at execution; do not
sum rounded partial products or inject reference activations.
"""
import torch


def output_weight_shard(full_weight, rank, size):
    if (type(rank) is not int or type(size) is not int or size!=4 or not 0<=rank<size
            or full_weight.ndim!=2 or full_weight.shape[0]%size):
        raise ValueError('four equal output weight shards required')
    return full_weight.chunk(size,0)[rank].contiguous().clone()


def complete_k_forward(module, x):
    from vllm.distributed import tensor_model_parallel_all_gather
    if x.dtype!=torch.bfloat16 or x.shape[-1]*4!=module.weight.shape[-1]:
        raise ValueError('BF16 local input shard required')
    gathered=tensor_model_parallel_all_gather(x,dim=-1)
    local=torch.nn.functional.linear(gathered,module.weight)
    output=tensor_model_parallel_all_gather(local,dim=-1)
    module._jev_complete_k_calls+=1
    module._jev_complete_k_max_input_bytes=max(module._jev_complete_k_max_input_bytes,
        gathered.numel()*gathered.element_size())
    module._jev_complete_k_max_output_bytes=max(module._jev_complete_k_max_output_bytes,
        output.numel()*output.element_size())
    return (output,None) if module.return_bias else output


def install_complete_k_projections(model):
    from types import MethodType
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.model_executor.layers.linear import RowParallelLinear
    if get_tensor_model_parallel_world_size()!=4 or type(model).__name__!='Qwen3ASRForSpanDecisionsTP4Diagnostic':
        raise ValueError('isolated TP4 diagnostic model required')
    rank=get_tensor_model_parallel_rank()
    selected=[]
    for i,layer in enumerate(model.language_model.model.layers):
        for suffix,module in (('o_proj',layer.self_attn.o_proj),('down_proj',layer.mlp.down_proj)):
            if (not isinstance(module,RowParallelLinear) or module.tp_size!=4 or module.bias is not None
                    or not module.input_is_parallel or not module.reduce_results
                    or module.weight.dtype!=torch.bfloat16 or hasattr(module,'lora_a_stacked')
                    or hasattr(module,'_jev_complete_k_calls')):
                raise ValueError('unadapted unbiased BF16 TP4 row projection required')
            selected.append((f'layer.{i}.{suffix}',module))
    receipts=[]
    for name,module in selected:
        old_shape=list(module.weight.shape)
        full=tensor_model_parallel_all_gather(module.weight,dim=-1)
        shard=output_weight_shard(full,rank,4)
        full_bytes=full.numel()*full.element_size()
        del full
        # Deliberately no checkpoint saving/loading from this diagnostic model:
        # original vLLM loaders describe the old input-column partition.
        module.weight=torch.nn.Parameter(shard,requires_grad=False)
        module._jev_complete_k_calls=0
        module._jev_complete_k_max_input_bytes=0
        module._jev_complete_k_max_output_bytes=0
        module.forward=MethodType(complete_k_forward,module)
        receipts.append(dict(name=name,old_shape=old_shape,new_shape=list(shard.shape),
                             transient_full_weight_bytes=full_bytes,local_weight_bytes=shard.numel()*shard.element_size()))
    return dict(rank=rank,tensor_parallel_size=4,diagnostic_only=True,modules=receipts,
                scope='Frozen weights redistributed once; BF16 complete-K local GEMM; input/output all-gather; no live weight reload')


def inspect_complete_k_projections(model):
    return {name:dict(calls=module._jev_complete_k_calls,
                     maximum_gathered_input_bytes=module._jev_complete_k_max_input_bytes,
                     maximum_gathered_output_bytes=module._jev_complete_k_max_output_bytes)
            for name,module in model.named_modules() if hasattr(module,'_jev_complete_k_calls')}
