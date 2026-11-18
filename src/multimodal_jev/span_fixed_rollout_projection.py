"""Experimental TP1 fixed-tile base projections using live vLLM weights.

Native adapter selection and LoRA kernels remain active. Thus this alone does
not assert equivalence to the actor's fixed-tile LoRA leaves. No weight merge,
stored activations, or production profile changes are performed.
"""
from types import MethodType

import torch

from .span_fixed_shape_linear import fixed_shape_linear


def project(x, weight, bias=None):
    if x.ndim not in (2, 3):
        raise ValueError('complete 2D/3D projection input required')
    result = fixed_shape_linear(x[None] if x.ndim == 2 else x, weight, bias)
    return result[0] if x.ndim == 2 else result


def parallel_forward(module, x):
    bias = None if module.skip_bias_add else module.bias
    splits = getattr(module, '_jev_projection_splits', None)
    if splits is None:
        output = project(x, module.weight, bias)
    else:
        weights = module.weight.split(splits, 0)
        biases = [None] * len(splits) if bias is None else bias.split(splits, 0)
        output = torch.cat([project(x, w, b) for w, b in zip(weights, biases, strict=True)], -1)
    return (output, module.bias if module.skip_bias_add else None) if module.return_bias else output


def qkv_native_lora_apply(module, x, bias=None):
    widths = tuple(module.output_slices)
    weights = module.base_layer.weight.split(widths, 0)
    biases = [None] * 3 if bias is None else bias.split(widths, 0)
    base = torch.cat([project(x, w, b) for w, b in zip(weights, biases, strict=True)], -1)
    return module._apply_lora_to_output(x, base)


def fixed_mlp_forward(module, x):
    gate, up = module.gate_up_proj.weight.chunk(2, 0)
    intermediate = torch.nn.functional.silu(project(x, gate)) * project(x, up)
    return module.down_proj(intermediate)[0]


def fixed_head(processor, head, hidden, embedding_bias):
    if processor.head_dtype not in (None, hidden.dtype):
        raise ValueError('model dtype output head required')
    return project(hidden, head.weight, embedding_bias)


def install_fixed_rollout_projections(model):
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.lora.layers.column_parallel_linear import MergedQKVParallelLinearWithLoRA

    if model.vllm_config.parallel_config.tensor_parallel_size != 1:
        raise ValueError('TP1 fixed projection candidate required')
    if getattr(model, '_jev_fixed_base_candidate', False):
        raise ValueError('fresh candidate model required')
    selected = []
    qkv = []
    for name, module in model.named_modules():
        if name.startswith('audio_tower.') and isinstance(module, LinearBase):
            splits = (module.weight.shape[0] // 3,) * 3 if name.endswith('.qkv_proj') else None
            selected.append((name, module, splits))
    for i, layer in enumerate(model.language_model.model.layers):
        wrapper = layer.self_attn.qkv_proj
        if (not isinstance(wrapper, MergedQKVParallelLinearWithLoRA)
                or wrapper.tp_size != 1 or len(wrapper.output_slices) != 3):
            raise ValueError('native TP1 QKV LoRA wrappers required')
        qkv.append(wrapper)
        for name, module in [('o_proj', layer.self_attn.o_proj), ('down_proj', layer.mlp.down_proj)]:
            selected.append((f'language.{i}.{name}', module, None))
        if (layer.mlp.gate_up_proj.bias is not None
                or hasattr(layer.mlp.gate_up_proj, 'lora_a_stacked')):
            raise ValueError('unadapted unbiased gate/up required')
    if not selected or not qkv:
        raise ValueError('complete audio and language model required')
    for _, module, splits in selected:
        if (module.weight.dtype != torch.bfloat16 or getattr(module, 'tp_size', 1) != 1
                or (splits is not None and sum(splits) != module.weight.shape[0])):
            raise ValueError('unquantized BF16 TP1 aligned weights required')
    head = model.language_model.lm_head
    if head.weight.dtype != torch.bfloat16:
        raise ValueError('BF16 vocabulary head required')
    for _, module, splits in selected:
        module._jev_projection_splits = splits
        module.forward = MethodType(parallel_forward, module)
    for module in qkv:
        module.apply = MethodType(qkv_native_lora_apply, module)
    for layer in model.language_model.model.layers:
        layer.mlp.forward = MethodType(fixed_mlp_forward, layer.mlp)
    processor = model.language_model.logits_processor
    processor._apply_head = MethodType(fixed_head, processor)
    model._jev_fixed_base_candidate = True
    return dict(parallel_projections=[name for name, _, _ in selected], language_qkv=len(qkv),
                tile_rows=128, tile_batch=8, lora='native_vllm_unchanged',
                production_promoted=False, scope='fixed base/audio/head; native LoRA not yet aligned')
