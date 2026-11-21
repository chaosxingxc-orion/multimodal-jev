"""Finite intervention for tracing native Q/V LoRA arithmetic against the actor."""
from functools import partial
from types import MethodType

import torch

from multimodal_jev.bounded_prefill_attention import bounded_prefill_kernel
from multimodal_jev.span_fixed_rollout_projection import project


def actor_order_qkv(module, inputs, bias=None):
    weights = module.base_layer.weight.split(module.output_slices, dim=0)
    biases = (None,) * len(weights) if bias is None else bias.split(module.output_slices, dim=0)
    outputs = []
    for weight, part_bias, adapter in zip(weights, biases, module._jev_actor_order_lora, strict=True):
        value = project(inputs, weight, part_bias)
        if adapter is not None:
            a, b, scale = adapter
            value = value + project(project(inputs, a), b) * scale
        outputs.append(value)
    return torch.cat(outputs, dim=-1)


def install(worker, *, attention_requests):
    """Freeze the authenticated current adapter for this one-policy probe."""
    model = worker.get_model()
    activation = worker._jev_activation
    if (activation.phase != 'ready' or model._jev_policy_id != activation.current['policy_id']
            or model.vllm_config.parallel_config.tensor_parallel_size != 1):
        raise ValueError('one authenticated TP1 policy required')
    bundle = activation.current
    layers = model.language_model.model.layers
    for index, layer in enumerate(layers):
        wrapper = layer.self_attn.qkv_proj
        device = wrapper.base_layer.weight.device
        prefix = f'language_model.layers.{index}.self_attn.'
        adapters = []
        for suffix in ('q_proj', 'k_proj', 'v_proj'):
            name = prefix + suffix
            if suffix == 'k_proj':
                if name in bundle['contract']['lora']:
                    raise ValueError('unexpected K adapter')
                adapters.append(None)
                continue
            spec = bundle['contract']['lora'][name]
            weights = tuple(bundle['tensors'][f'backbone.{name}.lora_{k}.default.weight'].to(device)
                            for k in ('A', 'B'))
            adapters.append((*weights, spec['scaling']))
        wrapper._jev_actor_order_lora = tuple(adapters)
        wrapper.apply = MethodType(actor_order_qkv, wrapper)
    if attention_requests is not None:
        from tools.reports import context_segmented_attention_candidate as attention
        attention.masked_row_varlen_kernel = partial(bounded_prefill_kernel, max_requests=attention_requests)
    return dict(policy_id=bundle['policy_id'], layers=len(layers), attention_requests=attention_requests)
