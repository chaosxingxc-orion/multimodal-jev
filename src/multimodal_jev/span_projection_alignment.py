"""Diagnostic separate Q/K/V base GEMMs with native vLLM LoRA retained."""
import torch


def split_qkv_base(x, weight, widths, bias=None):
    if (weight.ndim!=2 or len(widths)!=3 or any(type(w)!=int or w<1 for w in widths)
            or sum(widths)!=weight.shape[0] or x.shape[-1]!=weight.shape[1]
            or (bias is not None and bias.shape!=(weight.shape[0],))):
        raise ValueError('three aligned QKV projection slices required')
    weights=weight.split(widths,0)
    biases=(None,)*3 if bias is None else bias.split(widths,0)
    return torch.cat([torch.nn.functional.linear(x,w,b) for w,b in zip(weights,biases)],-1)


def split_base_native_lora_apply(module, x, bias=None):
    output=split_qkv_base(x,module.base_layer.weight,tuple(module.output_slices),bias)
    return module._apply_lora_to_output(x,output)


def install_separate_qkv(model):
    from types import MethodType
    from vllm.lora.layers.column_parallel_linear import MergedQKVParallelLinearWithLoRA
    layers=model.language_model.model.layers
    modules=[layer.self_attn.qkv_proj for layer in layers]
    for module in modules:
        if (not isinstance(module,MergedQKVParallelLinearWithLoRA) or module.tp_size!=1
                or module.base_layer.weight.dtype!=torch.bfloat16 or len(module.output_slices)!=3):
            raise ValueError('TP1 BF16 native merged QKV LoRA wrappers required')
    for module in modules:
        module.apply=MethodType(split_base_native_lora_apply,module)
    return dict(layers=len(layers),implementation='three separate base F.linear calls; original native LoRA add',
                scope='Diagnostic language QKV only; no adapter merging or activation substitution')


def separate_mlp_forward(module,x):
    gate_weight,up_weight=module.gate_up_proj.weight.chunk(2,0)
    gate=torch.nn.functional.linear(x,gate_weight)
    up=torch.nn.functional.linear(x,up_weight)
    # Preserve HF's BF16 materialization between SiLU and multiplication.
    intermediate=torch.nn.functional.silu(gate)*up
    return module.down_proj(intermediate)[0]


def install_separate_mlp(model):
    from types import MethodType
    layers=model.language_model.model.layers
    for layer in layers:
        module=layer.mlp
        if (module.gate_up_proj.tp_size!=1 or module.gate_up_proj.bias is not None
                or module.gate_up_proj.weight.dtype!=torch.bfloat16
                or hasattr(module.gate_up_proj,'lora_a_stacked')):
            raise ValueError('unadapted TP1 BF16 gate/up base required')
    for layer in layers:
        layer.mlp.forward=MethodType(separate_mlp_forward,layer.mlp)
    return dict(layers=len(layers),implementation='separate gate/up F.linear and BF16 SiLU then multiply',
                scope='Diagnostic unadapted language MLP; native down projection retained')
