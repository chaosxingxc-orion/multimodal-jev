"""Diagnostic HF-order RMSNorm in native vLLM, no feature substitution.

Keep BF16 execution and explicit BF16 residual addition. Match HF's rounding
before weight multiplication, instead of an opaque fused norm/add kernel.
"""
import torch


def hf_order_rms_norm(module, x, residual=None):
    if getattr(module,'bias_loaded',False):
        raise ValueError('quantized/bias norm outside diagnostic scope')
    if residual is not None:
        x = x + residual
        residual = x
    input_dtype = x.dtype
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1,keepdim=True)+module.variance_epsilon)
    output = module.weight * value.to(input_dtype)
    return output if residual is None else (output,residual)


def install_hf_order_language_norm(model):
    from types import MethodType
    from vllm.model_executor.layers.layernorm import RMSNorm
    names = []
    for name,module in model.language_model.model.named_modules():
        if isinstance(module,RMSNorm):
            if module.weight.dtype != torch.bfloat16 or getattr(module,'bias_loaded',False):
                raise ValueError('BF16 unbiased norm required')
            module.forward = MethodType(hf_order_rms_norm,module)
            names.append(name)
    if not names:
        raise ValueError('no language norms installed')
    return {'modules':names,'implementation':'HF Qwen3 RMSNorm operation order; BF16 residual addition',
            'scope':'diagnostic intervention, not validated production backend'}
