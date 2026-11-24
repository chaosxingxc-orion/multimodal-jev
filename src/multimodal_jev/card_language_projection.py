"""Apply combined BF16 tiles to explicitly approved language projections."""
from contextlib import contextmanager
from types import MethodType

import torch

from .span_fixed_shape_linear import fixed_shape_linear


def combined(module, x):
    values = x[None] if x.ndim == 2 else x
    out = fixed_shape_linear(values, module.weight, module.bias, combined=True)
    return out[0] if x.ndim == 2 else out


@contextmanager
def combined_language(model, allowed):
    if model.training or not torch.is_inference_mode_enabled():
        raise ValueError('evaluation-only projection')
    if not allowed:
        raise ValueError('combined tiles require an explicit projection allowlist')
    modules = [(name, m) for name, m in model.backbone.language_model.named_modules()
               if type(m) is torch.nn.Linear]
    if not modules or any(m.weight.dtype != torch.bfloat16 for _, m in modules):
        raise ValueError('BF16 language projections required')
    saved = []
    matched = set()
    try:
        for name, module in modules:
            layer = name.split('.', 2)
            if len(layer) != 3 or layer[0] != 'layers' or not layer[1].isdigit() or layer[2] not in allowed:
                continue
            matched.add(layer[2])
            saved.append((module, vars(module).get('forward')))
            module.forward = MethodType(combined, module)
        if matched != set(allowed):
            raise ValueError('configured projections missing from installed language model')
        yield
    finally:
        for module, original in reversed(saved):
            if original is None:
                del module.forward
            else:
                module.forward = original
