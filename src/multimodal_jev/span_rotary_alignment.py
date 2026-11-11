"""Diagnostic HF-order default RoPE; full-head NeoX layout only.

Build frequencies from configuration, never from recorded HF activations.
Explicit BF16 multiply then add matches the eager actor operation boundaries.
"""
import torch


def hf_order_rotary(module, positions, query, key=None):
    if (not module.is_neox_style or module.rotary_dim != module.head_size
            or positions.ndim != 1 or query.ndim != 2
            or query.shape[0] != positions.numel()):
        raise ValueError('full-head NeoX rotary and one sequence required')
    # HF constructs default inverse frequencies on CPU, then moves its buffer.
    dim = module.head_size
    inv = 1.0 / (module.base ** (torch.arange(0, dim, 2, device='cpu', dtype=torch.int64).float() / dim))
    freq = (inv.to(query.device)[None, :, None] @ positions[None, None, :].float()).transpose(1, 2)
    emb = torch.cat((freq, freq), -1)
    cos, sin = emb.cos().to(query.dtype)[0, :, None], emb.sin().to(query.dtype)[0, :, None]

    def rotate(x):
        shape = x.shape
        x = x.reshape(len(positions), -1, dim)
        half = torch.cat((-x[..., dim//2:], x[..., :dim//2]), -1)
        return (x * cos + half * sin).reshape(shape)

    return rotate(query), None if key is None else rotate(key)


def install_hf_order_rotary(model):
    from types import MethodType
    layers = model.language_model.model.layers
    modules = {id(layer.self_attn.rotary_emb): layer.self_attn.rotary_emb for layer in layers}
    for module in modules.values():
        if not module.is_neox_style or module.rotary_dim != module.head_size:
            raise ValueError('unsupported rotary configuration')
        module.forward = MethodType(hf_order_rotary, module)
    return dict(layers=len(layers), distinct_modules=len(modules),
                implementation='default CPU inverse frequencies, FP32 phase, BF16 multiply/add',
                scope='Diagnostic default full-head NeoX RoPE only')
