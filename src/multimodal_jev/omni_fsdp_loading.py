"""CPU/meta Thinker preparation with BF16 parameters and original FP32 buffers.

Callers authenticate runtime and checkpoint before loading. These are pre-FSDP
helpers, not a distributed worker, collective implementation or training loader.
"""
import hashlib

BUFFER_NAMES = {
    'audio_tower.positional_embedding.positional_embedding',
    'visual.rotary_pos_emb.inv_freq',
    'model.rotary_emb.inv_freq',
    'model.rotary_emb.original_inv_freq',
}


def buffer_manifest(model):
    import torch
    result = {}
    for name, value in model.named_buffers():
        if value.is_meta or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('real finite FP32 positional buffers required')
        parent, leaf = name.rsplit('.', 1)
        if leaf not in model.get_submodule(parent)._non_persistent_buffers_set:
            raise ValueError('original nonpersistent positional buffers required')
        cpu = value.detach().cpu().contiguous()
        result[name] = dict(shape=list(cpu.shape), dtype=str(cpu.dtype), persistent=False,
                            sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest())
    if set(result) != BUFFER_NAMES:
        raise ValueError('complete original Thinker positional buffer roster required')
    return result


def require_buffers(model, expected):
    actual = buffer_manifest(model)
    if actual != expected:
        raise ValueError('positional buffer values or identity differ from reference')
    return actual


def parameter_manifest(model, *, device_type):
    import torch
    result = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if (param.device.type != device_type or param.dtype != torch.bfloat16
                or param.requires_grad or param.grad is not None):
            raise ValueError('frozen BF16 parameters on the declared device required')
        result[name] = dict(shape=list(param.shape), dtype=str(param.dtype), numel=param.numel())
    if not result:
        raise ValueError('nonempty Thinker parameters required')
    return result


def build_meta_thinker(full_config):
    import torch
    from accelerate import init_empty_weights
    from transformers import initialization
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

    if full_config.get('model_type') != 'qwen3_omni_moe':
        raise ValueError('original full Omni configuration required')
    config = Qwen3OmniMoeThinkerConfig(**full_config['thinker_config'])
    config._attn_implementation = 'sdpa'
    with init_empty_weights(include_buffers=False), initialization.no_init_weights():
        model = Qwen3OmniMoeThinkerForConditionalGeneration(config)
    before = buffer_manifest(model)
    if any(not p.is_meta for p in model.parameters()):
        raise ValueError('all parameters must be meta before dtype preparation')
    # Only storage-free parameters are changed; no optimizer or materialized
    # weights exist here. model.to(dtype) would also round the real buffers.
    for param in model.parameters():
        param.data = param.data.to(dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    parameter_manifest(model, device_type='meta')
    require_buffers(model, before)
    return model


def load_cpu_thinker(path):
    import torch
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

    model, info = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        device_map='cpu', attn_implementation='sdpa', output_loading_info=True)
    if (any(info.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs'))
            or any(not k.startswith(('talker.', 'code2wav.')) for k in info.get('unexpected_keys', []))):
        raise ValueError('complete original Thinker checkpoint loading required')
    model.eval().requires_grad_(False)
    parameter_manifest(model, device_type='cpu')
    buffer_manifest(model)
    return model, info
