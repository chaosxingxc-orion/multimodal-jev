"""Canonical PEFT LoRA export names, including nested native FSDP wrappers."""
import re


def adapter_key(name, *, native=False):
    name = name.replace('_fsdp_wrapped_module.', '')
    if native:
        if not name.startswith('backbone.language_model.'):
            raise ValueError('native Qwen ASR language LoRA required')
        name = 'base_model.model.model.' + name[len('backbone.'):]
    name = name.replace('.default.', '.')
    pattern = (r'base_model\.model\.model\.language_model\.layers\.\d+\.'
               r'self_attn\.(q_proj|v_proj)\.lora_[AB]\.weight')
    if not re.fullmatch(pattern, name):
        raise ValueError('canonical q/v PEFT LoRA weight name required: '+name)
    return name


def legacy_broken_key(canonical):
    """Only for identifying the known prefix-v1 export defect; never for saving."""
    if adapter_key(canonical) != canonical:
        raise ValueError('canonical export key required')
    return canonical.replace('.lora_A.weight', '.lora_Aweight').replace('.lora_B.weight', '.lora_Bweight')


def checkpoint_export_key(canonical, *, legacy_repair=False):
    """Select the declared source layout; never silently guess broken exports."""
    if type(legacy_repair) is not bool:
        raise ValueError('explicit legacy repair flag required')
    return legacy_broken_key(canonical) if legacy_repair else adapter_key(canonical)
