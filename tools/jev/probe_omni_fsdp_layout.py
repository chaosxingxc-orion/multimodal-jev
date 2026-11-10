#!/usr/bin/env python3
"""Inspect real fixed-runtime meta Thinker and verl FSDP wrap policies.

No checkpoint tensors, NPU, distributed collectives, forward or optimizer.
Counts describe potential FSDP1 units, not measured allocation or support.
"""
import argparse
import inspect
import json
from pathlib import Path
import time

from multimodal_jev.baseline import atomic_json, file_hash


def layout(model, policy):
    """Count disjoint parameter ownership after recursive wrapping decisions."""
    groups = []

    def visit(module, name, root=False):
        total = sum(p.numel() for p in module.parameters())
        wrapped = 0
        if policy(module=module, recurse=True, nonwrapped_numel=total):
            for child_name, child in module.named_children():
                wrapped += visit(child, f'{name}.{child_name}' if name else child_name)
        remaining = total-wrapped
        if not root and policy(module=module, recurse=False, nonwrapped_numel=remaining):
            groups.append(dict(name=name, module_class=type(module).__name__, parameters=remaining,
                               bf16_bytes=2*remaining))
            return total
        if root:
            groups.append(dict(name='<root>', module_class=type(module).__name__, parameters=remaining,
                               bf16_bytes=2*remaining))
        return wrapped

    visit(model, '', root=True)
    if sum(g['parameters'] for g in groups) != sum(p.numel() for p in model.parameters()):
        raise ValueError('wrapping units do not partition the model parameters')
    return groups


def run(args):
    import torch
    import transformers
    from transformers import AutoConfig
    from transformers import initialization
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    start = time.monotonic()
    torch.set_num_threads(2)
    if file_hash(args.config) != args.config_hash:
        raise ValueError('registered original config required')
    raw = json.loads(args.config.read_text())
    full_config = AutoConfig.for_model(**raw)
    mappings = []
    for name in ('AutoModelForVision2Seq', 'AutoModelForCausalLM', 'AutoModelForImageTextToText', 'AutoModel'):
        cls = getattr(transformers, name, None)
        if cls is None:
            mappings.append(dict(auto_class=name, available=False, supports_full_config=False, resolved_class=None))
            continue
        supported = type(full_config) in cls._model_mapping.keys()
        target = cls._model_mapping[type(full_config)] if supported else None
        mappings.append(dict(auto_class=cls.__name__, available=True, supports_full_config=supported,
            resolved_class=target.__name__ if target is not None else None))
    config = Qwen3OmniMoeThinkerConfig(**raw['thinker_config'])
    config._attn_implementation = 'sdpa'
    with torch.device('meta'), initialization.no_init_weights():
        model = hf.Qwen3OmniMoeThinkerForConditionalGeneration(config).to(torch.bfloat16)
    if any(p.device.type != 'meta' for p in model.parameters()):
        raise ValueError('all parameters must remain storage-free meta tensors')
    default_error = None
    try:
        default = layout(model, get_fsdp_wrap_policy(model, config={}))
    except Exception as exc:
        default = []
        default_error = repr(exc)
    classes = ['Qwen3OmniMoeThinkerTextDecoderLayer', 'Qwen3OmniMoeAudioEncoderLayer', 'Qwen3OmniMoeVisionBlock']
    explicit = layout(model, get_fsdp_wrap_policy(model, config={'transformer_layer_cls_to_wrap': classes}))
    if sum(g['module_class'] == classes[0] for g in explicit) != 48:
        raise ValueError('all 48 original decoder layers must have separate units')
    report = dict(status='diagnostic_complete', config_sha256=args.config_hash,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__),
        source_sha256={inspect.getfile(hf): file_hash(inspect.getfile(hf)),
                       inspect.getfile(get_fsdp_wrap_policy): file_hash(inspect.getfile(get_fsdp_wrap_policy)),
                       'runner': file_hash(__file__)},
        full_model_type=full_config.model_type, auto_mappings=mappings,
        thinker_class=type(model).__name__, thinker_no_split_modules=model._no_split_modules,
        thinker_parameters=sum(p.numel() for p in model.parameters()),
        experts_implementation=model.config.text_config._experts_implementation,
        actual_module_classes=sorted({type(m).__name__ for m in model.modules()}),
        default_policy_error=default_error,
        default_units=default, explicit_wrap_classes=classes, explicit_units=explicit,
        default_largest_unit_bytes=max((g['bf16_bytes'] for g in default), default=None),
        explicit_largest_unit_bytes=max(g['bf16_bytes'] for g in explicit),
        model_weights_loaded=False, npu_used=False, weight_updates=0,
        elapsed_seconds=time.monotonic()-start,
        scope='Real meta Thinker and verl FSDP1 wrap policy traversal only. Not FSDP initialization, NPU/TP/HCCL support, peak memory or actor forward acceptance.')
    atomic_json(args.output/'result.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-hash', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json', dict(exit_code=1, error=repr(exc)))
        raise
    atomic_json(args.output/'terminal.json', dict(exit_code=0))
