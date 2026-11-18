"""Opt-in bridge to pinned vLLM-Ascend complete-prefill scheduler metadata.

Install only AFTER the ordinary context profile and startup have completed on
a separate engine. This does not alter the frozen profile or admit a run.
"""
from types import MethodType

import torch

from tools.reports.context_segmented_attention_candidate import (
    segmented_attention_forward, validate_segments,
)
from tools.reports.context_segmented_audio_candidate import segmented_audio_forward


def scheduler_lengths(module, context, positions, total, *, prefill_state, eager_mode):
    """Bind this exact native Attention object to its current scheduler entry."""
    if (type(context.attn_metadata) is not dict or type(context.no_compile_layers) is not dict
            or context.cudagraph_runtime_mode != eager_mode
            or context.dp_metadata is not None or context.ubatch_slices is not None):
        raise ValueError('eager unpartitioned per-layer scheduler metadata required')
    keys = [name for name, layer in context.no_compile_layers.items() if layer is module.attn]
    if len(keys) != 1 or keys[0] not in context.attn_metadata:
        raise ValueError('unique actual attention layer metadata binding required')
    meta = context.attn_metadata[keys[0]]
    lengths = meta.seq_lens_list
    if (meta.attn_state != prefill_state or meta.causal is not True
            or meta.model_runner_type != 'pooling'
            or type(meta.num_actual_tokens) is not int or meta.num_actual_tokens != total
            or type(meta.num_decodes) is not int or meta.num_decodes != 0
            or type(meta.num_decode_tokens) is not int or meta.num_decode_tokens != 0
            or type(lengths) is not list or not lengths
            or any(type(n) is not int or n < 1 for n in lengths)):
        raise ValueError('pooling complete uncached causal prefills only')
    ends = []
    for n in lengths:
        ends.append(n + (ends[-1] if ends else 0))
    if (type(meta.num_prefills) is not int or meta.num_prefills != len(lengths)
            or type(meta.actual_seq_lengths_q) is not list
            or any(type(n) is not int for n in meta.actual_seq_lengths_q)
            or meta.actual_seq_lengths_q != ends or ends[-1] != total):
        raise ValueError('scheduled query lengths must equal full KV lengths without padding')
    validate_segments(positions, lengths, total)
    padding = context.is_padding
    if padding is not None and (not isinstance(padding, torch.Tensor)
            or padding.dtype != torch.bool or padding.shape != (total,) or padding.any()):
        raise ValueError('no graph or scheduler padding admitted')
    return list(lengths)


def worker_attention_forward(module, positions, hidden_states):
    from vllm.forward_context import get_forward_context
    from vllm.config import CUDAGraphMode
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    lengths = scheduler_lengths(module, get_forward_context(), positions, len(hidden_states),
        prefill_state=AscendAttentionState.PrefillNoCache, eager_mode=CUDAGraphMode.NONE)
    return segmented_attention_forward(module, positions, hidden_states, lengths)


def install_segmented_worker(model):
    """Preserve live projections/LoRA/head and original per-recording acoustics.

    Caller still needs a separately versioned owner with encode(list), request
    authentication, batch capacities and policy activation. No new owner here.
    """
    config = model.vllm_config
    if (config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1
            or config.scheduler_config.enable_chunked_prefill
            or config.cache_config.enable_prefix_caching
            or config.model_config.dtype != torch.bfloat16
            or not config.model_config.enforce_eager
            or config.quant_config is not None or config.lora_config is None
            or model.pooler.memory_profiling or not model._jev_base_weights_loaded):
        raise ValueError('loaded BF16 eager TP1 PP1 no-cache no-chunk native-LoRA context engine required')
    layers = model.language_model.model.layers
    if (not layers or not getattr(model, '_jev_fixed_base_candidate', False)
            or not getattr(model.audio_tower, '_jev_short_padding', False)
            or getattr(model, '_jev_segmented_candidate', False)):
        raise ValueError('fresh fully aligned context profile required')
    # Validate every layer before modifying any layer or audio method.
    for layer in layers:
        attention = layer.self_attn
        if (attention.qkv_proj.tp_size != 1 or not getattr(attention, '_jev_packed_candidate', False)
                or not hasattr(attention, 'attn')):
            raise ValueError('all original aligned TP1 language layers required')
    original_audio = model.audio_tower.forward
    def audio_forward(tower, input_features, feature_lens, aftercnn_lens):
        return segmented_audio_forward(original_audio, input_features, feature_lens, aftercnn_lens)
    for layer in layers:
        layer.self_attn.forward = MethodType(worker_attention_forward, layer.self_attn)
    model.audio_tower.forward = MethodType(audio_forward, model.audio_tower)
    model._jev_segmented_candidate = True
    return dict(layers=len(layers), audio='serial_original_recording_boundaries',
        language='single_varlen_kernel_scheduler_bound', head='existing_per_request_readout',
        production_promoted=False, training_dispatch_allowed=False)
