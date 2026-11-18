from enum import Enum
from types import SimpleNamespace as NS

import pytest
import torch

from tools.reports.context_segmented_worker_candidate import scheduler_lengths, install_segmented_worker


class State(Enum):
    PREFILL = 0
    CACHE = 1


def case():
    module = NS(attn=object())
    meta = NS(attn_state=State.PREFILL, causal=True, model_runner_type='pooling',
        num_actual_tokens=7, num_decodes=0, num_decode_tokens=0, num_prefills=2,
        seq_lens_list=[3, 4], actual_seq_lengths_q=[3, 7])
    context = NS(attn_metadata={'layer': meta}, no_compile_layers={'layer': module.attn},
        cudagraph_runtime_mode='NONE', dp_metadata=None, ubatch_slices=None, is_padding=None)
    return module, meta, context, torch.tensor([0, 1, 2, 0, 1, 2, 3])


def resolve(module, context, positions):
    return scheduler_lengths(module, context, positions, len(positions),
                             prefill_state=State.PREFILL, eager_mode='NONE')


def test_real_layer_identity_and_full_query_boundaries():
    module, meta, context, positions = case()
    assert resolve(module, context, positions) == [3, 4]
    context.is_padding = torch.zeros(7, dtype=torch.bool)
    assert resolve(module, context, positions) == [3, 4]
    context.attn_metadata['unrelated'] = object()
    assert resolve(module, context, positions) == [3, 4]


@pytest.mark.parametrize('fault', ['cache', 'decode', 'decode_tokens', 'chunk', 'wrong_end',
    'count', 'padding_tokens', 'noncausal', 'generation', 'graph', 'dp', 'ubatch', 'list_metadata',
    'wrong_layer', 'duplicate_layer', 'missing_layer', 'bool_length', 'bool_end', 'position',
    'padding_mask', 'bad_padding_shape'])
def test_scheduler_mismatch_fails_closed(fault):
    module, meta, context, positions = case()
    if fault == 'cache': meta.attn_state = State.CACHE
    if fault == 'decode': meta.num_decodes = 1
    if fault == 'decode_tokens': meta.num_decode_tokens = 1
    if fault == 'chunk': meta.seq_lens_list = [3, 5]
    if fault == 'wrong_end': meta.actual_seq_lengths_q = [2, 7]
    if fault == 'count': meta.num_prefills = 1
    if fault == 'padding_tokens': meta.num_actual_tokens = 6
    if fault == 'noncausal': meta.causal = False
    if fault == 'generation': meta.model_runner_type = 'generate'
    if fault == 'graph': context.cudagraph_runtime_mode = 'FULL'
    if fault == 'dp': context.dp_metadata = object()
    if fault == 'ubatch': context.ubatch_slices = []
    if fault == 'list_metadata': context.attn_metadata = [context.attn_metadata]
    if fault == 'wrong_layer': context.no_compile_layers = {'layer': object()}
    if fault == 'duplicate_layer': context.no_compile_layers['second'] = module.attn
    if fault == 'missing_layer': context.attn_metadata = {}
    if fault == 'bool_length': meta.seq_lens_list = [True, 6]
    if fault == 'bool_end': meta.actual_seq_lengths_q = [True, 7]
    if fault == 'position': positions[-1] = 5
    if fault == 'padding_mask': context.is_padding = torch.ones(7, dtype=torch.bool)
    if fault == 'bad_padding_shape': context.is_padding = torch.zeros(8, dtype=torch.bool)
    with pytest.raises(ValueError): resolve(module, context, positions)


def model_case():
    layers = [NS(self_attn=NS(attn=object(), qkv_proj=NS(tp_size=1),
                             _jev_packed_candidate=True, forward=lambda *a: None)) for _ in range(2)]
    audio = NS(_jev_short_padding=True, forward=lambda x, frames, tokens: x[:, :int(tokens[0])].t())
    config = NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
        scheduler_config=NS(enable_chunked_prefill=False), cache_config=NS(enable_prefix_caching=False),
        model_config=NS(dtype=torch.bfloat16, enforce_eager=True), quant_config=None, lora_config=object())
    return NS(vllm_config=config, pooler=NS(memory_profiling=False), _jev_base_weights_loaded=True,
              _jev_fixed_base_candidate=True, audio_tower=audio, language_model=NS(model=NS(layers=layers)))


def test_installer_validates_all_layers_before_mutation_and_keeps_audio_split():
    model = model_case(); layers = model.language_model.model.layers
    original = [layer.self_attn.forward for layer in layers]; old_audio = model.audio_tower.forward
    layers[-1].self_attn.qkv_proj.tp_size = 2
    with pytest.raises(ValueError): install_segmented_worker(model)
    assert [layer.self_attn.forward for layer in layers] == original
    assert model.audio_tower.forward is old_audio
    layers[-1].self_attn.qkv_proj.tp_size = 1
    receipt = install_segmented_worker(model)
    assert receipt['layers'] == 2 and not receipt['training_dispatch_allowed']
    x = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    out = model.audio_tower.forward(x, torch.tensor([3, 4]), torch.tensor([1, 2]))
    torch.testing.assert_close(out, torch.cat([x[:, :1].t(), x[:, 3:5].t()]))
    with pytest.raises(ValueError): install_segmented_worker(model)


@pytest.mark.parametrize('fault', ['chunk', 'cache', 'profiling', 'precision', 'unloaded', 'unaligned'])
def test_installer_rejects_unsupported_runtime(fault):
    model = model_case()
    if fault == 'chunk': model.vllm_config.scheduler_config.enable_chunked_prefill = True
    if fault == 'cache': model.vllm_config.cache_config.enable_prefix_caching = True
    if fault == 'profiling': model.pooler.memory_profiling = True
    if fault == 'precision': model.vllm_config.model_config.dtype = torch.float32
    if fault == 'unloaded': model._jev_base_weights_loaded = False
    if fault == 'unaligned': model._jev_fixed_base_candidate = False
    with pytest.raises(ValueError): install_segmented_worker(model)
