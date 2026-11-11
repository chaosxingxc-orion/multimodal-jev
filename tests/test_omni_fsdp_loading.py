import importlib.util
from pathlib import Path

import pytest
import torch

from multimodal_jev.omni_fsdp_loading import (
    BUFFER_NAMES, buffer_manifest, parameter_manifest, require_buffers,
)


def model_fixture():
    model = torch.nn.Module()
    for name in sorted(BUFFER_NAMES):
        parts = name.split('.'); current = model
        for part in parts[:-1]:
            if not hasattr(current, part): current.add_module(part, torch.nn.Module())
            current = getattr(current, part)
        current.register_buffer(parts[-1], torch.tensor([.123456789, .33333333]), persistent=False)
    model.register_parameter('weight', torch.nn.Parameter(torch.empty(2,3,device='meta',dtype=torch.bfloat16),requires_grad=False))
    return model


def test_restored_dtype_does_not_hide_buffer_rounding():
    model = model_fixture()
    original = buffer_manifest(model)
    model.bfloat16().float()
    assert all(b.dtype == torch.float32 for b in model.buffers())
    with pytest.raises(ValueError, match='differ from reference'):
        require_buffers(model, original)


@pytest.mark.parametrize('kind', ['missing', 'extra', 'meta', 'nonfinite', 'bf16', 'persistent'])
def test_buffer_guard_rejects_changed_roster_or_storage(kind):
    model = model_fixture(); module = model.model.rotary_emb
    if kind == 'missing': del module.inv_freq
    if kind == 'extra': model.register_buffer('extra',torch.zeros(1),persistent=False)
    if kind == 'meta': module.inv_freq = module.inv_freq.to('meta')
    if kind == 'nonfinite': module.inv_freq[0] = float('nan')
    if kind == 'bf16': module.inv_freq = module.inv_freq.bfloat16()
    if kind == 'persistent': module._non_persistent_buffers_set.remove('inv_freq')
    with pytest.raises(ValueError): buffer_manifest(model)


def test_parameter_preflight_is_frozen_meta_bf16_only():
    model = model_fixture()
    assert parameter_manifest(model,device_type='meta')['weight']['numel'] == 6
    with pytest.raises(ValueError): parameter_manifest(model,device_type='cpu')
    model.weight.requires_grad_(True)
    with pytest.raises(ValueError): parameter_manifest(model,device_type='meta')
    model.weight.requires_grad_(False)
    model.weight.data = model.weight.data.float()
    with pytest.raises(ValueError): parameter_manifest(model,device_type='meta')


def test_independent_checkpoint_views_preserve_gate_up_expert_order():
    path = Path(__file__).parents[1]/'tools/jev/probe_omni_fsdp_loading.py'
    spec = importlib.util.spec_from_file_location('loading_probe',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    gate = torch.tensor([[[1.,2.]],[[11.,12.]]])
    up = torch.tensor([[[3.,4.]],[[13.,14.]]])
    down = torch.tensor([[[5.],[6.]],[[15.],[16.]]])
    prefix = 'model.layers.10.mlp.experts'
    params = {prefix+'.gate_up_proj':torch.cat([gate,up],dim=1),prefix+'.down_proj':down,
              'model.norm.weight':torch.tensor([1.,2.])}
    for i in range(2):
        for projection, original in [('gate',gate),('up',up),('down',down)]:
            observed = module.original_parameter_view(params,f'{prefix}.{i}.{projection}_proj.weight',1)
            assert torch.equal(observed,original[i])
    assert module.original_parameter_view(params,'model.norm.weight',1) is params['model.norm.weight']
