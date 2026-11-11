import importlib.util
from pathlib import Path

import pytest
import torch

spec=importlib.util.spec_from_file_location('fsdp_initialization_probe',
    Path(__file__).parents[1]/'tools/jev/probe_omni_fsdp_initialization.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class Wrapped(torch.nn.Module):
    def __init__(self,module):
        super().__init__();self._fsdp_wrapped_module=module


def test_owned_parameters_partition_nested_units_without_full_gather():
    root=torch.nn.Module();root.head=torch.nn.Linear(3,4,bias=False)
    root.layers=torch.nn.ModuleList([Wrapped(torch.nn.Linear(3,3)),Wrapped(torch.nn.Linear(3,3))])
    root_params=dict(probe.owned_parameters(root,Wrapped))
    assert set(root_params)=={'head.weight'}
    all_params={**root_params}
    for i,unit in enumerate(root.layers):
        for name,param in probe.owned_parameters(unit._fsdp_wrapped_module,Wrapped):
            all_params[f'layers.{i}.{name}']=param
    assert len(all_params)==5
    assert sum(p.numel() for p in all_params.values())==sum(p.numel() for p in root.parameters())
    assert probe.logical_path('_fsdp_wrapped_module.layers.0._fsdp_wrapped_module')=='layers.0'
    assert probe.logical_path('_fsdp_wrapped_module')==''


def test_raw_parameter_identity_detects_values_shape_and_dtype():
    value=torch.tensor([[1,2],[3,4]],dtype=torch.bfloat16)
    initial=probe.tensor_identity(value)
    assert probe.tensor_identity(value.clone())==initial
    assert probe.tensor_identity(value.float())!=initial
    assert probe.tensor_identity(value.flatten())!=initial
    value[1,1]=5
    assert probe.tensor_identity(value)!=initial
    with pytest.raises(ValueError):probe.tensor_identity(torch.empty(2,device='meta'))
    with pytest.raises(ValueError):probe.tensor_identity(torch.tensor([float('nan')]))


def test_hash_does_not_make_fsdp_backing_storage_nonresizable():
    value=torch.arange(24,dtype=torch.bfloat16)
    original=value.clone()
    for view in [value,value[3:9],value.reshape(4,6).t()]:
        probe.tensor_identity(view)
    assert torch.equal(value,original)
    value.untyped_storage().resize_(0)
    assert value.untyped_storage().nbytes()==0
