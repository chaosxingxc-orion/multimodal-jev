import pytest
import torch

from multimodal_jev.omni_fsdp_loading import MetaParameterMaterializer


def fixture():
    model = torch.nn.Module()
    model.register_parameter('weight',torch.nn.Parameter(torch.empty(2,3,device='meta',dtype=torch.bfloat16),requires_grad=False))
    model.register_buffer('position',torch.tensor([.123456789,.33333333]),persistent=False)
    model.add_module('child',torch.nn.Linear(3,2,bias=False,device='meta',dtype=torch.bfloat16))
    model.child.requires_grad_(False)
    return model


def test_materialization_nonrecursive_and_preserves_exact_buffers():
    model=fixture();before=model.position.clone();callback=MetaParameterMaterializer('cpu')
    callback(model)
    assert model.weight.device.type=='cpu' and model.weight.dtype==torch.bfloat16
    assert model.child.weight.is_meta and not model.weight.requires_grad
    assert torch.equal(before,model.position) and model.position.dtype==torch.float32
    assert 'position' not in model.state_dict()
    model.weight.data.fill_(7)  # Simulate a later sync; reentry must not empty it.
    callback(model)
    assert torch.all(model.weight==7) and callback.events[-1]['allocated_parameters']==0
    callback(model.child)
    assert not model.child.weight.is_meta
    assert sum(e['allocated_parameters'] for e in callback.events)==12


def test_shared_parameters_and_buffers_remain_aliased():
    model=fixture()
    model.child.weight=model.weight
    model.child.register_buffer('position',model.position,persistent=False)
    callback=MetaParameterMaterializer('cpu')
    callback(model);callback(model.child)
    assert model.child.weight is model.weight
    assert model.child.position is model.position
    assert sum(e['allocated_parameters'] for e in callback.events)==6


@pytest.mark.parametrize('kind',['trainable','parameter_fp32','buffer_meta','buffer_bf16','buffer_nan'])
def test_invalid_state_rejected_before_mutation(kind):
    model=fixture();callback=MetaParameterMaterializer('cpu')
    if kind=='trainable':model.weight.requires_grad_(True)
    if kind=='parameter_fp32':model.weight.data=model.weight.data.float()
    if kind=='buffer_meta':model.position=model.position.to('meta')
    if kind=='buffer_bf16':model.position=model.position.bfloat16()
    if kind=='buffer_nan':model.position[0]=float('nan')
    old=model.weight
    with pytest.raises(ValueError):callback(model)
    assert model.weight is old and model.weight.is_meta and callback.events==[]


def test_meta_destination_rejected():
    with pytest.raises(ValueError):MetaParameterMaterializer('meta')


def test_actual_fsdp_materialization_traversal():
    from torch.distributed.fsdp._init_utils import _materialize_with_param_init_fn
    model=fixture();callback=MetaParameterMaterializer('cpu');before=model.position.clone()
    _materialize_with_param_init_fn(model,callback,set())
    assert len(callback.events)==2 and all(not p.is_meta for p in model.parameters())
    assert torch.equal(model.position,before)


def test_callback_does_not_retain_full_parameters_after_fsdp_replaces_them():
    import gc
    import weakref
    model=fixture();callback=MetaParameterMaterializer('cpu')
    original=model.weight
    callback(model);allocation=weakref.ref(model.weight)
    model.weight=torch.nn.Parameter(torch.zeros(1,dtype=torch.bfloat16),requires_grad=False)
    gc.collect()
    assert allocation() is None
    model.child.weight=original
    with pytest.raises(ValueError,match='alias crossed'):
        callback(model.child)
